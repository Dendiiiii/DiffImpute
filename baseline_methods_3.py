import sys
import rtdl
import zero
import torch
import random
import logging
import numpy as np
import torch.optim
import pandas as pd
import scipy.special
import sklearn.metrics
import sklearn.datasets
from tqdm.auto import tqdm
import sklearn.preprocessing
import sklearn.model_selection
import torch.nn.functional as F

import os

os.environ["NUMEXPR_MAX_THREADS"] = "32"

import rtdl
import zero
import math
import time
import json
import torch
import random
import warnings
import numpy as np
import torch.optim
import pandas as pd
import scipy.special
import torch.nn as nn
import sklearn.metrics
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import torch.nn.functional as F

from torch import Tensor
from tqdm.auto import tqdm
from functools import partial
from inspect import isfunction
from einops import rearrange, reduce
from rtdl import functional as rtdlF
from einops.layers.torch import Rearrange
from hyperimpute.plugins.imputers import Imputers
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast


def init_logger(filename, logger_name):
    """
    @brief:
        initialize logger that redirect info to a file just in case we lost connection to the notebook
    @params:
        filename: to which file should we log all the info
        logger_name: an alias to the logger
    """

    # get current timestamp
    # timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H-%M-%S")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Test
    logger = logging.getLogger(logger_name)
    logger.info("### Init. Logger {} ###".format(logger_name))
    return logger


dataset = "CA"  # Choose the dataset from "CA, HE, JA, HI, AL, YE, CO"

# Initialize logger
my_logger = init_logger("./{}_baseline_2_log.log".format(dataset), "ml_logger")

device = "cuda" if torch.cuda.is_available() else "cpu"
zero.improve_reproducibility(123456)

# Read the data
N_train = np.load(r"{}/N_train.npy".format(dataset))
N_val = np.load(r"{}/N_val.npy".format(dataset))
N_test = np.load(r"{}/N_test.npy".format(dataset))
y_train = np.load(r"{}/y_train.npy".format(dataset))
y_val = np.load(r"{}/y_val.npy".format(dataset))
y_test = np.load(r"{}/y_test.npy".format(dataset))

if dataset in ["CA", "YE"]:
    task_type = "regression"
    train = np.concatenate((N_train, N_val, N_test), axis=0)
    solution = np.concatenate((y_train, y_val, y_test), axis=0)
elif dataset in ["HI"]:
    task_type = "binclass"
    # Get rid of missing one missing value
    train_raw = np.concatenate((N_train, N_val, N_test), axis=0)
    solution_raw = np.concatenate((y_train, y_val, y_test), axis=0)
    train = train_raw[~torch.any(torch.from_numpy(train_raw).isnan(), dim=1)]
    solution = solution_raw[~torch.any(torch.from_numpy(train_raw).isnan(), dim=1)]
else:
    task_type = "multiclass"

    train = np.concatenate((N_train, N_val, N_test), axis=0)
    solution = np.concatenate((y_train, y_val, y_test), axis=0)

if task_type in ["multiclass", "binclass"]:
    X_all = train.astype("float32")
    y_all = solution.astype("float32" if task_type == "regression" else "int64")

    n_classes = int(max(y_all)) + 1 if task_type == "multiclass" else None

    train_size = int(len(train) - len(N_test))
    train_df_X = train[:train_size, :]
    train_df_Y = y_all[:train_size]
    test_df_X = train[train_size:, :]
    test_df_Y = y_all[train_size:]

    preprocess_X = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(
        train_df_X
    )

    norm_train_df_X = (
        torch.from_numpy(preprocess_X.transform(train_df_X))
        .type("torch.FloatTensor")
        .to(device)
    )
    norm_test_df_X = (
        torch.from_numpy(preprocess_X.transform(test_df_X))
        .type("torch.FloatTensor")
        .to(device)
    )

    norm_train_df_Y = torch.from_numpy(train_df_Y).to(device)
    norm_test_df_Y = torch.from_numpy(test_df_Y).to(device)

    x_start = norm_train_df_X.clone().type("torch.FloatTensor").to(device)

elif task_type == "regression":
    X_all = train.astype("float32")
    y_all = solution.astype("float32" if task_type == "regression" else "int64")
    n_classes = None

    df = np.concatenate([X_all, np.expand_dims(y_all, axis=1)], 1)

    train_size = int(len(df) * 0.8)
    train_df_X = df[:train_size, :-1]
    train_df_Y = df[:train_size, -1]
    test_df_X = df[train_size:, :-1]
    test_df_Y = df[train_size:, -1]

    Y_min = train_df_Y.min()
    Y_max = train_df_Y.max()

    preprocess_X = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(
        train_df_X
    )
    preprocess_Y = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(
        train_df_Y.reshape(-1, 1)
    )

    norm_train_df_X = torch.from_numpy(preprocess_X.transform(train_df_X)).to(device)
    norm_train_df_Y = torch.from_numpy(
        preprocess_Y.transform(train_df_Y.reshape(-1, 1))
    ).to(device)

    norm_test_df_X = torch.from_numpy(preprocess_X.transform(test_df_X)).to(device)
    norm_test_df_Y = torch.from_numpy(
        preprocess_Y.transform(test_df_Y.reshape(-1, 1))
    ).to(device)

    x_start = norm_train_df_X.clone()


def apply_model_ds(model, x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f"Looks like you are using a custom model: {type(model)}."
            " Then you have to implement this branch first."
        )


# not the best way to preprocess features, but enough for the demonstration
d_out = n_classes or 1

lr = 0.001
weight_decay = 1e-5

model_ds = rtdl.FTTransformer.make_default(
    n_num_features=norm_train_df_X.shape[1],
    cat_cardinalities=None,
    last_layer_query_idx=[
        -1
    ],  # it makes the model faster and does NOT affect its output
    d_out=d_out,
)

model_ds.to(device)
optimizer_ds = (
    model_ds.make_default_optimizer()
    if isinstance(model_ds, rtdl.FTTransformer)
    else torch.optim.AdamW(model_ds.parameters(), lr=lr, weight_decay=weight_decay)
)

loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == "binclass"
    else F.cross_entropy
    if task_type == "multiclass"
    else F.mse_loss
)


@torch.no_grad()
def evaluate(part, test_X, test_Y):
    model_ds.eval()
    prediction = []
    if part == "val":
        pass
    elif part == "test":
        X = test_X.to(device)
        Y = test_Y.to(device)

    for batch in zero.iter_batches(X, 1024):
        prediction.append(apply_model_ds(model_ds, batch))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = Y.cpu().numpy()

    if task_type == "binclass":
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == "multiclass":
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == "regression"
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * (
            Y_max - Y_min
        )
    return score


# Create a dataloader for batches of indices
# Docs: https://yura52.github.io/delu/reference/api/zero.data.IndexLoader.html
batch_size = 256
train_loader = zero.data.IndexLoader(len(norm_train_df_X), batch_size, device=device)

# Create a progress tracker for early stopping
# Docs: https://yura52.github.io/delu/reference/api/zero.ProgressTracker.html
progress = zero.ProgressTracker(patience=100)

my_logger.info(
    f'Test score before training: {evaluate("test", norm_test_df_X, norm_test_df_Y):.4f}'
)

save_path = f"{dataset}/sampling_results/downstream_model_epoch20.pt"
checkpoint = torch.load(save_path)
model_ds.load_state_dict(checkpoint["model_state_dict"])
optimizer_ds.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

model_ds.eval()
my_logger.info(
    f'Test score after training: {evaluate("test", norm_test_df_X, norm_test_df_Y):.4f}'
)

mask_mode = 1  # Choose a number between 0 and 1 for random mask and choose the number equal or greater than one for the column mask

methods = ["gain", "mice"]

mice_seed_loss = []
gain_seed_loss = []

mice_seed_pcor = []
gain_seed_pcor = []

mice_seed_score = []
gain_seed_score = []

if 0 < mask_mode < 1:
    for method in methods:
        plugin = Imputers().get(method)
        my_logger.info("=" * 50 + f"{method} Model Evaluation" + "=" * 50)
        for seed in tqdm(range(5), desc=f"Imputation time", total=5):
            torch.manual_seed(seed)  # control the randomness of the random mask
            masks = (
                torch.FloatTensor(
                    norm_test_df_X.shape[0], norm_test_df_X.shape[1]
                ).uniform_()
                > mask_mode
            ).to(device)
            masks[[0, -1], :] = True
            torch.save(
                masks, f"{dataset}/sampling_results/{mask_mode * 100}%_{seed}_mask.pt"
            )

            mask_test = norm_test_df_X.masked_fill(masks < 1, float("nan"))
            full_set = torch.cat([x_start.to(device), mask_test], 0)

            out = (
                torch.tensor(plugin.fit_transform(full_set.cpu().numpy().copy()).values)
                .to(device)
                .float()[-mask_test.shape[0] :, :]
            )
            torch.save(out, f"model_store/10%_{seed}_pred_tensor_{method}.pt")
            loss = np.nanmean(
                np.square(
                    (
                        out.masked_fill(masks, float("nan"))
                        - norm_test_df_X.masked_fill(masks, float("nan"))
                    ).cpu()
                )
            )

            ground_truth_values = np.array(
                norm_test_df_X.masked_fill(masks, float("nan"))
                .reshape(norm_test_df_X.shape[0] * norm_test_df_X.shape[1])
                .cpu()
            )

            imp_values = np.array(
                out.to(device)
                .masked_fill(masks, float("nan"))
                .reshape(out.shape[0] * out.shape[1])
                .cpu()
            )
            r, p = scipy.stats.pearsonr(
                imp_values[~np.isnan(imp_values)],
                ground_truth_values[~np.isnan(ground_truth_values)],
            )

            score = evaluate("test", out, norm_test_df_Y)

            if method == "mice":
                mice_seed_loss.append(loss)
                mice_seed_pcor.append(r)
                mice_seed_score.append(score)
            elif method == "gain":
                gain_seed_loss.append(loss)
                gain_seed_pcor.append(r)
                gain_seed_score.append(score)

    avg_mice_seed_loss = sum(mice_seed_loss) / len(mice_seed_loss)
    avg_gain_seed_loss = sum(gain_seed_loss) / len(gain_seed_loss)

    avg_mice_seed_pcor = sum(mice_seed_pcor) / len(mice_seed_pcor)
    avg_gain_seed_pcor = sum(gain_seed_pcor) / len(gain_seed_pcor)

    avg_mice_seed_score = sum(mice_seed_score) / len(mice_seed_score)
    avg_gain_seed_score = sum(gain_seed_score) / len(gain_seed_score)

    my_logger.info("=" * 50 + "Additional Methods Evaluation" + "=" * 50)
    my_logger.info(
        "MICE Imputation Loss:{0:.4f}\nGAIN Imputation Loss:{1:.4f}".format(
            avg_mice_seed_loss, avg_gain_seed_loss
        )
    )

    my_logger.info("=" * 50 + "Additional Methods Evaluation" + "=" * 50)
    my_logger.info(
        "MICE Pearson Correlation:{0:.4f}\nGAIN Pearson Correlation:{1:.4f}".format(
            avg_mice_seed_pcor, avg_gain_seed_pcor
        )
    )

    my_logger.info("=" * 50 + "Additional Methods Evaluation" + "=" * 50)
    my_logger.info(
        "MICE Imputation Score:{0:.4f}\nGAIN Imputation Score:{1:.4f}".format(
            avg_mice_seed_score, avg_gain_seed_score
        )
    )

elif mask_mode >= 1 and isinstance(mask_mode, int):
    for method in methods:
        plugin = Imputers().get(method)
        my_logger.info("=" * 50 + f"{method} Model Evaluation" + "=" * 50)

        random.seed(999)
        mask_col = random.sample(
            range(norm_test_df_X.shape[1]), min(norm_test_df_X.shape[1], mask_mode * 5)
        )
        for j in tqdm(range(5), desc=f"Imputation time", total=5):
            random.seed(j)
            col = random.sample(mask_col, mask_mode)
            masks = (
                torch.FloatTensor(
                    norm_test_df_X.shape[0], norm_test_df_X.shape[1]
                ).uniform_()
                > 0
            ).to(device)
            masks[:, [col]] = False
            mask_test = norm_test_df_X.masked_fill(masks < 1, float("nan"))
            full_set = torch.cat([x_start.to(device), mask_test], 0)

            out = (
                torch.tensor(plugin.fit_transform(full_set.cpu().numpy().copy()).values)
                .to(device)
                .float()[-mask_test.shape[0] :, :]
            )
            torch.save(out, f"model_store/10%_{j}_pred_tensor_{method}.pt")
            loss = np.nanmean(
                np.square(
                    (
                        out.masked_fill(masks, float("nan"))
                        - norm_test_df_X.masked_fill(masks, float("nan"))
                    ).cpu()
                )
            )

            ground_truth_values = np.array(
                norm_test_df_X.masked_fill(masks, float("nan"))
                .reshape(norm_test_df_X.shape[0] * norm_test_df_X.shape[1])
                .cpu()
            )

            imp_values = np.array(
                out.to(device)
                .masked_fill(masks, float("nan"))
                .reshape(out.shape[0] * out.shape[1])
                .cpu()
            )
            r, p = scipy.stats.pearsonr(
                imp_values[~np.isnan(imp_values)],
                ground_truth_values[~np.isnan(ground_truth_values)],
            )

            score = evaluate("test", out, norm_test_df_Y)

            if method == "mice":
                mice_seed_loss.append(loss)
                mice_seed_pcor.append(r)
                mice_seed_score.append(score)
            elif method == "gain":
                gain_seed_loss.append(loss)
                gain_seed_pcor.append(r)
                gain_seed_score.append(score)

    avg_mice_seed_loss = sum(mice_seed_loss) / len(mice_seed_loss)
    avg_gain_seed_loss = sum(gain_seed_loss) / len(gain_seed_loss)

    avg_mice_seed_pcor = sum(mice_seed_pcor) / len(mice_seed_pcor)
    avg_gain_seed_pcor = sum(gain_seed_pcor) / len(gain_seed_pcor)

    avg_mice_seed_score = sum(mice_seed_score) / len(mice_seed_score)
    avg_gain_seed_score = sum(gain_seed_score) / len(gain_seed_score)

    my_logger.info("=" * 50 + "Additional Methods Evaluation" + "=" * 50)
    my_logger.info(
        "MICE Imputation Loss:{0:.4f}\nGAIN Imputation Loss:{1:.4f}".format(
            avg_mice_seed_loss, avg_gain_seed_loss
        )
    )

    my_logger.info("=" * 50 + "Additional Methods Evaluation" + "=" * 50)
    my_logger.info(
        "MICE Pearson Correlation:{0:.4f}\nGAIN Pearson Correlation:{1:.4f}".format(
            avg_mice_seed_pcor, avg_gain_seed_pcor
        )
    )

    my_logger.info("=" * 50 + "Additional Methods Evaluation" + "=" * 50)
    my_logger.info(
        "MICE Imputation Score:{0:.4f}\nGAIN Imputation Score:{1:.4f}".format(
            avg_mice_seed_score, avg_gain_seed_score
        )
    )
