import rtdl
import zero

import math
import enum
import time
import json
import torch
import models
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import logging
import sys
import datetime


def init_logger(filename, logger_name):
    '''
    @brief:
        initialize logger that redirect info to a file just in case we lost connection to the notebook
    @params:
        filename: to which file should we log all the info
        logger_name: an alias to the logger
    '''

    # get current timestamp
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Test
    logger = logging.getLogger(logger_name)
    logger.info('### Init. Logger {} ###'.format(logger_name))
    return logger


dataset = "CA"  # Choose the dataset from "CA, HE, JA, HI, AL, YE, CO"

# Initialize logger
my_logger = init_logger("./{}_log.log".format(dataset), "ml_logger")

device = "cuda" if torch.cuda.is_available() else "cpu"
zero.improve_reproducibility(123456)

# Read the data
N_train = np.load(r'{}/N_train.npy'.format(dataset))
N_val = np.load(r'{}/N_val.npy'.format(dataset))
N_test = np.load(r'{}/N_test.npy'.format(dataset))
y_train = np.load(r'{}/y_train.npy'.format(dataset))
y_val = np.load(r'{}/y_val.npy'.format(dataset))
y_test = np.load(r'{}/y_test.npy'.format(dataset))


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

if task_type in ['multiclass', 'binclass']:
    X_all = train.astype('float32')
    y_all = solution.astype('float32' if task_type == 'regression' else 'int64')

    n_classes = int(max(y_all)) + 1 if task_type == 'multiclass' else None

    train_size = int(len(train) - len(N_test))
    train_df_X = train[:train_size, :]
    train_df_Y = y_all[:train_size]
    test_df_X = train[train_size:, :]
    test_df_Y = y_all[train_size:]

    preprocess_X = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_df_X)

    norm_train_df_X = torch.from_numpy(preprocess_X.transform(train_df_X)).type('torch.FloatTensor').to(device)
    norm_test_df_X = torch.from_numpy(preprocess_X.transform(test_df_X)).type('torch.FloatTensor').to(device)

    norm_train_df_Y = torch.from_numpy(train_df_Y).to(device)
    norm_test_df_Y = torch.from_numpy(test_df_Y).to(device)

    x_start = norm_train_df_X.clone().type('torch.FloatTensor').to(device)

elif task_type == 'regression':
    X_all = dataset['data'].astype('float32')
    y_all = dataset['target'].astype('float32' if task_type == 'regression' else 'int64')
    n_classes = None

    df = np.concatenate([X_all, np.expand_dims(y_all, axis=1)], 1)

    train_size = int(len(df) * 0.8)
    train_df_X = df[:train_size, :-1]
    train_df_Y = df[:train_size, -1]
    test_df_X = df[train_size:, :-1]
    test_df_Y = df[train_size:, -1]

    Y_min = train_df_Y.min()
    Y_max = train_df_Y.max()

    preprocess_X = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_df_X)
    preprocess_Y = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_df_Y.reshape(-1, 1))

    norm_train_df_X = torch.from_numpy(preprocess_X.transform(train_df_X)).to(device)
    norm_train_df_Y = torch.from_numpy(preprocess_Y.transform(train_df_Y.reshape(-1, 1))).to(device)

    norm_test_df_X = torch.from_numpy(preprocess_X.transform(test_df_X)).to(device)
    norm_test_df_Y = torch.from_numpy(preprocess_Y.transform(test_df_Y.reshape(-1, 1))).to(device)

    x_start = norm_train_df_X.clone()

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

timesteps = 1000

# define beta schedule
betas = cosine_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    if t[0] != 0:
        return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).to(device)
    else:
        return (sqrt_alphas_cumprod_t * x_start).to(device)


def undo(x_out, t):
    betas_t = extract(betas, t, x_out.shape)
    x_in_est = torch.sqrt(1 - betas_t) * x_out + torch.sqrt(betas_t) * torch.randn_like(x_out)
    return x_in_est


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * apply_model(model, x, time=t).squeeze(1) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop(model, gt, masks, shape):
    device = next(model.parameters()).device
    b = shape[0]  # Batch size
    train_loader = zero.data.IndexLoader(len(gt), b, device=device)

    # start from pure noise (for each example in the batch)
    img = torch.randn((len(gt), shape[1]), device=device)
    imgs = []

    times = get_schedule_jump_paper(t_T=500, jump_length=5, jump_n_sample=5)
    time_pairs = list(zip(times[:-1], times[1:]))

    for t_last, t_cur in tqdm(time_pairs, desc='Sampling loop time step', total=len(time_pairs)):
        for iteration, batch_idx in enumerate(train_loader):
            if t_cur < t_last:
                t_last_t = torch.full((batch_idx.shape[0],), t_last, device=device, dtype=torch.long)

                x_known = (q_sample(x_start=gt[batch_idx], t=torch.tensor([t_last]).to(device)) * masks[
                    batch_idx].int()).to(device)
                x_unknown = p_sample(model, img[batch_idx],
                                     torch.full((batch_idx.shape[0],), t_last, device=device, dtype=torch.long),
                                     t_last) * (1 - masks[batch_idx].int()).to(device)
                img[batch_idx] = x_known + x_unknown
            else:
                t_shift = 1
                img[batch_idx] = undo(x_out=img[batch_idx], t=torch.tensor([t_last + t_shift]).to(device))

        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, gt, masks):
    feature_size = gt.shape[1]
    batch_size = 2048  # gt.shape[0] # if you want to do a batch inference, you need to add data loader inside of the p_sample_loop function
    return p_sample_loop(model, gt, masks, shape=(batch_size, feature_size))

def apply_model(model, x_num, x_cat=None, time=None):
    if isinstance(model, FTTransformer):
        return model(x_num, x_cat, time)
    else:
        assert x_cat is None
        return model(x_num, time)

def apply_model_ds(model, x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )

def p_losses_fn(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = apply_model(denoise_model, x_noisy, time=t).squeeze(1)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def evaluate(part, test_X, test_Y):
    model_ds.eval()
    prediction = []
    if part == 'val':
        pass
    elif part == 'test':
        X = test_X.to(device)
        Y = test_Y.to(device)

    for batch in zero.iter_batches(X, 1024):
        prediction.append(apply_model_ds(model_ds, batch))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = Y.cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * (Y_max - Y_min)
    return score

# Instantiate the model
num_features = x_start.shape[1]
d_out = x_start.shape[1]
lr = 0.001
weight_decay = 1e-5

# Instantiate the MLP model
mlp_model = models.MLP.make_baseline(
    d_in=x_start.shape[1],
    d_layers=[128, 256, 128],
    dropout=0.1,
    d_out=d_out,
)

mlp_model.to(device)
mlp_optimizer = (
    mlp_model.make_default_optimizer()
    if isinstance(mlp_model, rtdl.FTTransformer)
    else torch.optim.AdamW(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)
)

# Instantiate the ResNet model
res_model = models.ResNet.make_baseline(
    d_in=x_start.shape[1],
    d_main=128,
    d_hidden=256,
    dropout_first=0.2,
    dropout_second=0.0,
    n_blocks=3,
    d_out=d_out,
)

res_model.to(device)
res_optimizer = (
    res_model.make_default_optimizer()
    if isinstance(res_model, rtdl.FTTransformer)
    else torch.optim.AdamW(res_model.parameters(), lr=lr, weight_decay=weight_decay)
)

# Instantiate the Transformer model
ft_model = models.FTTransformer.make_default(
    n_num_features=x_start.shape[1],
    cat_cardinalities=None,
    last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
    d_out=d_out
)

ft_model.to(device)
ft_optimizer = (
    ft_model.make_default_optimizer()
    if isinstance(ft_model, rtdl.FTTransformer)
    else torch.optim.AdamW(ft_model.parameters(), lr=lr, weight_decay=weight_decay)
)

# Instantiate the U-Net model
unet_model = models.UNet1D(num_features, d_out=d_out)
unet_model.to(device)
unet_optimizer = torch.optim.AdamW(unet_model.parameters(), lr=0.001)

models = [(unet_model, unet_optimizer, "U-Net"), (mlp_model, mlp_optimizer, "MLP"),
          (res_model, res_optimizer, "ResNet"), (ft_model, ft_optimizer, "FT")]


batch_size = 64
feature_size =  x_start.shape[1]
train_loader = zero.data.IndexLoader(len(x_start), batch_size, device=device, shuffle=True)

# Create a progress tracker for early stopping
# Docs: https://yura52.github.io/delu/reference/api/zero.ProgressTracker.html
progress = zero.ProgressTracker(patience=100)
# x_start = x_start.to(device)

for model, optimizer, model_type in models:
    if isinstance(model, models.MLP):
        my_logger.info('=' * 50 + 'MLP Model Training' + '=' * 50)
    elif isinstance(model, models.ResNet):
        my_logger.info('=' * 50 + 'ResNet Model Training' + '=' * 50)
    elif isinstance(model, models.FTTransformer):
        my_logger.info('=' * 50 + 'FT Transformer Model Training' + '=' * 50)
    else:
        my_logger.info('=' * 50 + 'U-Net Model Training' + '=' * 50)
    n_epochs = 20
    report_frequency = len(x_start) // batch_size // 1
    for epoch in range(1, n_epochs + 1):
        for iteration, batch_idx in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x_batch = x_start.to(device)[batch_idx]
            t = torch.randint(0, timesteps, (batch_idx.shape[0],), device=device).to(device)
            loss = p_losses_fn(model, x_batch, t, loss_type='huber')

            loss.backward()
            optimizer.step()
            if iteration % report_frequency == 0:
                my_logger.info(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

        # Save the model after each epoch
        save_path = f"model_store/{model_type}_epoch{epoch}.pt"
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                # Add any other desired information
            },
            save_path
        )
        my_logger.info(f"Model saved at {save_path}")


# Instantiate the downstream task model
d_out = n_classes or 1

model_ds = rtdl.FTTransformer.make_default(
    n_num_features=norm_train_df_X.shape[1],
    cat_cardinalities=None,
    last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
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
    if task_type == 'binclass'
    else F.cross_entropy
    if task_type == 'multiclass'
    else F.mse_loss
)

batch_size = 256
train_loader = zero.data.IndexLoader(len(norm_train_df_X), batch_size, device=device)

# Create a progress tracker for early stopping
# Docs: https://yura52.github.io/delu/reference/api/zero.ProgressTracker.html
progress = zero.ProgressTracker(patience=100)

my_logger.info(f'Test score before training: {evaluate("test", norm_test_df_X, norm_test_df_Y):.4f}')

# Downstream model training
n_epochs = 20
report_frequency = len(norm_train_df_X) // batch_size // 1
for epoch in range(1, n_epochs + 1):
    for iteration, batch_idx in enumerate(train_loader):
        model_ds.train()
        optimizer_ds.zero_grad()
        x_batch = norm_train_df_X.to(device)[batch_idx]
        y_batch = norm_train_df_Y.to(device)[batch_idx]
        loss = loss_fn(apply_model_ds(model_ds, x_batch), y_batch)
        loss.backward()
        optimizer_ds.step()
        if iteration % report_frequency == 0:
            my_logger.info(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

    test_score = evaluate('test', norm_test_df_X, norm_test_df_Y)
    my_logger.info(f'Epoch | Test score: {test_score:.4f}')
    progress.update((-1 if task_type == 'regression' else 1) * test_score)
    save_path = f"model_store/downstream_model_epoch{epoch}.pt"
    # Save the model
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model_ds.state_dict(),
            'optimizer_state_dict': optimizer_ds.state_dict(),
            'loss': loss.item(),
            # Add any other desired information
        },
        save_path
    )
    my_logger.info(f"Model saved at {save_path}")

    if progress.success:
        my_logger.info(' <<< BEST TEST EPOCH')
    if progress.fail:
        break

mask_mode = 0.1  #Choose a number between 0 and 1 for random mask and choose the number equal or greater than one for the column mask

# Start sampling
mean_seed_loss = []
median_seed_loss = []
mode_seed_loss = []
arbitrary_0_seed_loss = []
arbitrary_1_seed_loss = []
LOCF_seed_loss = []
NOCB_seed_loss = []

mean_seed_pcor = []
median_seed_pcor = []
mode_seed_pcor = []
LOCF_seed_pcor = []
NOCB_seed_pcor = []

mean_seed_score = []
median_seed_score = []
mode_seed_score = []
arbitrary_0_seed_score = []
arbitrary_1_seed_score = []
LOCF_seed_score = []
NOCB_seed_score = []
result = {}
for model, optimizer, model_type in models:
    if isinstance(model, models.MLP):
        my_logger.info('=' * 50 + 'MLP Model Evaluation' + '=' * 50)
    elif isinstance(model, models.ResNet):
        my_logger.info('=' * 50 + 'ResNet Model Evaluation' + '=' * 50)
    elif isinstance(model, models.FTTransformer):
        my_logger.info('=' * 50 + 'FT Transformer Model Evaluation' + '=' * 50)
    else:
        my_logger.info('=' * 50 + 'U-Net Model Evaluation' + '=' * 50)

    repaint_seed_losses = []

    repaint_seed_pcor = []

    repaint_seed_score = []

    count = 0
    if 0 < mask_mode < 1:
        for seed in range(5):
            torch.manual_seed(seed)  # control the randomness of the random mask
            masks = (torch.FloatTensor(norm_test_df_X.shape[0], norm_test_df_X.shape[1]).uniform_() > mask_mode).to(device)
            masks[[0, -1], :] = True
            torch.save(masks, f'model_store/{mask_mode*100}%_{seed}_mask.pt')

            # Take the average repaint prediction result
            preds = []
            for i in range(5):
                torch.manual_seed(i + 15)  # control the randomness of the random noise in DDPM inference
                preds.append(torch.from_numpy(sample(model, gt=norm_test_df_X, masks=masks.to(device))[-1]))

            pred_avg = torch.zeros_like(preds[0])
            for i in preds:
                pred_avg = pred_avg.add(i / len(preds))
            torch.save(pred_avg, f'model_store/{mask_mode*100}%_{seed}_pred_tensor_{model_type}_timestep.pt')

            loss = np.nanmean(np.square(np.array((pred_avg.to(device).masked_fill(masks, float(
                'nan')) - norm_test_df_X.masked_fill(masks, float('nan'))).cpu())))
            repaint_seed_losses.append(loss)

            ground_truth_values = np.array(
                norm_test_df_X.masked_fill(masks, float('nan')).reshape(pred_avg.shape[0] * pred_avg.shape[1]).cpu())

            repaint_values = np.array(
                pred_avg.to(device).masked_fill(masks, float('nan')).reshape(pred_avg.shape[0] * pred_avg.shape[1]).cpu())
            r, p = scipy.stats.pearsonr(repaint_values[~np.isnan(repaint_values)],
                                        ground_truth_values[~np.isnan(ground_truth_values)])

            repaint_seed_pcor.append(r)

            repaint_seed_score.append(evaluate("test", pred_avg, norm_test_df_Y))

            if count == 0:
                # take the average default method result
                mask_test = pd.DataFrame(norm_test_df_X.masked_fill(masks < 1, float('nan')).to('cpu'))

                mean_imp = torch.tensor(mask_test.fillna(mask_test.mean()).values).to(device)
                mean_loss = np.nanmean(np.square(
                    (mean_imp.masked_fill(masks, float('nan')) - norm_test_df_X.masked_fill(masks, float('nan'))).cpu()))
                med_imp = torch.tensor(mask_test.fillna(mask_test.median()).values).to(device)
                med_loss = np.nanmean(np.square(
                    (med_imp.masked_fill(masks, float('nan')) - norm_test_df_X.masked_fill(masks, float('nan'))).cpu()))
                mode_imp = torch.tensor(mask_test.fillna(mask_test.mode().iloc[0, :]).values).to(device)
                mode_loss = np.nanmean(np.square(
                    (mode_imp.masked_fill(masks, float('nan')) - norm_test_df_X.masked_fill(masks, float('nan'))).cpu()))
                zero_imp = torch.tensor(mask_test.fillna(0).values).to(device)
                zero_loss = np.nanmean(np.square(
                    (zero_imp.masked_fill(masks, float('nan')) - norm_test_df_X.masked_fill(masks, float('nan'))).cpu()))
                one_imp = torch.tensor(mask_test.fillna(1).values).to(device)
                one_loss = np.nanmean(np.square(
                    (one_imp.masked_fill(masks, float('nan')) - norm_test_df_X.masked_fill(masks, float('nan'))).cpu()))
                locf_imp = torch.tensor(mask_test.fillna(method='ffill').values).to(device)
                locf_loss = np.nanmean(np.square(
                    (locf_imp.masked_fill(masks, float('nan')) - norm_test_df_X.masked_fill(masks, float('nan'))).cpu()))
                nocb_imp = torch.tensor(mask_test.fillna(method='bfill').values).to(device)
                nocb_loss = np.nanmean(np.square(
                    (nocb_imp.masked_fill(masks, float('nan')) - norm_test_df_X.masked_fill(masks, float('nan'))).cpu()))

                mean_seed_loss.append(mean_loss)
                median_seed_loss.append(med_loss)
                mode_seed_loss.append(mode_loss)
                arbitrary_0_seed_loss.append(zero_loss)
                arbitrary_1_seed_loss.append(one_loss)
                LOCF_seed_loss.append(locf_loss)
                NOCB_seed_loss.append(nocb_loss)

                mean_imp_values = np.array(mean_imp.to(device).masked_fill(masks, float('nan')).reshape(
                    mean_imp.shape[0] * mean_imp.shape[1]).cpu())
                r, _ = scipy.stats.pearsonr(mean_imp_values[~np.isnan(mean_imp_values)],
                                            ground_truth_values[~np.isnan(ground_truth_values)])
                mean_seed_pcor.append(r)

                med_imp_values = np.array(
                    med_imp.to(device).masked_fill(masks, float('nan')).reshape(med_imp.shape[0] * med_imp.shape[1]).cpu())
                r, _ = scipy.stats.pearsonr(med_imp_values[~np.isnan(med_imp_values)],
                                            ground_truth_values[~np.isnan(ground_truth_values)])
                median_seed_pcor.append(r)

                mode_imp_values = np.array(mode_imp.to(device).masked_fill(masks, float('nan')).reshape(
                    mode_imp.shape[0] * mode_imp.shape[1]).cpu())
                r, _ = scipy.stats.pearsonr(mode_imp_values[~np.isnan(mode_imp_values)],
                                            ground_truth_values[~np.isnan(ground_truth_values)])
                mode_seed_pcor.append(r)

                locf_imp_values = np.array(locf_imp.to(device).masked_fill(masks, float('nan')).reshape(
                    locf_imp.shape[0] * locf_imp.shape[1]).cpu())
                r, _ = scipy.stats.pearsonr(locf_imp_values[~np.isnan(locf_imp_values)],
                                            ground_truth_values[~np.isnan(ground_truth_values)])
                LOCF_seed_pcor.append(r)

                nocb_imp_values = np.array(nocb_imp.to(device).masked_fill(masks, float('nan')).reshape(
                    nocb_imp.shape[0] * nocb_imp.shape[1]).cpu())
                r, _ = scipy.stats.pearsonr(nocb_imp_values[~np.isnan(nocb_imp_values)],
                                            ground_truth_values[~np.isnan(ground_truth_values)])
                NOCB_seed_pcor.append(r)

                mean_seed_score.append(evaluate("test", mean_imp, norm_test_df_Y))
                median_seed_score.append(evaluate("test", med_imp, norm_test_df_Y))
                mode_seed_score.append(evaluate("test", mode_imp, norm_test_df_Y))
                arbitrary_0_seed_score.append(evaluate("test", zero_imp, norm_test_df_Y))
                arbitrary_1_seed_score.append(evaluate("test", one_imp, norm_test_df_Y))
                LOCF_seed_score.append(evaluate("test", locf_imp, norm_test_df_Y))
                NOCB_seed_score.append(evaluate("test", nocb_imp, norm_test_df_Y))

        avg_repaint_seed_losses = sum(repaint_seed_losses) / len(repaint_seed_losses)
        avg_repaint_seed_pcor = sum(repaint_seed_pcor) / len(repaint_seed_pcor)
        avg_repaint_seed_score = sum(repaint_seed_score) / len(repaint_seed_score)

        count += 1
        my_logger.info("Repaint Imputation Loss:{0:.4f}:".format(avg_repaint_seed_losses))
        my_logger.info('-' * 120)
        my_logger.info("Repaint Imputation Pearson Correlation:{0:.4f}".format(avg_repaint_seed_pcor))

        my_logger.info('-' * 120)
        my_logger.info("Repaint Imputation MSE Score:{0:.4f}".format(avg_repaint_seed_score))




