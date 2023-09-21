import rtdl
import zero
import math
import enum
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

def exists(x):
    return x is not None

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings


ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'

def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )

def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)

class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rtdlF.reglu(x)

class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rtdlF.geglu(x)

class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class NumericalFeatureTokenizer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(n_features, d_token))
        self.bias = nn.Parameter(Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        return len(self.weight)

    @property
    def d_token(self) -> int:
        return self.weight.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class CategoricalFeatureTokenizer(nn.Module):
    category_offsets: Tensor
    def __init__(
        self,
        cardinalities: List[int],
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        super().__init__()
        assert cardinalities, 'cardinalities must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        initialization_ = _TokenInitialization.from_str(initialization)

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_token)
        self.bias = nn.Parameter(Tensor(len(cardinalities), d_token)) if bias else None

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
    ) -> None:
        super().__init__()
        assert n_num_features >= 0, 'n_num_features must be non-negative'
        assert (
            n_num_features or cat_cardinalities
        ), 'at least one of n_num_features or cat_cardinalities must be positive/non-empty'
        self.initialization = 'uniform'
        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                n_features=n_num_features,
                d_token=d_token,
                bias=True,
                initialization=self.initialization,
            )
            if n_num_features
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(
                cat_cardinalities, d_token, True, self.initialization
            )
            if cat_cardinalities
            else None
        )

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(
            x.n_tokens
            for x in [self.num_tokenizer, self.cat_tokenizer]
            if x is not None
        )

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return (
            self.cat_tokenizer.d_token  # type: ignore
            if self.num_tokenizer is None
            else self.num_tokenizer.d_token
        )

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        assert (
            x_num is not None or x_cat is not None
        ), 'At least one of x_num and x_cat must be presented'
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), 'If self.num_tokenizer is (not) None, then x_num must (not) be None'
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), 'If self.cat_tokenizer is (not) None, then x_cat must (not) be None'
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class CLSToken(nn.Module):
    def __init__(self, d_token: int, initialization: str) -> None:
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) -> Tensor:
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            return ReGLU()
        elif module_type == 'GEGLU':
            return GEGLU()
        elif module_type == 'BatchNorm1d':
            return torch.nn.BatchNorm1d(*args, device=device)
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
            return cls(*args)
    else:
        return module_type(*args)


# MLP model definition
class MLP(nn.Module):
    class Block(nn.Module):

        def __init__(
                self,
                *,
                d_in: int,
                d_out: int,
                bias: bool,
                activation: ModuleType,
                dropout: float,
                time_emb_dim=None,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias, device=device)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

            self.mlp = (
                nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, d_out * 2, device=device))
                if exists(time_emb_dim)
                else None
            )

        def forward(self, x: Tensor, time_emb=None) -> Tensor:
            x = self.linear(x)
            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                # time_emb = rearrange(time_emb, "b c -> b c")
                scale, shift = time_emb.chunk(2, dim=1)
                x = x * (scale + 1) + shift
            x = self.activation(x)
            x = self.dropout(x)
            return x

    def __init__(
            self,
            *,
            d_in: int,
            d_layers: List[int],
            dropouts: Union[float, List[float]],
            activation: Union[str, Callable[[], nn.Module]],
            d_out: int,
    ) -> None:

        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)

        # time embeddings
        time_dim = d_in * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_in),
            nn.Linear(d_in, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.blocks = CustomSequential(
            *[
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                    time_emb_dim=time_dim,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
            cls: Type['MLP'],
            d_in: int,
            d_layers: List[int],
            dropout: float,
            d_out: int,
    ) -> 'MLP':

        assert isinstance(dropout, float), 'In this constructor, dropout must be float'
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (

            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor, time=None) -> Tensor:
        if time is not None:
            t = self.time_mlp(time)
            x = self.blocks(x, time_emb=t)
        else:
            x = self.blocks(x)
        x = self.head(x)
        return x


# ResNet model definition
class ResNet(nn.Module):
    class Block(nn.Module):
        def __init__(
                self,
                *,
                d_main: int,
                d_hidden: int,
                bias_first: bool,
                bias_second: bool,
                dropout_first: float,
                dropout_second: float,
                normalization: ModuleType,
                activation: ModuleType,
                skip_connection: bool,
                time_emb_dim=None,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first, device=device)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second, device=device)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

            self.mlp = (
                nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, d_hidden * 2, device=device))
                if exists(time_emb_dim)
                else None
            )

        def forward(self, x: Tensor, time_emb=None) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)

            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                # time_emb = rearrange(time_emb, "b c -> b c")
                scale, shift = time_emb.chunk(2, dim=1)
                x = x * (scale + 1) + shift

            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        def __init__(
                self,
                *,
                d_in: int,
                d_out: int,
                bias: bool,
                normalization: ModuleType,
                activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias, device=device)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
            self,
            *,
            d_in: int,
            n_blocks: int,
            d_main: int,
            d_hidden: int,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            d_out: int,
    ) -> None:

        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main, device=device)
        if d_main is None:
            d_main = d_in

        # time embeddings
        time_dim = d_in * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_in),
            nn.Linear(d_in, time_dim, device=device),
            nn.GELU(),
            nn.Linear(time_dim, time_dim, device=device),
        )

        self.blocks = CustomSequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                    time_emb_dim=time_dim,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
            cls: Type['ResNet'],
            *,
            d_in: int,
            n_blocks: int,
            d_main: int,
            d_hidden: int,
            dropout_first: float,
            dropout_second: float,
            d_out: int,
    ) -> 'ResNet':

        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor, time=None) -> Tensor:
        x = self.first_layer(x)
        if time is not None:
            t = self.time_mlp(time)
            x = self.blocks(x, time_emb=t)
        else:
            x = self.blocks(x)
        x = self.head(x)
        return x


# Transformer model definition
class MultiheadAttention(nn.Module):
    def __init__(
            self,
            *,
            d_token: int,
            n_heads: int,
            dropout: float,
            bias: bool,
            initialization: str,
    ) -> None:

        super().__init__()
        if n_heads > 1:
            assert d_token % n_heads == 0, 'd_token must be a multiple of n_heads'
        assert initialization in ['kaiming', 'xavier']

        self.W_q = nn.Linear(d_token, d_token, bias, device=device)
        self.W_k = nn.Linear(d_token, d_token, bias, device=device)
        self.W_v = nn.Linear(d_token, d_token, bias, device=device)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == 'xavier' and (
                    m is not self.W_v or self.W_out is not None
            ):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
            self,
            x_q: Tensor,
            x_kv: Tensor,
            key_compression: Optional[nn.Linear],
            value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:

        assert _all_or_none(
            [key_compression, value_compression]
        ), 'If key_compression is (not) None, then value_compression must (not) be None'
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, _INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            'attention_logits': attention_logits,
            'attention_probs': attention_probs,
        }


class Transformer(nn.Module):
    WARNINGS = {'first_prenormalization': True, 'prenormalization': True}

    class FFN(nn.Module):

        def __init__(
                self,
                *,
                d_token: int,
                d_hidden: int,
                bias_first: bool,
                bias_second: bool,
                dropout: float,
                activation: ModuleType,
                time_emb_dim=None,
        ):
            super().__init__()
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation(activation) else 1),
                bias_first,
                device=device
            )
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second, device=device)

            self.mlp = (
                nn.Sequential(nn.SiLU(),
                              nn.Linear(time_emb_dim, d_hidden * (2 if _is_glu_activation(activation) else 1) * 2,
                                        device=device))
                if exists(time_emb_dim)
                else None
            )

        def forward(self, x: Tensor, time_emb=None) -> Tensor:
            x = self.linear_first(x)
            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                time_emb = rearrange(time_emb, "b c -> b 1 c")
                scale, shift = time_emb.chunk(2, dim=2)
                x = x * (scale + 1) + shift
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    class Head(nn.Module):

        def __init__(
                self,
                *,
                d_in: int,
                bias: bool,
                activation: ModuleType,
                normalization: ModuleType,
                d_out: int,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias, device=device)

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
            self,
            *,
            d_token: int,
            n_blocks: int,
            attention_n_heads: int,
            attention_dropout: float,
            attention_initialization: str,
            attention_normalization: str,
            ffn_d_hidden: int,
            ffn_dropout: float,
            ffn_activation: str,
            ffn_normalization: str,
            residual_dropout: float,
            prenormalization: bool,
            first_prenormalization: bool,
            last_layer_query_idx: Union[None, List[int], slice],
            n_tokens: Optional[int],
            kv_compression_ratio: Optional[float],
            kv_compression_sharing: Optional[str],
            head_activation: ModuleType,
            head_normalization: ModuleType,
            d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                'last_layer_query_idx must be None, list[int] or slice. '
                f'Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?'
            )
        if not prenormalization:
            assert (
                not first_prenormalization
            ), 'If `prenormalization` is False, then `first_prenormalization` must be False'
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), (
            'If any of the following arguments is (not) None, then all of them must (not) be None: '
            'n_tokens, kv_compression_ratio, kv_compression_sharing'
        )
        assert kv_compression_sharing in [None, 'headwise', 'key-value', 'layerwise']
        if not prenormalization:
            if self.WARNINGS['prenormalization']:
                warnings.warn(
                    'prenormalization is set to False. Are you sure about this? '
                    'The training can become less stable. '
                    'You can turn off this warning by tweaking the '
                    'rtdl.Transformer.WARNINGS dictionary.',
                    UserWarning,
                )
            assert (
                not first_prenormalization
            ), 'If prenormalization is False, then first_prenormalization is ignored and must be set to False'
        if (
                prenormalization
                and first_prenormalization
                and self.WARNINGS['first_prenormalization']
        ):
            warnings.warn(
                'first_prenormalization is set to True. Are you sure about this? '
                'For example, the vanilla FTTransformer with '
                'first_prenormalization=True performs SIGNIFICANTLY worse. '
                'You can turn off this warning by tweaking the '
                'rtdl.Transformer.WARNINGS dictionary.',
                UserWarning,
            )
            time.sleep(3)

        # time embeddings
        time_dim = d_token * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_token),
            nn.Linear(d_token, time_dim, device=device),
            nn.GELU(),
            nn.Linear(time_dim, time_dim, device=device),
        )

        def make_kv_compression():
            assert (
                    n_tokens and kv_compression_ratio
            ), _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False, device=device)

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_ratio and kv_compression_sharing == 'layerwise'
            else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    'ffn': Transformer.FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                        time_emb_dim=time_dim,
                    ),
                    'attention_residual_dropout': nn.Dropout(residual_dropout),
                    'ffn_residual_dropout': nn.Dropout(residual_dropout),
                    'output': nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(
                    attention_normalization, d_token
                )
            layer['ffn_normalization'] = _make_nn_module(ffn_normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert (
                            kv_compression_sharing == 'key-value'
                    ), _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)

        self.head = Transformer.Head(
            d_in=d_token,
            d_out=d_out,
            bias=True,
            activation=head_activation,  # type: ignore
            normalization=head_normalization if prenormalization else 'Identity',
        )

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, layer, stage, x):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'{stage}_normalization'](x)
        return x

    def forward(self, x: Tensor, time=None) -> Tensor:
        assert (
                x.ndim == 3
        ), 'The input must have 3 dimensions: (n_objects, n_tokens, d_token)'
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = (
                self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            )
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, 'attention', x, x_residual)

            x_residual = self._start_residual(layer, 'ffn', x)
            if time is not None:
                t = self.time_mlp(time)
                x_residual = layer['ffn'](x_residual, time_emb=t)
            else:
                x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)
            x = layer['output'](x)

        x = self.head(x)
        return x


class FTTransformer(nn.Module):
    def __init__(
            self, feature_tokenizer: FeatureTokenizer, transformer: Transformer
    ) -> None:
        """
        Note:
            `make_baseline` and `make_default` are the recommended constructors.
        """
        super().__init__()
        if transformer.prenormalization:
            assert 'attention_normalization' not in transformer.blocks[0], (
                'In the prenormalization setting, FT-Transformer does not '
                'allow using the first normalization layer '
                'in the first transformer block'
            )
        self.feature_tokenizer = feature_tokenizer
        self.cls_token = CLSToken(
            feature_tokenizer.d_token, feature_tokenizer.initialization
        )
        self.transformer = transformer

    @classmethod
    def get_baseline_transformer_subconfig(
            cls: Type['FTTransformer'],
    ) -> Dict[str, Any]:
        """Get the baseline subset of parameters for the backbone."""
        return {
            'attention_n_heads': 8,
            'attention_initialization': 'kaiming',
            'ffn_activation': 'ReGLU',
            'attention_normalization': 'LayerNorm',
            'ffn_normalization': 'LayerNorm',
            'prenormalization': True,
            'first_prenormalization': False,
            'last_layer_query_idx': None,
            'n_tokens': None,
            'kv_compression_ratio': None,
            'kv_compression_sharing': None,
            'head_activation': 'ReLU',
            'head_normalization': 'LayerNorm',
        }

    @classmethod
    def get_default_transformer_config(
            cls: Type['FTTransformer'], *, n_blocks: int = 3
    ) -> Dict[str, Any]:
        """Get the default parameters for the backbone.

        Note:
            The configurations are different for different values of:code:`n_blocks`.
        """
        assert 1 <= n_blocks <= 6
        grid = {
            'd_token': [96, 128, 192, 256, 320, 384],
            'attention_dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
            'ffn_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        }
        arch_subconfig = {k: v[n_blocks - 1] for k, v in grid.items()}  # type: ignore
        baseline_subconfig = cls.get_baseline_transformer_subconfig()
        # (4 / 3) for ReGLU/GEGLU activations results in almost the same parameter count
        # as (2.0) for element-wise activations (e.g. ReLU or GELU; see the "else" branch)
        ffn_d_hidden_factor = (
            (4 / 3) if _is_glu_activation(baseline_subconfig['ffn_activation']) else 2.0
        )
        return {
            'n_blocks': n_blocks,
            'residual_dropout': 0.0,
            'ffn_d_hidden': int(arch_subconfig['d_token'] * ffn_d_hidden_factor),
            **arch_subconfig,
            **baseline_subconfig,
        }

    @classmethod
    def _make(
            cls,
            n_num_features,
            cat_cardinalities,
            transformer_config,
    ):
        feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=transformer_config['d_token'],
        )
        if transformer_config['d_out'] is None:
            transformer_config['head_activation'] = None
        if transformer_config['kv_compression_ratio'] is not None:
            transformer_config['n_tokens'] = feature_tokenizer.n_tokens + 1
        return FTTransformer(
            feature_tokenizer,
            Transformer(**transformer_config),
        )

    @classmethod
    def make_baseline(
            cls: Type['FTTransformer'],
            *,
            n_num_features: int,
            cat_cardinalities: Optional[List[int]],
            d_token: int,
            n_blocks: int,
            attention_dropout: float,
            ffn_d_hidden: int,
            ffn_dropout: float,
            residual_dropout: float,
            last_layer_query_idx: Union[None, List[int], slice] = None,
            kv_compression_ratio: Optional[float] = None,
            kv_compression_sharing: Optional[str] = None,
            d_out: int,
    ) -> 'FTTransformer':
        transformer_config = cls.get_baseline_transformer_subconfig()
        for arg_name in [
            'n_blocks',
            'd_token',
            'attention_dropout',
            'ffn_d_hidden',
            'ffn_dropout',
            'residual_dropout',
            'last_layer_query_idx',
            'kv_compression_ratio',
            'kv_compression_sharing',
            'd_out',
        ]:
            transformer_config[arg_name] = locals()[arg_name]
        return cls._make(n_num_features, cat_cardinalities, transformer_config)

    @classmethod
    def make_default(
            cls: Type['FTTransformer'],
            *,
            n_num_features: int,
            cat_cardinalities: Optional[List[int]],
            n_blocks: int = 3,
            last_layer_query_idx: Union[None, List[int], slice] = None,
            kv_compression_ratio: Optional[float] = None,
            kv_compression_sharing: Optional[str] = None,
            d_out: int,
    ) -> 'FTTransformer':

        transformer_config = cls.get_default_transformer_config(n_blocks=n_blocks)
        for arg_name in [
            'last_layer_query_idx',
            'kv_compression_ratio',
            'kv_compression_sharing',
            'd_out',
        ]:
            transformer_config[arg_name] = locals()[arg_name]
        return cls._make(n_num_features, cat_cardinalities, transformer_config)

    def optimization_param_groups(self) -> List[Dict[str, Any]]:

        no_wd_names = ['feature_tokenizer', 'normalization', '.bias']
        assert isinstance(
            getattr(self, no_wd_names[0], None), FeatureTokenizer
        ), _INTERNAL_ERROR_MESSAGE
        assert (
                sum(1 for name, _ in self.named_modules() if no_wd_names[1] in name)
                == len(self.transformer.blocks) * 2
                - int('attention_normalization' not in self.transformer.blocks[0])  # type: ignore
                + 1
        ), _INTERNAL_ERROR_MESSAGE

        def needs_wd(name):
            return all(x not in name for x in no_wd_names)

        return [
            {'params': [v for k, v in self.named_parameters() if needs_wd(k)]},
            {
                'params': [v for k, v in self.named_parameters() if not needs_wd(k)],
                'weight_decay': 0.0,
            },
        ]

    def make_default_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.optimization_param_groups(),
            lr=1e-4,
            weight_decay=1e-5,
        )

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor], time=None) -> Tensor:
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.cls_token(x)
        x = self.transformer(x, time)
        return x


# Unet model definition
class CustomSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def forward(self, x, **kwargs):
        for module in self.modules:
            x = module(x, **kwargs)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, max(1, in_channels//8), 1)
        self.key = nn.Conv1d(in_channels, max(1, in_channels//8), 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.bmm(query.permute(0, 2, 1), key)
        attn = self.softmax(scores)
        out = torch.bmm(attn, value.permute(0, 2, 1))
        return out.permute(0, 2, 1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=1):
        super().__init__()

        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=1):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 ")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class UNet1D(nn.Module):
    def __init__(self, n_num_features, d_out, resnet_block_groups=1):
        super().__init__()

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = n_num_features * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(n_num_features),
            nn.Linear(n_num_features, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.down_conv1 = block_klass(1, 64, time_emb_dim=time_dim)
        self.down_conv2 = block_klass(64, 128, time_emb_dim=time_dim)
        self.down_conv3 = block_klass(128, 256, time_emb_dim=time_dim)
        self.down_conv4 = block_klass(256, 512, time_emb_dim=time_dim)

        # Added Bottleneck Layer
        self.bottleneck = nn.Sequential(
            block_klass(512, 512, time_emb_dim=time_dim),
            Residual(nn.GroupNorm(1, 512)),
            SelfAttention(512)
        )

        # Decoder
        self.up_conv4 = block_klass(512 + 256, 256, time_emb_dim=time_dim)
        self.up_conv3 = block_klass(256 + 128, 128, time_emb_dim=time_dim)
        self.up_conv2 = block_klass(128 + 64, 64, time_emb_dim=time_dim)
        self.up_conv1 = block_klass(64 + 1, 1, time_emb_dim=time_dim)

        # Added Group Normalization
        self.down_norm1 = Residual(nn.GroupNorm(1, 64))
        self.down_norm2 = Residual(nn.GroupNorm(1, 128))
        self.down_norm3 = Residual(nn.GroupNorm(1, 256))

        self.up_norm3 = Residual(nn.GroupNorm(1, 256))
        self.up_norm2 = Residual(nn.GroupNorm(1, 128))
        self.up_norm1 = Residual(nn.GroupNorm(1, 64))

        # Added Attention Layer
        self.down_attn1 = SelfAttention(64)
        self.down_attn2 = SelfAttention(128)
        self.down_attn3 = SelfAttention(256)

        self.up_attn3 = SelfAttention(256)
        self.up_attn2 = SelfAttention(128)
        self.up_attn1 = SelfAttention(64)

        self.last_layer = nn.Linear(n_num_features, d_out)

    def forward(self, x, time: Optional[Tensor]):
        # Add channel dimension (C) to the 2D tabular data (N, L) -> (N, C, L)
        x = x.unsqueeze(1)
        if time is not None:
            t = self.time_mlp(time)
            # Encoder path
            x1 = self.down_conv1(x, t)
            x1 = self.down_norm1(x1)
            x1 = self.down_attn1(x1)
            x2 = self.down_conv2(x1, t)
            x2 = self.down_norm2(x2)
            x2 = self.down_attn2(x2)
            x3 = self.down_conv3(x2, t)
            x3 = self.down_norm3(x3)
            x3 = self.down_attn3(x3)
            x4 = self.down_conv4(x3, t)

            # Bottleneck
            bottleneck = self.bottleneck(x4)

            # Decoder path with skip connections
            y4 = self.up_conv4(torch.cat([bottleneck, x3], 1), t)
            y4 = self.up_norm3(y4)
            y4 = self.up_attn3(y4)
            y3 = self.up_conv3(torch.cat([y4, x2], 1), t)
            y3 = self.up_norm2(y3)
            y3 = self.up_attn2(y3)
            y2 = self.up_conv2(torch.cat([y3, x1], 1), t)
            y2 = self.up_norm1(y2)
            y2 = self.up_attn1(y2)
            y1 = self.up_conv1(torch.cat([y2, x], 1), t)
            y1 = y1.squeeze(1)
            y = self.last_layer(y1)
        else:
            # Encoder path
            x1 = self.down_conv1(x)
            x1 = self.down_norm1(x1)
            x1 = self.down_attn1(x1)
            x2 = self.down_conv2(x1)
            x2 = self.down_norm2(x2)
            x2 = self.down_attn2(x2)
            x3 = self.down_conv3(x2)
            x3 = self.down_norm3(x3)
            x3 = self.down_attn3(x3)
            x4 = self.down_conv4(x3)

            # Bottleneck
            bottleneck = self.bottleneck(x4)

            # Decoder path with skip connections
            y4 = self.up_conv4(torch.cat([bottleneck, x3], 1))
            y4 = self.up_norm3(y4)
            y4 = self.up_attn3(y4)
            y3 = self.up_conv3(torch.cat([y4, x2], 1))
            y3 = self.up_norm2(y3)
            y3 = self.up_attn2(y3)
            y2 = self.up_conv2(torch.cat([y3, x1], 1))
            y2 = self.up_norm1(y2)
            y2 = self.up_attn1(y2)
            y1 = self.up_conv1(torch.cat([y2, x], 1))
            y1 = y1.squeeze(1)
            y = self.last_layer(y1)

        # Squeeze the output to remove the channel dimension (N, C, L) -> (N, L)
        return y

