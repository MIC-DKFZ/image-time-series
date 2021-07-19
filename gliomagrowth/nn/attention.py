import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple

from gliomagrowth.nn.block import ConcatCoords


class MultiheadAttention(nn.Module):
    """MultiheadAttention with convenience functionality for spatial attention.

    This is almost the same as the PyTorch implementation, but it can work with tensors
    with spatial dimensions. It's also a bit slower :/

    Args:
        embed_dim: Embedding dimension (where the dot product is calculated).
        num_heads: Number of attention heads.
        spatial_attention: Use spatial attention. This allows you to use other shapes
            than (B, N, C) and reshaping will be handled automatically.
        concat_coords: Concatenate coordinates to keys and queries. This is a simpler
            form of position embedding.
        bias: Use bias term in projection layers.
        batch_first: Activate this if the batch dimension (instead of the element
            dimension) is the first axis.
        embed_v: Also do embedding and output projection for values.
        qdim: Dimension of queries. If None, we expect embed_dim.
        kdim: Dimension of keys. If None, we expect embed_dim.
        vdim: Dimension of values. If None, we expect embed_dim.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        spatial_attention: bool = False,
        concat_coords: int = 0,
        bias: bool = True,
        batch_first: bool = False,
        embed_v: bool = True,
        qdim: Optional[int] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):

        if embed_dim % num_heads != 0:
            raise ValueError("'embed_dim' must be divisible by 'num_heads'.")

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.spatial_attention = spatial_attention
        self.concat_coords = concat_coords
        self.bias = bias
        self.batch_first = batch_first
        self.embed_v = embed_v
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.make_projections()

    def make_projections(self):
        """Create the projection layers from the current attributes."""

        self.q_proj = nn.Linear(
            self.qdim + self.concat_coords, self.embed_dim, bias=self.bias
        )
        self.k_proj = nn.Linear(
            self.kdim + self.concat_coords, self.embed_dim, bias=self.bias
        )
        if self.embed_v:
            self.v_proj = nn.Linear(self.vdim, self.embed_dim, bias=self.bias)
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)

        if self.concat_coords:
            self.cc_layer = ConcatCoords()

    def single_head(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute a single attention head.

        Args:
            query: Shape (B, ..., M, embed_dim/num_heads).
            key: Shape (B, ..., N, embed_dim/num_heads).
            value: Shape (B, ..., N, vdim/num_heads).

        Returns:
            The modified and aggregated values.

        """

        # spatial attention requires same spatial dimensions for key and value
        if self.spatial_attention:
            query_shape = query.shape
            query = query.reshape(query.shape[0], -1, query.shape[-1])
            key_shape = key.shape
            key = key.reshape(key.shape[0], -1, key.shape[-1])
            value_shape = value.shape
            value = value.reshape(value.shape[0], -1, value.shape[-1])

        d = query.shape[-1]
        weights = nn.functional.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d), dim=-1
        )
        unsqueeze_counter = 0
        while weights.ndim < value.ndim:
            weights = weights.unsqueeze(1)
            unsqueeze_counter += 1
        value = torch.matmul(weights, value)
        while unsqueeze_counter > 0:
            weights = weights[:, 0]
            unsqueeze_counter -= 1

        if self.spatial_attention:
            value = value.reshape(
                value.shape[0], *value_shape[1:-2], query_shape[-2], value.shape[-1]
            )
            weights = weights.reshape(
                weights.shape[0],
                query_shape[-2],
                *query_shape[1:-2],
                key_shape[-2],
                *key_shape[1:-2]
            )

        return value, weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the attention mechanism.

        Args:
            query: Shape (M, B, Cq, ...) if batch_first is not set,
                otherwise (B, M, Cq, ...).
            key: Shape (N, B, Cq, ...) if batch_first is not set,
                otherwise (B, N, Cq, ...).
            value: Shape (N, B, Cv, ...) if batch_first is not set,
                otherwise (B, N, Cv, ...).
            need_weights: If True, will also return the attention weights.

        Returns:
            The modified values. If need_weights is True,
                will return a tuple of tensors, otherwise (tensor, None)

        """

        # move the batch axis to the front
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # concatenate spatial coordinates if desired
        if self.concat_coords:
            query = self.cc_layer(query, batch_dims=2)
            key = self.cc_layer(key, batch_dims=2)

        # project everything by moving sequence and channel axes to the back,
        # for further processing it stays there
        q_permute = [0] + list(range(3, query.ndim)) + [1, 2]
        query = query.permute(*q_permute)
        query = self.q_proj(query)
        k_permute = [0] + list(range(3, key.ndim)) + [1, 2]
        key = key.permute(*k_permute)
        key = self.k_proj(key)
        v_permute = [0] + list(range(3, value.ndim)) + [1, 2]
        value = value.permute(*v_permute)
        if self.embed_v:
            value = self.v_proj(value)

        # shapes are now (B, ..., N/M, embed_dim/Cv)
        result = []
        for h in range(self.num_heads):

            q_block_size = query.shape[-1] // self.num_heads
            v_block_size = value.shape[-1] // self.num_heads

            v, w = self.single_head(
                query[..., h * q_block_size : (h + 1) * q_block_size],
                key[..., h * q_block_size : (h + 1) * q_block_size],
                value[..., h * v_block_size : (h + 1) * v_block_size],
            )
            result.append(v)
            if need_weights:
                if h == 0:
                    weights = w
                else:
                    weights = weights + w
        result = torch.cat(result, -1)  # (B, ..., M, Cv)
        weights = weights / self.num_heads

        # output projection and permutation to original order
        if self.embed_v:
            result = self.out_proj(result)
        result_permute = [0, result.ndim - 2, result.ndim - 1] + list(
            range(1, result.ndim - 2)
        )
        result = result.permute(*result_permute)
        if not self.batch_first:
            result = result.transpose(0, 1)

        if need_weights:
            return result, weights
        else:
            return result, None