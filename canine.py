"""CANINE character processing.

This file contains copied out ans simplified parts of the CANINE model as
implemented in Huggingface Transformers. The original CANINE model uses a
special handling of the [CLS] embedding which was hardcoded and does not make
sense in MT.

https://huggingface.co/transformers/_modules/transformers/models/canine/modeling_canine.html#CanineModel
"""

from typing import Tuple
import math

import torch
import torch.nn as nn
from torch.functional import F
from transformers import apply_chunking_to_forward


T = torch.Tensor


class CanineSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            attention_probs_dropout_prob: float=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        from_tensor,
        to_tensor,
        attention_mask=None,
        head_mask=None,
    ):
        mixed_query_layer = self.query(from_tensor)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        key_layer = self.transpose_for_scores(self.key(to_tensor))
        value_layer = self.transpose_for_scores(self.value(to_tensor))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # TODO check if this is the same as in Transformers
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                # if attention_mask is 3D, do the following:
                attention_mask = torch.unsqueeze(attention_mask, dim=1)
                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and -10000.0 for masked positions.
                attention_mask = (1.0 - attention_mask.float()) * -10000.0
            # Apply the attention mask (precomputed for all layers in CanineModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)
        return outputs


class CanineSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CanineAttention(nn.Module):
    """
    Additional arguments related to local attention:

        - **local** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether to apply local attention.
        - **always_attend_to_first_position** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Should all blocks
          be able to attend
        to the :obj:`to_tensor`'s first position (e.g. a [CLS] position)? - **first_position_attends_to_all**
        (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Should the `from_tensor`'s first position be able to
        attend to all positions within the `from_tensor`? - **attend_from_chunk_width** (:obj:`int`, `optional`,
        defaults to 128) -- The width of each block-wise chunk in :obj:`from_tensor`. - **attend_from_chunk_stride**
        (:obj:`int`, `optional`, defaults to 128) -- The number of elements to skip when moving to the next block in
        :obj:`from_tensor`. - **attend_to_chunk_width** (:obj:`int`, `optional`, defaults to 128) -- The width of each
        block-wise chunk in `to_tensor`. - **attend_to_chunk_stride** (:obj:`int`, `optional`, defaults to 128) -- The
        number of elements to skip when moving to the next block in :obj:`to_tensor`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
        attend_from_chunk_width: int = 128,
        attend_from_chunk_stride: int = 128,
        attend_to_chunk_width: int = 128,
        attend_to_chunk_stride: int = 128,
    ):
        super().__init__()
        self.self = CanineSelfAttention(hidden_size, num_attention_heads, dropout)
        self.output = CanineSelfOutput(hidden_size, dropout)

        # additional arguments related to local attention
        if attend_from_chunk_width < attend_from_chunk_stride:
            raise ValueError(
                "`attend_from_chunk_width` < `attend_from_chunk_stride`"
                "would cause sequence positions to get skipped."
            )
        if attend_to_chunk_width < attend_to_chunk_stride:
            raise ValueError(
                "`attend_to_chunk_width` < `attend_to_chunk_stride`" "would cause sequence positions to get skipped."
            )
        self.attend_from_chunk_width = attend_from_chunk_width
        self.attend_from_chunk_stride = attend_from_chunk_stride
        self.attend_to_chunk_width = attend_to_chunk_width
        self.attend_to_chunk_stride = attend_to_chunk_stride

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        from_seq_length = to_seq_length = hidden_states.shape[1]
        from_tensor = to_tensor = hidden_states

        # Create chunks (windows) that we will attend *from* and then concatenate them.
        from_chunks = []
        from_start = 0
        for chunk_start in range(from_start, from_seq_length, self.attend_from_chunk_stride):
            chunk_end = min(from_seq_length, chunk_start + self.attend_from_chunk_width)
            from_chunks.append((chunk_start, chunk_end))

        # Determine the chunks (windows) that will will attend *to*.
        to_chunks = []
        for chunk_start in range(0, to_seq_length, self.attend_to_chunk_stride):
            chunk_end = min(to_seq_length, chunk_start + self.attend_to_chunk_width)
            to_chunks.append((chunk_start, chunk_end))

        if len(from_chunks) != len(to_chunks):
            raise ValueError(
                f"Expected to have same number of `from_chunks` ({from_chunks}) and "
                f"`to_chunks` ({from_chunks}). Check strides."
            )

        # next, compute attention scores for each pair of windows and concatenate
        attention_output_chunks = []
        attention_probs_chunks = []
        for (from_start, from_end), (to_start, to_end) in zip(from_chunks, to_chunks):
            from_tensor_chunk = from_tensor[:, from_start:from_end, :]
            to_tensor_chunk = to_tensor[:, to_start:to_end, :]
            # `attention_mask`: <float>[batch_size, from_seq, to_seq]
            # `attention_mask_chunk`: <float>[batch_size, from_seq_chunk, to_seq_chunk]
            attention_mask_chunk = attention_mask[:, from_start:from_end, to_start:to_end]
            attention_outputs_chunk = self.self(
                from_tensor_chunk, to_tensor_chunk, attention_mask_chunk, head_mask,
            )
            attention_output_chunks.append(attention_outputs_chunk[0])

        attention_output = torch.cat(attention_output_chunks, dim=1)

        attention_output = self.output(attention_output, hidden_states)
        outputs = (attention_output,)
        outputs = outputs + tuple(attention_probs_chunks)  # add attentions if we output them
        return outputs


class CanineIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CanineOutput(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CanineLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float,
        attend_from_chunk_width: int,
        attend_from_chunk_stride: int,
        attend_to_chunk_width: int,
        attend_to_chunk_stride: int,
    ):
        super().__init__()
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = CanineAttention(
            hidden_size,
            num_attention_heads,
            dropout,
            attend_from_chunk_width,
            attend_from_chunk_stride,
            attend_to_chunk_width,
            attend_to_chunk_stride,
        )
        self.intermediate = CanineIntermediate(hidden_size, 2 * hidden_size)
        self.output = CanineOutput(hidden_size, 2 * hidden_size, dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask)
        attention_output = self_attention_outputs[0]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CharactersToMolecules(nn.Module):
    """Convert character sequence to initial molecule sequence (i.e. downsample) using strided convolutions."""

    def __init__(
            self,
            hidden_size: int,
            shrink_factor: int):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=shrink_factor,
            stride=shrink_factor,
            padding=0
        )
        self.activation = F.gelu

        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, char_encoding: T) -> T:

        # char_encoding has shape [batch, char_seq, hidden_size]
        # We transpose it to be [batch, hidden_size, char_seq]
        char_encoding = torch.transpose(char_encoding, 1, 2)
        downsampled = self.conv(char_encoding)
        downsampled = torch.transpose(downsampled, 1, 2)
        downsampled = self.activation(downsampled)

        downsampled_truncated = downsampled[:, 0:-1, :]
        result = self.norm(downsampled_truncated)

        return result

class CanineEncoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            dropout: float,
            shrink_factor: int,
            attend_from_chunk_width: int,
            attend_from_chunk_stride: int,
            attend_to_chunk_width: int,
            attend_to_chunk_stride: int) -> None:
        super().__init__()

        self.transformer_layer = CanineLayer(
            hidden_size,
            num_attention_heads,
            dropout,
            attend_from_chunk_width,
            attend_from_chunk_stride,
            attend_to_chunk_width,
            attend_to_chunk_stride)
        self.downsampling = CharactersToMolecules(hidden_size, shrink_factor)

        self.downsample_mask = torch.nn.MaxPool1d(
            kernel_size=shrink_factor, stride=shrink_factor,
            padding=0, ceil_mode=False)

    def forward(self, data: T, mask: T) -> Tuple[T, T]:
        attention_mask = self._create_3d_attention_mask_from_input_mask(data, mask)

        result = self.transformer_layer(data, attention_mask=attention_mask)
        result = self.downsampling(data)
        shrinked_mask = self._downsample_attention_mask(mask)
        return result, shrinked_mask.squeeze(1)[:, 1:]

    def _downsample_attention_mask(self, char_attention_mask: T):
        """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""

        # first, make char_attention_mask 3D by adding a channel dim
        batch_size, char_seq_len = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))

        # next, apply MaxPool1d to get pooled_molecule_mask of shape (batch_size, 1, mol_seq_len)
        pooled_molecule_mask = self.downsample_mask(
            poolable_char_mask.float())

        # finally, squeeze to get tensor of shape (batch_size, mol_seq_len)
        molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)

        return molecule_attention_mask


    def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        batch_size, from_seq_length = from_tensor.shape[0], from_tensor.shape[1]

        to_seq_length = to_mask.shape[1]

        to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        broadcast_ones = torch.ones(size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask
