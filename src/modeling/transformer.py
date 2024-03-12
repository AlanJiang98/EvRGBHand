# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see https://github.com/facebookresearch/detr/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Transformer encoder-decoder architecture in FastMETRO model.
"""
import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor


class Transformer_QS(nn.Module):
    """Transformer encoder-decoder"""
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False):
        """
        Parameters:
            - model_dim: The hidden dimension size in the transformer architecture
            - nhead: The number of attention heads in the attention modules
            - num_enc_layers: The number of encoder layers in the transformer encoder
            - num_dec_layers: The number of decoder layers in the transformer decoder
            - feedforward_dim: The hidden dimension size in MLP
            - dropout: The dropout rate in the transformer architecture
            - activation: The activation function used in MLP
        """
        super().__init__()
        self.model_dim = d_model
        self.nhead = nhead

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # transformer decoder
        decoder_layer= TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, img_features, jv_tokens, pos_embed, attention_mask=None):
        device = img_features.device
        hw, bs, c= img_features.shape
        src = img_features
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = jv_tokens.unsqueeze(1).repeat(1, bs, 1)
        mask = torch.zeros((bs, hw), dtype=torch.bool, device=device)

        # Transformer Encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # Transformer Decoder
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, tgt_mask=attention_mask, 
                                   memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed) # (num_joints + num_vertices) X batch_size X feature_dim
        
        return hs.transpose(1, 2), memory.transpose(0, 1)

class Transformer(nn.Module):
    """Transformer encoder-decoder"""
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False, query_pos_update=False):
        """
        Parameters:
            - model_dim: The hidden dimension size in the transformer architecture
            - nhead: The number of attention heads in the attention modules
            - num_enc_layers: The number of encoder layers in the transformer encoder
            - num_dec_layers: The number of decoder layers in the transformer decoder
            - feedforward_dim: The hidden dimension size in MLP
            - dropout: The dropout rate in the transformer architecture
            - activation: The activation function used in MLP
        """
        super().__init__()
        self.model_dim = d_model
        self.nhead = nhead

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # transformer decoder
        decoder_layer= TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

        if query_pos_update:
            query_pos_embed = MLP(d_model, d_model, d_model, 2)
            self.query_pos_refine = _get_clones(query_pos_embed, num_decoder_layers-1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, img_features, jv_tokens, pos_embed, attention_mask=None):
        device = img_features.device
        hw, bs, c= img_features.shape
        src = img_features
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = jv_tokens#.unsqueeze(1).repeat(1, bs, 1)
        mask = torch.zeros((bs, hw), dtype=torch.bool, device=device)

        # Transformer Encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # Transformer Decoder
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, tgt_mask=attention_mask, 
                                   memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed) # (num_joints + num_vertices) X batch_size X feature_dim
        
        return hs.transpose(1, 2), memory.transpose(0, 1)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(decoder_layer, num_layers)
        self.query_pos_refine = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            output= layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.query_pos_refine is not None and layer_id < self.num_layers-1:
                query_pos = query_pos + self.query_pos_refine[layer_id](query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # tensor[0] is for a camera token (no positional encoding)
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):

    def __init__(self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

from einops import rearrange

class TemporalEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = _get_clones(encoder_layer, num_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        return output
    
class TemporalAttentionLayer(nn.Module):
    def __init__(self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu") -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.nhead = nhead
        self.to_qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(model_dim, model_dim)
        # rel pos
        self.pos_bias = DynamicPositionBias(model_dim, nhead)
        self.logit_scale = nn.Parameter(torch.log(8 * torch.ones((nhead, 1, 1))), requires_grad=True)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # tensor[0] is for a camera token (no positional encoding)
        return tensor if pos is None else tensor + pos
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        T,B,J,d = src.shape

        # self attention
        src = rearrange(src, 't b j d -> j (t b) d')
        q = k = self.with_pos_embed(src, pos)
        src1 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src1)
        src = self.norm1(src)

        # temporal attention
        src = rearrange(src, 'j (t b) d -> (b j) t d', t=T)

        qkv= self.to_qkv(src)
        qkv = rearrange(qkv, 'bj t (k h d) -> k bj h t d', k=3, h = self.nhead)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(20.)).item()).exp()
        attn = attn * logit_scale

        attn_bias = self.pos_bias(T, device=src.device, dtype=src.dtype)
        attn = attn + attn_bias.unsqueeze(0)

        attn = self.softmax(attn)
        src2 = attn @ v
        src2 = rearrange(src2, 'bj h t d -> bj t (h d)')
        src2 = self.proj(src2)

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = rearrange(src, '(b j) t d -> t b j d', b=B, j=J)

        src3 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src3)
        src = self.norm3(src)

        return src
        

class DynamicPositionBias(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()

        self.cpb_mlp = nn.Sequential(nn.Linear(1, dim),
                                     nn.LayerNorm(dim),
                                     nn.SiLU(),
                                     nn.Linear(dim, heads, bias=False))

    def forward(self, n, device, dtype):
        i = torch.arange(n, device = device)
        j = torch.arange(n, device = device)

        indices = rearrange(i, 'i -> i 1') - rearrange(j, 'j -> 1 j')
        indices += (n - 1)

        pos = torch.arange(-n + 1, n, device = device, dtype = dtype)
        pos = rearrange(pos, '... -> ... 1')

        pos = self.cpb_mlp(pos)

        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_transformer(transformer_config):
    return Transformer(**transformer_config)

def build_temporal_encoder(transformer_config):
    encoder_layer = TemporalAttentionLayer(transformer_config['d_model'], transformer_config['nhead'], transformer_config['dim_feedforward'], transformer_config['dropout'], transformer_config['activation'])
    return TemporalEncoder(encoder_layer, transformer_config['num_encoder_layers'])