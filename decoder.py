import math
import copy
from typing import Optional, List
import pickle as cp
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# Helper module to clone a layer
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Layer Normalization module with an epsilon for numerical stability
class LayerNormEpsilon(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x - mean) / std + self.bias

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        #  Attention Layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward Layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
    
        # Layer Normalization Layers
        self.norm1 = LayerNormEpsilon(d_model)
        self.norm2 = LayerNormEpsilon(d_model)
        
        # Activation function and boolean flag for normalization
        self.activation = F.relu
        self.normalize_before = normalize_before

    # RoPE
    def pos_embed(self, tensor, pos):
        if pos is None:
            return tensor

        d_model = tensor.size(-1)
        omega = math.pi / d_model  # Base angle for rotation

        # Create rotation matrices for each position
        rotations = torch.zeros((tensor.size(0), pos.size(1), d_model, d_model), device=tensor.device)
        for i in range(d_model):
            if i % 2 == 0:
                rotations[:, :, i, i+1] = torch.sin(pos * i * omega)
            else:
                rotations[:, :, i, i-1] = torch.cos(pos * i * omega)

        # Rotate the tensor with position-specific rotation matrices
        encoding = torch.einsum('btf,bfd->bfd', tensor, rotations)
        return encoding

    
    def forward(self, tgt, memory,
                  tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None,
                  tgt_key_padding_mask: Optional[Tensor] = None,
                  memory_key_padding_mask: Optional[Tensor] = None,
                  pos: Optional[Tensor] = None,
                  query_pos: Optional[Tensor] = None,
                  residual=True):

        # Applying layer normalization before attention if specified
        if self.normalize_before:
            tgt = self.norm1(tgt)

        # Performing self-attention using the first MHA layer
        q = k = self.pos_embed(tgt, query_pos)
        tgt2, ws = self.self_attn(q, k, value=tgt, attn_mask = tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
        
        # Adding a residual connection and applying dropout
        tgt = tgt + tgt2 if residual else tgt2
        tgt = self.dropout(tgt)

        # Applying layer normalization after attention, if specified
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        # Applying encoder-decoder attention using the second MHA layer
        tgt2, ws = self.multihead_attn(query=self.pos_embed(tgt, query_pos),
                                    key=self.pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)
        
        # Adding residual connection and applying dropout
        tgt = tgt + tgt2 if residual else tgt2
        tgt = self.dropout(tgt)
    
        # Applying layer normalization
        tgt = self.norm2(tgt)
        
        # Feedforward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
        # Adding residual connection and applying dropout
        tgt = tgt + tgt2 if residual else tgt2
        tgt = self.dropout(tgt)
        return tgt, ws

class TransformerDecoder(nn.Module):
    
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # Creating the decoding layers
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # Initializing the output with the target sequence
        output = tgt
        # Get the dimensions of the memory tensor
        T, B, C = memory.shape
        # Lists to store intermediate outputs and attention weights
        intermediate, attn_layers = [], []

        # Passing the input through each decoder layer
        for n, layer in enumerate(self.layers):
            residual = True
            # Passing the current decoder layer with input, memory, and masks
            output, ws = layer(output, memory, tgt_mask, memory_mask,
                               tgt_key_padding_mask, memory_key_padding_mask,
                               pos, query_pos, residual)
            # Store the attention weights from this layer
            attn_layers.append(ws)

            # Storing the intermediate outputs
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # Applying layer normalization to the final decoder output
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
            
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output, attn_layers