import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
from .layers import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock
import pdb


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        probs = self.center(x.view(-1, dim))
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]
        return indexes.view(bs, length)


class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token, index_v, index_m):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token
    

class Output_reconstruct(nn.Module):
    def __init__(self, args, d_model):
        super(Output_reconstruct, self).__init__()
        self.proj1 = nn.Conv1d(d_model, args.wave_length, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.proj1(x)
        x = x.reshape(x.shape[0], -1, 1)
        return x


class Pred_heads(nn.Module):
    def __init__(self, d_model, num_class):
        super(Pred_heads, self).__init__()
        self.linear1 = nn.Linear(d_model, 256)
        self.linear2 = nn.Linear(256, num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        return x 


class NANOSSL(nn.Module):
    def __init__(self, args):
        super(NANOSSL, self).__init__()
        self.d_model = args.d_model

        self.momentum = args.momentum
        self.linear_proba = True
        self.device = args.device
        self.data_shape = args.data_shape
        self.max_len = int(self.data_shape[0] / args.wave_length)
        print('max_len: {}'.format(self.max_len))
        self.mask_len = int(args.mask_ratio * self.max_len)
        self.position = PositionalEmbedding(self.max_len, self.d_model)

        self.mask_token = nn.Parameter(torch.randn(self.d_model, ))
        self.input_projection = nn.Conv1d(args.data_shape[1] * args.wave_length, self.d_model, kernel_size=1, stride=1)

        self.encoder = Encoder(args)
        self.momentum_encoder = Encoder(args)
        self.reg = Regressor(self.d_model, args.attn_heads, 4 * self.d_model, 1, args.reg_layers)

        self.output_reconstruct = Output_reconstruct(args, self.d_model)
        self.pred_heads = Pred_heads(self.d_model, args.num_class)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def forward_pretrain(self, x):  
        x0 = x.reshape(x.shape[0], self.max_len, -1)
        x = self.input_projection(x0.transpose(1, 2)).transpose(1, 2).contiguous() 
        tokens = self.tokenizer(x) 
        x += self.position(x)
        rep_mask_token = self.mask_token.repeat(x.shape[0], x.shape[1], 1) + self.position(x)

        index = np.arange(x.shape[1])
        random.shuffle(index)
        v_index = index[:-self.mask_len]
        m_index = index[-self.mask_len:]
        visible = x[:, v_index, :] 
        mask = x[:, m_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]

        rep_visible = self.encoder(visible)
        with torch.no_grad():
            rep_mask = self.momentum_encoder(mask)
        rep_mask_prediction = self.reg(rep_visible, rep_mask_token, v_index, m_index)

        return [rep_mask, rep_mask_prediction]

    def forward(self, x):
        if self.linear_proba:
            with torch.no_grad():
                x = x.reshape(x.shape[0], self.max_len, -1)
                x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
                x += self.position(x)
                x = self.encoder(x)
                return torch.mean(x, dim=1)  # torch.Size([bs, 64])
        else:               # 用于fine-tune
            x = x.reshape(x.shape[0], self.max_len, -1)
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            x += self.position(x)
            x = self.encoder(x)
            x = self.pred_heads(torch.mean(x, dim=1))
            return x
