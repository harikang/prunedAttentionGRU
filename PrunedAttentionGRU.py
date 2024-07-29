import torch.nn as nn 
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import math 
import random 

from MaskedAttention import *
from prunedGRU import *

class PruningModule(nn.Module):
    def prune_by_std(self, s, k): #가중치의 절대값 표준편차를 기준으로 pruning
        #s 표준편차의 임계값을 설정하는데 사용(e.g. s=0.5, std의 절반을 임계값으로 사용)
        #k pruning ratio를 설정. 
        for name, module in self.named_modules():
            if isinstance(module, (MaskedLinear, CustomGRU, MaskedAttention)):
                self._prune_weights(module, s, k, name)

    def _prune_weights(self, module, s, k, name): 
        for attr_name in ['weight_ih', 'weight_hh', 'weight']:
            if hasattr(module, attr_name):
                weight = getattr(module, attr_name)
                threshold = np.std(weight.data.abs().cpu().numpy()) * s
                print(f'Pruning {attr_name} with threshold: {threshold} for layer {name}')
                while not module.prune(threshold, k):
                    threshold *= 0.99

    def prune_by_random(self, connectivity): #무작위로 pruning
        for name, module in self.named_modules():
            if isinstance(module, (MaskedLinear, CustomGRU, MaskedAttention)):
                self._random_prune_weights(module, connectivity, name)

    def _random_prune_weights(self, module, connectivity, name):
        for attr_name in ['weight_ih', 'weight_hh', 'weight']:
            if hasattr(module, attr_name):
                weight = getattr(module, attr_name)
                print(f'Pruning {attr_name} randomly for layer {name}')
                row = weight.shape[0]
                column = weight.shape[1]
                weight_mask = torch.tensor(self.generate_weight_mask((row, column), connectivity)).float()
                weight_data = nn.init.orthogonal_(weight.data)
                weight_data = weight_data * weight_mask
                weight.data = weight_data

    def generate_weight_mask(self, shape, connection):
        sub_shape = (shape[0], shape[1])
        w = []
        w.append(self.generate_mask_matrix(sub_shape, connection))
        return w[0]

    @staticmethod
    def generate_mask_matrix(shape, connection): #connection으로 설정된 비율에 따라 mask 행렬 생성. 
        random.seed(1)
        s = np.random.uniform(size=shape)
        s_flat = s.flatten()
        s_flat.sort()
        threshold = s_flat[int(shape[0] * shape[1] * (1 - connection))]
        super_threshold_indices = s >= threshold
        lower_threshold_indices = s < threshold
        s[super_threshold_indices] = 1.
        s[lower_threshold_indices] = 0.
        return s
    
# Define the AttentionMaskedGRU
class prunedAttentionGRU(PruningModule):
    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim):
        super(prunedAttentionGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = CustomGRU(input_dim, hidden_dim, bias=True, batch_first=True)
        self.attention = MaskedAttention(hidden_dim, attention_dim)
        self.fc = MaskedLinear(attention_dim, output_dim)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        context_vector = self.attention(gru_out)
        output = self.fc(context_vector)
        return output
    