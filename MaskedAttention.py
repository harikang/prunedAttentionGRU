import torch.nn as nn 
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import math 
class MaskedAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(MaskedAttention, self).__init__()
        self.query = MaskedLinear(hidden_dim, attention_dim)
        self.key = MaskedLinear(hidden_dim, attention_dim)
        self.value = MaskedLinear(hidden_dim, attention_dim)
        self.context_vector = MaskedLinear(attention_dim, 1, bias=False)


    def forward(self, hidden_states):
        query = self.query(hidden_states)  
        key = self.key(hidden_states)  
        value = self.value(hidden_states)  

        # Compute attention scores using a combination of query and key tensors
        attn_scores = torch.tanh(query + key)  
        # Transform attention scores to a single dimension and squeeze the last dimension
        attn_scores = self.context_vector(attn_scores).squeeze(-1)  

        # Apply softmax to obtain attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  

        # Compute weighted hidden states using the attention weights
        weighted_hidden_states = value * attn_weights.unsqueeze(-1) 

        # Sum the weighted hidden states along the sequence length to get the context vector
        context_vector = weighted_hidden_states.sum(dim=1)  

        return context_vector
    
    def prune(self, threshold, k):
        weight_dev = self.query.weight.device
        for linear in [self.query, self.key, self.value, self.context_vector]:
            tensor = linear.weight.data.cpu().numpy()
            mask = linear.mask.data.cpu().numpy()
            new_mask = np.where(abs(tensor) < threshold, 0, mask)

            nz_count = np.count_nonzero(new_mask)
            if k <= nz_count / (linear.in_features * linear.out_features):
                linear.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                linear.mask.data = torch.from_numpy(new_mask).to(weight_dev)
            else:
                return False
        return True
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_check = bias
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_check:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.ones_(self.mask)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def prune(self, threshold, k):
        weight_dev = self.weight.device
        mask_dev = self.mask.device

        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)

        nz_count = np.count_nonzero(new_mask)
        if k <= nz_count / (self.in_features * self.out_features):
            self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            self.mask.data = torch.from_numpy(new_mask).to(mask_dev)
            return True
        else:
            return False
        
    def prune(self, threshold, k):
        weight_dev = self.weight.device
        mask_dev = self.mask.device

        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)

        nz_count = np.count_nonzero(new_mask)
        if k <= nz_count / (self.in_features * self.out_features):
            self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            self.mask.data = torch.from_numpy(new_mask).to(mask_dev)
            return True
        else:
            return False