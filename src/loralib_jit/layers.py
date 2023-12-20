import math
from typing import Optional, List

import jittor as jt
from jittor import nn
from jittor import init

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


def _conv1d_jit(x, weight, groups):
    padding = [0, 0]
    dilation = [1, 1]
    stride = [1, 1]
    out_channels = weight.shape[0]

    # Convert to 2d conv
    x = x.unsqueeze(-1)
    weight = weight.unsqueeze(-1)
    
    # Copyed from jittor.nn.Conv2d
    if groups == 1:
        N,C,H,W = x.shape
        Kh, Kw = weight.shape[-2:]
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        assert oh>0 and ow>0
        with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
            xx = x.reindex([N,out_channels,C,oh,ow,Kh,Kw], [
                'i0', # Nid
                'i2', # Cid
                f'i3*{stride[0]}-{padding[0]}+i5*{dilation[0]}', # Hid+Khid
                f'i4*{stride[1]}-{padding[1]}+i6*{dilation[1]}', # Wid+KWid
            ])
            ww = weight.broadcast(xx.shape, [0,3,4])
            yy = xx*ww
            y = yy.sum([2,5,6]) # Kc, Kh, Kw
    else:
        N,C,H,W = x.shape
        Kh, Kw = weight.shape[-2:]
        G = groups
        CpG = C // G # channels per group
        oc = out_channels
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        assert oh>0 and ow>0
        xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
            'i0', # Nid
            f'i1*{CpG}+i3', # Gid
            f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
            f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
        ])
        # w: [oc, CpG, Kh, Kw]
        ww = weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
            f'i1*{oc//G}+i2',
            'i3',
            'i6',
            'i7'
        ])
        ww.compile_options = xx.compile_options = {"G":G,"C":C}
        yy = xx*ww
        y = yy.reindex_reduce('add', [N, oc, oh, ow], [
            'i0',
            f'i1*{oc//G}+i2',
            'i4',
            'i5'
        ])
    
    return y.squeeze(-1)

class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, )
            ).view(len(enable_lora), -1).bool()
            self.lora_ind[jt.array(enable_lora), :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight = self.weight.transpose(0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self) # Implement the parameter reset as follow (adapted from jittor.nn.Linear):
        self.weight = init.invariant_uniform((self.out_features, self.in_features), "float32")
        bound = 1.0/math.sqrt(self.in_features)
        self.bias = init.uniform((self.out_features,), "float32",-bound,bound) if self.bias is not None else None

        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.constant_(self.lora_B, value=0.0)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = _conv1d_jit(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight += self.merge_AB() * self.scaling
                self.merged = True        

    def execute(self, x: jt.Var):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return nn.linear(x, T(self.weight), bias=self.bias)
        else:
            result = nn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().transpose(0, 1)) * self.scaling
            return result