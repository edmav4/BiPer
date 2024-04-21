import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from utils.options import args
torch.manual_seed(0)
np.random.seed(0)

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(self.weight.size(0), 1, 1), requires_grad=False)
        self.register_buffer('tau', torch.tensor(1.))

        self.freq = args.freq
        self.stage2 = args.stage2
        # print(f"using stage2: {self.stage2}")
        # print(self.freq)

    def forward(self, input):
        a = input
        w = self.weight

        if self.training:
            a0 = a / torch.sqrt(a.var([1,2,3], keepdim=True) + 1e-5)
        else:
            a0 = a

        #* binarize
        if not self.stage2:
            bw = torch.sin(self.freq * w)
        else:
            bw = BinaryQuantize().apply(torch.sin(self.freq*w))
        ba = BinaryQuantize_a().apply(a0)

        #* 1+bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output

class BinarizeConv2d_32a(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d_32a, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))
        self.freq = args.freq
        self.stage2 = args.stage2
        print(self.freq)

    def forward(self, input):
        a = input
        w = self.weight

        #* binarize
        if not self.stage2:
            bw = torch.sin(self.freq * w)
        else:
            bw = BinaryQuantize().apply(torch.sin(self.freq*w))

        #* 1+bit conv
        output = F.conv2d(a, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input
