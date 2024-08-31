import torch
import torch.nn as nn


class MemoryMomentum(nn.Module):
    def __init__(self, node_num, feature_dim, **kwargs):
        super(MemoryMomentum, self).__init__(**kwargs)
        self.register_buffer('memory', torch.rand(node_num, feature_dim))
        # self.memory = nn.Parameter(torch.rand(node_num, feature_dim), requires_grad=False)
    
    def momentum_update(self, proto: torch.Tensor, keep_coefficient=1.0,
                        synchronize_memory=False, warming_up=True, momentum_flag=False):
        if synchronize_memory:
            self.memory.data = proto.detach_().data
            print(self.memory)
        elif warming_up:
            pass
        elif momentum_flag:
            memory_updated = (1 - keep_coefficient) * proto + keep_coefficient * self.memory
            self.memory.data = memory_updated.detach_().data
            # print(memory_updated)
        else:
            pass
    
    def forward(self, inputs: torch.Tensor, keep_coefficient=1.0, synchronize_memory=False, warming_up=True, momentum_flag=False, **kwargs):
        proto = inputs
        # print(proto.shape)
        self.momentum_update(proto, keep_coefficient=keep_coefficient, warming_up=warming_up, 
                             synchronize_memory=synchronize_memory, momentum_flag=momentum_flag)
        return self.memory * 1.0
