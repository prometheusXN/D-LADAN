import torch
import torch.nn as nn


class CosClassifier(nn.Module):
    def __init__(self, class_num, feature_dim, with_proto=False, **kwargs):
        super(CosClassifier, self).__init__(**kwargs)
        self.out_dim = class_num
        self.with_proto = with_proto
        
        self.proto = nn.Parameter(torch.rand(class_num, feature_dim))
        self.Scalar = nn.Parameter(torch.tensor([1.0]))
        
    def forward(self, inputs: torch.Tensor):
        inputs = inputs.unsqueeze(dim=1)  # [batch_size, 1, feature_dim]
        weights = self.proto.unsqueeze(dim=0)  # [1, class_num, feature_dim]
        sim_matrix = torch.cosine_similarity(inputs, weights, dim=-1)  # [batch_size, class_num]
        
        sim_matrix = sim_matrix * self.Scalar * 10.0
        if self.with_proto:
            prototype = self.proto.detach().clone()
            return sim_matrix, prototype
        else:
            return sim_matrix
