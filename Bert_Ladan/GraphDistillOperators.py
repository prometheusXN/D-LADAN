import torch
import torch.nn as nn

def softmax_with_mask(logits, masks=None, dim=-1):
    if masks is not None:
        logits = logits + (1-masks) * (-1e32)
    score = torch.softmax(logits, dim=dim)
    return score


class GraphDistillOperator(nn.Module): # [node_num, 768] -> [node_num, 512]
    def __init__(self, config, input_dim, activation=True, withAgg=False):
        super(GraphDistillOperator, self).__init__()
        self.withAgg = withAgg
        self.activation = activation
        self.activation_fuc = nn.Tanh()
        self.dropout = nn.Dropout(p=config.GraphDistill.dropout)
        
        self.input_dim = input_dim # 768
        self.out_dim = config.net.hidden_size # 512
        
        self.distill_dence = nn.Linear(self.input_dim * 2, self.out_dim)
        self.distill_out_dence = nn.Linear(self.input_dim, self.out_dim)
        
        if self.withAgg:
            self.aggregate_dense = nn.Linear(self.input_dim, self.out_dim)
            self.aggregate_out_dense = nn.Linear(self.input_dim, self.out_dim)
        
    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor):
        # features [node_num, feature_dim]
        adj_matrix = adj_matrix.float()
        node_num, feature_dim = features.shape
        head_features = features.unsqueeze(dim=1).repeat(1, node_num, 1)
        tail_features = features.unsqueeze(dim=0).repeat(node_num, 1, 1)
        neight_features = torch.cat([head_features, tail_features], dim=-1)
        # [node_num, node_num, feature_dim]
        
        neight_features_sum = torch.sum(adj_matrix.unsqueeze(dim=-1) * neight_features, dim=1)
        neigh_mask = torch.max(adj_matrix, dim=-1, keepdim=True).values
    
        neigh_num = adj_matrix.sum(dim=-1, keepdims=True) + (1 - neigh_mask) * 1
        neight_features_ave = neight_features_sum / neigh_num

        neigh_features = self.distill_dence(neight_features_ave)
        feature_updated: torch.Tensor = self.distill_out_dence(features) - neigh_features
        feature_updated = feature_updated.reshape([node_num, self.out_dim])
        if self.activation:
            feature_updated = self.activation_fuc(feature_updated)
        feature_updated = self.dropout(feature_updated)
        
        if self.withAgg:
            neighbor_features_aggregate = self.aggregate_dense(adj_matrix @ features)
            feature_aggregate: torch.Tensor = self.aggregate_out_dense(features) + neighbor_features_aggregate
            feature_aggregate = feature_aggregate.reshape([node_num, self.out_dim])
            feature_aggregate = self.activation(feature_aggregate)
            feature_aggregate = self.dropout(feature_aggregate)
            return feature_updated, feature_aggregate
        else:
            return feature_updated, feature_updated
        

class GraphDistillOperatorWithEdgeWeight(nn.Module): # [node_num, ]
    def __init__(self, config, input_dim, activation=True, withAgg=False):
        super(GraphDistillOperatorWithEdgeWeight, self).__init__()
        self.withAgg = withAgg
        self.activation = activation
        self.activation_fuc = nn.Tanh()
        self.dropout = nn.Dropout(p=config.GraphDistill.dropout)
        
        self.input_dim = input_dim # 256
        self.out_dim = config.net.hidden_size # 256
        
        self.distill_dence = nn.Linear(self.input_dim * 2, self.out_dim)
        self.distill_out_dence = nn.Linear(self.input_dim, self.out_dim)
        
        if self.withAgg:
            self.aggregate_dense = nn.Linear(self.input_dim, self.out_dim)
            self.aggregate_out_dense = nn.Linear(self.input_dim, self.out_dim)
            
    def forward(self, features: torch.Tensor, key_features: torch.Tensor, adj_matrix: torch.Tensor):
        # features [node_num, feature_dim]
        adj_matrix = adj_matrix.float()
        node_num, feature_dim = features.shape
        head_features = features.unsqueeze(dim=1).repeat(1, node_num, 1)
        tail_features = features.unsqueeze(dim=0).repeat(node_num, 1, 1)
        neigh_features_distill = torch.cat([head_features, tail_features], dim=-1)
        # [node_num, node_num, feature_dim]
        
        self_loop_mask = 1.0 - torch.eye(n=node_num, dtype=torch.float)
        self_loop_mask = self_loop_mask.to(adj_matrix.device)
        adj_matrix_soft = softmax_with_mask(adj_matrix * 5.0, masks=self_loop_mask, dim=-1) # Gumbal Softmax
        neight_features_norm = torch.sum(adj_matrix_soft.unsqueeze(dim=-1) * neigh_features_distill, dim=1)
        
        neigh_features = self.distill_dence(neight_features_norm)
        feature_updated: torch.Tensor = self.distill_out_dence(features) - neigh_features
        feature_updated = feature_updated.reshape([node_num, self.out_dim])
        
        if self.activation:
            feature_updated = self.activation_fuc(feature_updated)
        feature_updated = self.dropout(feature_updated)
        
        if self.withAgg:
            neighbor_features_aggregate = self.aggregate_dense(adj_matrix @ key_features)
            feature_aggregate: torch.Tensor = self.aggregate_out_dense(key_features) + neighbor_features_aggregate
            feature_aggregate = feature_aggregate.reshape([node_num, self.out_dim])
            feature_aggregate = self.activation(feature_aggregate)
            feature_aggregate = self.dropout(feature_aggregate)
            return feature_updated, feature_aggregate
        else:
            return feature_updated, feature_updated
