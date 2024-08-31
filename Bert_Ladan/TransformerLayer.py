import torch
import torch.nn as nn
from torch.nn.modules import MultiheadAttention
import math

hidden_size = 768

class MHAttention(nn.Module):
    def __init__(self, multihead=False):
        super(MHAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.num_attention_heads = 12
        if multihead:
            self.attention_head_size = int(hidden_size / self.num_attention_heads)  # 12*64=768
        else:
            self.attention_head_size = hidden_size
        self.all_head_size = hidden_size
        self.multihead = multihead

    def transpose_for_scores(self, x):
        # print(x.size())
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, toks, masks:torch.Tensor):
        # mask [batch_size, sentence_len]
        masks = masks.float()
        mask = torch.bmm(masks.unsqueeze(dim=-1), masks.unsqueeze(dim=1))
        mask = (1.0 - mask) * -1e31
        if self.multihead:
            mixed_query_layer = self.query(toks)
            mixed_key_layer = self.key(toks)
            mixed_value_layer = self.value(toks)
            # law2accu,accu2law,law2term,term2law
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
        else:
            query_layer = self.query(toks)
            key_layer = self.key(toks)
            value_layer = self.value(toks)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print(attention_scores.size())
        # print(mask.size())
        attention_scores = attention_scores + mask.unsqueeze(dim=1).repeat(1, self.num_attention_heads, 1, 1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if self.multihead:
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.contiguous()
        return context_layer
    

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertCrossOutput(nn.Module):
    def __init__(self):
        super(BertCrossOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class TInterAttention(nn.Module):
    # 一个完整的transformer layer： multi-head self-attention + fead-forward 
    def __init__(self, m=True):
        super(TInterAttention, self).__init__()
        self.cross = MHAttention(multihead=m)
        self.output = BertCrossOutput()

    def forward(self, toks, masks:torch.Tensor):
        # masks [batch_size, sentence_len]
        cross_output = self.cross(toks, masks) # compute the neighbor aggregation
        attention_output = self.output(cross_output, toks)
        attention_output = attention_output * masks.unsqueeze(dim=-1)
        return attention_output

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        intermediate_size = 3072
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        intermediate_size = 3072
        hidden_dropout_prob = 0.1
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, m=True):
        super(BertLayer, self).__init__()
        self.attention = TInterAttention(m=m)
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, toks, masks):
        attention_output = self.attention(toks, masks)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# nn.Transformer
class TransformerFeatureWithLabel(nn.Module):
    def __init__(self, feature_dim, nhead, dropout):
        super().__init__()
        self.nhead = nhead
        self.self_atten = MultiheadAttention(feature_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, Transformer_input, mask:torch.Tensor=None, **kwargs):
        # mask: [batch_size, sentence_num] --> 
        Transformer_input = Transformer_input.transpose(0, 1) # [seq_len, batch_size, feature_dim]
        mask = mask.unsqueeze(dim=-1) @ mask.unsqueeze(dim=1)
        mask = mask.repeat(self.nhead, 1, 1) #[batch_size*head_num, sentence_num, sentence_num]
        Transformer_out: torch.Tensor = self.self_atten(Transformer_input, Transformer_input,
                                                        Transformer_input, attn_mask=mask)[0]
        # [seq_len, batch_size, feature_dim]
        Transformer_out = Transformer_out.transpose(0, 1)
        # [batch_size, seq_len, feature_dim]
        
        Transformer_out = Transformer_input.transpose(0, 1) + self.dropout1(Transformer_out)
        Transformer_out = self.norm1(Transformer_out)
        
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(Transformer_out))))
        Transformer_out = Transformer_out + self.dropout2(src2)
        Transformer_out = self.norm2(Transformer_out)
        
        return Transformer_out
        # CLS_output = Transformer_out[:, :CLS_num, :]
        # law_output = Transformer_out[:, CLS_num:(CLS_num+law_num), :]
        # fact_output = Transformer_out[:, (CLS_num+law_num):, :]
        
        # return CLS_output, law_output, fact_output
