from transformers import BertConfig, BertTokenizer, BertModel
import numpy as np
import torch.nn as nn
import torch
from AttenRNN import RNN, AttentionOriContext, ContextAttention
from GraphDistillOperators import GraphDistillOperator, GraphDistillOperatorWithEdgeWeight
from TransformerLayer import TransformerFeatureWithLabel, BertLayer
from common_utils import dynamic_partition
import torch.nn.functional as F


class Ladan(nn.Module):
    def __init__(self, config, group_num, accu_relation=1, **kwargs):
        super(Ladan, self).__init__(**kwargs)

        self.config = config
        self.law_sentence_len = config.train.law_sentence_len
        self.fact_sentence_len = config.train.fact_sentence_len
        self.fact_sentence_num = config.train.fact_sentence_num
        self.use_mean = config.train.use_mean_pooling
        self.group_num = group_num
        self.num_distill_layers = self.config.net.num_distill_layers
        self.accu_relation = accu_relation
        print("model_path: ", self.config.train.pretrain_model_path_1)
        self.bert_config = BertConfig.from_pretrained(config.train.pretrain_model_path_1)
        self.PLM_model = BertModel.from_pretrained(config.train.pretrain_model_path_1, config=self.bert_config)
        
        # 'define_encoder_base' 
        self.sentence_encoder = BertLayer(m=True)
        self.sentence_attention = AttentionOriContext(config=config, input_dim=config.net.bert_size)
        
        # 'define_distiller_prior'
        self.graph_distillers_prior = []
        graph_input_piror = config.net.hidden_size
        for i in range(self.num_distill_layers):
            distill_layer = GraphDistillOperator(config, input_dim=graph_input_piror)
            self.graph_distillers_prior.append(distill_layer)
            graph_input_piror = distill_layer.out_dim
        self.graph_distillers_prior = nn.ModuleList(self.graph_distillers_prior)

        # 'context_generator_prior'
        self.group_chosen_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.group_chosen = nn.Linear(self.hidden_size, self.group_num)
        self.context_s_prior = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # 'define_encoder_prior'
        self.sentence_encoder_prior = BertLayer(m=True)
        self.sentence_attention_prior = ContextAttention(config=config, input_dim=config.net.bert_size)
    
    def basic_encoding(self, input_ids, attention_mask, token_type_ids, using_pooling=True) -> torch.Tensor:
        """
        :param input_ids: [batch_size, max_length]
        :param attention_mask: [batch_size, max_length]
        :param using_pooling:
        :return: the embedding of each sentence.
        [batch_size, hidden_size]
        """

        def mean_pooling(input_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            :param input_emb: [batch_size, max_length, hidden_size]
            :param mask: [batch_size, max_length]
            :return: [batch_size, hidden_size]
            """
            s = torch.sum(input_emb * mask.unsqueeze(dim=-1).float(), dim=1)  # [batch_size, hidden_size]
            d = mask.sum(dim=1, keepdim=True).float() + (1e-9)
            return s / d

        output = self.PLM_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # return a tuple of Tensor -> sequence output, the [CLS] embedding with tanh
        # [last_hidden_states, pooler_output]
        last_hidden_states = output[0]  # [batch_size, max_length, hidden_size]
        if using_pooling:
            mean_emb = mean_pooling(input_emb=last_hidden_states, mask=attention_mask)
            return mean_emb, last_hidden_states
        else:
            cls_emb = last_hidden_states[:, 0, :].squeeze()
            # cls_emb = self.emb_dropout(cls_emb)# pooler_output
            return cls_emb, last_hidden_states
    
    def get_basic_embedding(self, input_ids:torch.Tensor, attention_mask:torch.Tensor,
                            token_type_ids:torch.Tensor, max_length):
        """
        Args:
            input_ids (_type_): [batch_size, ]
            attention_mask (_type_): []
        """
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, max_length) # [batch_size * sentence_num, sentence_len] or []
        attention_mask = attention_mask.view(-1, max_length)
        token_type_ids = token_type_ids.view(-1, max_length)
        ouputs, last_hidden_states = self.basic_encoding(input_ids, attention_mask, token_type_ids, using_pooling=self.use_mean)
        return ouputs, last_hidden_states
    
    def distill_pooling(self, features, group_index):
        node_num, feature_dim = features.shape
        features_grouped = dynamic_partition(features, group_index, num_partitions=self.group_num)

        group_contexts = []
        for i in range(self.group_num):
            u = torch.max(features_grouped[i], 0).values  # law_representation[i]: [n, law_size]
            u_2 = torch.min(features_grouped[i], 0).values
            group_contexts.append(torch.cat([u, u_2], dim=-1))
            # size: [graph_num, 2*law_size] whether this u can use attention to get
        group_contexts = torch.reshape(torch.cat(group_contexts, 0), (-1, 2 * feature_dim))
        # group_contexts = self.dropout(group_contexts)
        return group_contexts
    
    def re_encoding_fact(self, name, inputs, dense_funcs, context_funcs, encoder_funcs, masks):

        fact_base, key_list, context_list, fact_rep_sentences, fact_sentence_level = inputs
        law_Dense, fact_Dense = dense_funcs
        group_chosen, Transformer, context_generation_s = context_funcs
        sentence_re_encoder, sentence_reattention = encoder_funcs
        sentence_mask, sentence_mask_1, real_mask = masks

        group_pred_scores = group_chosen(self.group_chosen_hidden(fact_base))
        group_pred = torch.softmax(group_pred_scores, dim=-1)
        re_context = group_pred @ context_list

        context_sentence = torch.reshape(context_generation_s(re_context), shape=(-1, self.config.net.hidden_size))
        # 'Encoding'
        re_fact_sentence_level = sentence_re_encoder(fact_rep_sentences, sentence_mask)
        fact_prior, score_sentence = sentence_reattention(re_fact_sentence_level, context_sentence, masks=real_mask)

        # fact_prior = tf.concat([fact_prior, CLS_output], axis=-1)
        return fact_prior, re_fact_sentence_level, group_pred_scores, score_sentence

    @staticmethod
    def get_real_mask(masks:torch.Tensor):
        new_mask = masks.clone().to(masks.device)
        new_mask[:, 0] = 0
        sum_mask = masks.sum(dim=-1) # [batch_size]
        one_hot_matrix = F.one_hot(sum_mask-1, masks.shape[-1]).to(masks.device)
        return new_mask - one_hot_matrix
    
    def forward(self, inputs, law_information=None, warming_up=False, 
                fact_attention_mask=None, sentence_mask=None, 
                accu_information=None, time_information=None,
                law_attention_mask=None):
        
        fact_inputs_ids, fact_token_type_ids = inputs
        law_input_ids, law_token_type_ids, adj_matrix_law, group_indexes, law_inputs_posterior, adj_matrix_posterior = law_information 
        real_mask_law = self.get_real_mask(law_attention_mask)
        real_mask_fact = self.get_real_mask(fact_attention_mask)
        # law_encoding_base: [law_num, sentence_len] --> [law_num, feature_dim]
        word_embedding_law = self.PLM_model(input_ids=law_input_ids, attention_mask=law_attention_mask, token_type_ids=law_token_type_ids)[0]
        law_word_level = self.sentence_encoder(word_embedding_law, masks=law_attention_mask)
        law_base, _ = self.sentence_attention(law_word_level, masks=real_mask_law)
        # [law_num, hidden_dim] [103, 256]

        # GeneratePriorGroupInformation
        distilled_law_prior = law_base
        for i in range(self.num_distill_layers):
            distilled_law_prior, aggregate_law_prior = self.graph_distillers_prior[i](features=distilled_law_prior, adj_matrix=adj_matrix_law)
        context_list_prior = self.distill_pooling(features=distilled_law_prior, group_index=group_indexes)
            
        # EncodeFact
            # BaseEncodingFact' # [batch_size, sentence_num, sentence_len] --> [batch_size, sentence_num, feature_dim]
        word_embedding_fact = self.PLM_model(fact_inputs_ids, fact_attention_mask, fact_token_type_ids)[0]
        # [batch_size, sentence_num, feature_dim]
        fact_word_level = self.sentence_encoder(word_embedding_fact, sentence_mask)
        fact_base, score_s_base = self.sentence_attention(fact_word_level, masks=real_mask_fact)

            # PriorEncodingFact
        fact_prior, fact_prior_word, group_pred_prior, score_s_prior = \
            self.re_encoding_fact(inputs=[fact_base, None, context_list_prior, word_embedding_fact, None],
                                    dense_funcs=[None, None],
                                    context_funcs=[self.group_chosen, None, self.context_s_prior],
                                    encoder_funcs=[self.sentence_encoder_prior, self.sentence_attention_prior],
                                    masks=[sentence_mask, None, real_mask_fact], name='Prior')

        fact_rep = torch.cat([fact_base, fact_prior], axis=-1)
        law_rep = law_base
        return [fact_rep, law_rep, group_pred_prior, score_s_base, score_s_prior]
    

class Ladan_criminal(nn.Module):
    def __init__(self, config, group_num, accu_relation=1, **kwargs):
        super(Ladan_criminal, self).__init__(**kwargs)

        self.config = config
        self.law_sentence_len = config.train.law_sentence_len
        self.fact_sentence_len = config.train.fact_sentence_len
        self.use_mean = config.train.use_mean_pooling
        self.group_num = group_num
        print("group_num:", self.group_num)
        self.num_distill_layers = self.config.net.num_distill_layers
        self.accu_relation = accu_relation
        print("model_path: ", self.config.train.pretrain_model_path_1)
        self.bert_config = BertConfig.from_pretrained(config.train.pretrain_model_path_1)
        self.PLM_model = BertModel.from_pretrained(config.train.pretrain_model_path_1, config=self.bert_config)
            
        # 'define_encoder_base' 
        self.sentence_encoder = BertLayer(m=True)
        self.sentence_attention = AttentionOriContext(config=config, input_dim=config.net.bert_size)

        # 'define_distiller_prior'
        self.graph_distillers_prior = []
        graph_input_piror = config.net.hidden_size
        for i in range(self.num_distill_layers):
            distill_layer = GraphDistillOperator(config, input_dim=graph_input_piror)
            self.graph_distillers_prior.append(distill_layer)
            graph_input_piror = distill_layer.out_dim
        self.graph_distillers_prior = nn.ModuleList(self.graph_distillers_prior)

        # 'context_generator_prior'
        self.group_chosen_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.group_chosen = nn.Linear(self.hidden_size, self.group_num)
        self.context_s_prior = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 'define_encoder_prior'
        self.sentence_encoder_prior = BertLayer(m=True)
        self.sentence_attention_prior = ContextAttention(config=config, input_dim=config.net.bert_size)

    def basic_encoding(self, input_ids, attention_mask, token_type_ids, using_pooling=True) -> torch.Tensor:
        """
        :param input_ids: [batch_size, max_length]
        :param attention_mask: [batch_size, max_length]
        :param using_pooling:
        :return: the embedding of each sentence.
        [batch_size, hidden_size]
        """

        def mean_pooling(input_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            :param input_emb: [batch_size, max_length, hidden_size]
            :param mask: [batch_size, max_length]
            :return: [batch_size, hidden_size]
            """
            s = torch.sum(input_emb * mask.unsqueeze(dim=-1).float(), dim=1)  # [batch_size, hidden_size]
            d = mask.sum(dim=1, keepdim=True).float() + (1e-9)
            return s / d

        output = self.PLM_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # return a tuple of Tensor -> sequence output, the [CLS] embedding with tanh
        # [last_hidden_states, pooler_output]
        last_hidden_states = output[0]  # [batch_size, max_length, hidden_size]
        if using_pooling:
            mean_emb = mean_pooling(input_emb=last_hidden_states, mask=attention_mask)
            return mean_emb, last_hidden_states
        else:
            cls_emb = last_hidden_states[:, 0, :].squeeze()
            # cls_emb = self.emb_dropout(cls_emb)# pooler_output
            return cls_emb, last_hidden_states
    
    def get_basic_embedding(self, input_ids:torch.Tensor, attention_mask:torch.Tensor,
                            token_type_ids:torch.Tensor, max_length):
        """
        Args:
            input_ids (_type_): [batch_size, ]
            attention_mask (_type_): []
        """
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, max_length) # [batch_size * sentence_num, sentence_len] or []
        attention_mask = attention_mask.view(-1, max_length)
        token_type_ids = token_type_ids.view(-1, max_length)
        ouputs, last_hidden_states= self.basic_encoding(input_ids, attention_mask, token_type_ids, using_pooling=self.use_mean)
        return ouputs, last_hidden_states
    
    def distill_pooling(self, features, group_index):
        node_num, feature_dim = features.shape
        features_grouped = dynamic_partition(features, group_index, num_partitions=self.group_num)

        group_contexts = []
        for i in range(self.group_num):
            u = torch.max(features_grouped[i], 0).values  # law_representation[i]: [n, law_size]
            u_2 = torch.min(features_grouped[i], 0).values
            group_contexts.append(torch.cat([u, u_2], dim=-1))
            # size: [graph_num, 2*law_size] whether this u can use attention to get
        group_contexts = torch.reshape(torch.cat(group_contexts, 0), (-1, 2 * feature_dim))
        # group_contexts = self.dropout(group_contexts)
        return group_contexts
    
    def re_encoding_fact(self, name, inputs, dense_funcs, context_funcs, encoder_funcs, masks):

        fact_base, key_list, context_list, fact_rep_sentences, fact_sentence_level = inputs
        law_Dense, fact_Dense = dense_funcs
        group_chosen, Transformer, context_generation_s = context_funcs
        sentence_re_encoder, sentence_reattention = encoder_funcs
        sentence_mask, sentence_mask_1, real_mask = masks

        group_pred_scores = group_chosen(self.group_chosen_hidden(fact_base))
        group_pred = torch.softmax(group_pred_scores, dim=-1)
        re_context = group_pred @ context_list

        context_sentence = torch.reshape(context_generation_s(re_context), shape=(-1, self.config.net.hidden_size))
        # 'Encoding'
        re_fact_sentence_level = sentence_re_encoder(fact_rep_sentences, sentence_mask)
        fact_prior, score_sentence = sentence_reattention(re_fact_sentence_level, context_sentence, masks=real_mask)

        # fact_prior = tf.concat([fact_prior, CLS_output], axis=-1)
        return fact_prior, re_fact_sentence_level, group_pred_scores, score_sentence
    
    @staticmethod
    def get_real_mask(masks:torch.Tensor):
        new_mask = masks.clone().to(masks.device)
        new_mask[:, 0] = 0
        sum_mask = masks.sum(dim=-1) # [batch_size]
        one_hot_matrix = F.one_hot(sum_mask-1, masks.shape[-1]).to(masks.device)
        return new_mask - one_hot_matrix

    def forward(self, inputs, law_information=None, warming_up=False, 
                fact_attention_mask=None, sentence_mask=None, 
                law_attention_mask=None):
        
        fact_inputs_ids, fact_token_type_ids = inputs
        law_input_ids, law_token_type_ids, adj_matrix_law, group_indexes, accu_inputs_posterior, adj_matrix_posterior = law_information
        
        # law_encoding_base: [law_num, sentence_len] --> [law_num, feature_dim]
        real_mask_law = self.get_real_mask(law_attention_mask)
        real_mask_fact = self.get_real_mask(fact_attention_mask)
        word_embedding_law = self.PLM_model(input_ids=law_input_ids, attention_mask=law_attention_mask, token_type_ids=law_token_type_ids)[0]
        law_word_level = self.sentence_encoder(word_embedding_law, masks=law_attention_mask)
        law_base, _ = self.sentence_attention(law_word_level, masks=real_mask_law)

        # GeneratePriorGroupInformation
        distilled_law_prior = law_base
        for i in range(self.num_distill_layers):
            distilled_law_prior, aggregate_law_prior = self.graph_distillers_prior[i](features=distilled_law_prior, adj_matrix=adj_matrix_law)
        context_list_prior = self.distill_pooling(features=distilled_law_prior, group_index=group_indexes)
        
        # EncodeFact
            # BaseEncodingFact' # [batch_size, sentence_num, sentence_len] --> [batch_size, sentence_num, feature_dim]
        word_embedding_fact = self.PLM_model(fact_inputs_ids, fact_attention_mask, fact_token_type_ids)[0]
        # [batch_size, sentence_num, feature_dim]
        fact_word_level = self.sentence_encoder(word_embedding_fact, sentence_mask)
        fact_base, score_s_base = self.sentence_attention(fact_word_level, masks=real_mask_fact)
                
            # PriorEncodingFact
        fact_prior, fact_prior_word, group_pred_prior, score_s_prior = \
            self.re_encoding_fact(inputs=[fact_base, None, context_list_prior, word_embedding_fact, None],
                                    dense_funcs=[None, None],
                                    context_funcs=[self.group_chosen, None, self.context_s_prior],
                                    encoder_funcs=[self.sentence_encoder_prior, self.sentence_attention_prior],
                                    masks=[sentence_mask, None, real_mask_fact], name='Prior')

        fact_rep = torch.cat([fact_base, fact_prior], axis=-1)
        law_rep = law_base
        return [fact_rep, law_rep, group_pred_prior, score_s_base, score_s_prior]
            
         
