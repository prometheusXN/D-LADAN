from transformers import BertConfig, BertTokenizer, BertModel
import numpy as np
import torch.nn as nn
import torch
from AttenRNN import RNN, AttentionOriContext, ContextAttention
from GraphDistillOperators import GraphDistillOperator, GraphDistillOperatorWithEdgeWeight
from TransformerLayer import TransformerFeatureWithLabel, BertLayer
from common_utils import dynamic_partition
import torch.nn.functional as F

class dLadan_full(nn.Module):
  '''
  Define the Transformer version of D-LADAN.
  '''
    def __init__(self, config, group_num, accu_relation=1, **kwargs):
        super(dLadan_full, self).__init__(**kwargs)
        
        self.config = config
        self.law_sentence_len = config.train.law_sentence_len
        self.fact_sentence_len = 512
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
        
        # 'define_distiller_posterior'
        self.graph_distillers_posterior = []
        self.hidden_size = config.net.hidden_size
        graph_input_posterior = config.net.hidden_size
        for i in range(self.num_distill_layers):
            distill_layer = GraphDistillOperatorWithEdgeWeight(config, input_dim=graph_input_posterior)  # with edge weight
            self.graph_distillers_posterior.append(distill_layer)
            graph_input_posterior = distill_layer.out_dim
        self.graph_distillers_posterior = nn.ModuleList(self.graph_distillers_posterior)
                
        # 'context_generator_prior'
        self.group_chosen_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.group_chosen = nn.Linear(self.hidden_size, self.group_num)
        self.context_s_prior = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # 'define_encoder_prior'
        self.sentence_encoder_prior = BertLayer(m=True)
        self.sentence_attention_prior = ContextAttention(config=config, input_dim=config.net.bert_size)

        # define_encoder_posterior'
        self.sentence_encoder_posterior = BertLayer(m=True)
        self.sentence_attention_posterior = ContextAttention(config=config, input_dim=config.net.bert_size)
            
        # context_generator_posterior
        self.Transformer_posterior = TransformerFeatureWithLabel(feature_dim=self.hidden_size,
                                                                 nhead=4, dropout=config.GraphDistill.dropout)
        self.matching_law_posterior = nn.Linear(self.hidden_size, self.hidden_size)
        self.matching_fact_posterior = nn.Linear(config.net.bert_size * 2, self.hidden_size)
        self.context_s_posterior = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout=nn.Dropout(config.GraphDistill.dropout)

        if self.accu_relation is not None:
            # build posterior of charge 
            # define_distiller_posterior_accu
            self.graph_distillers_posterior_accu = []
            graph_input_posterior = config.net.hidden_size
            for i in range(self.num_distill_layers):
                distill_layer = GraphDistillOperatorWithEdgeWeight(config, input_dim=graph_input_posterior)
                self.graph_distillers_posterior_accu.append(distill_layer)
                graph_input_posterior = distill_layer.out_dim
            self.graph_distillers_posterior_accu = nn.ModuleList(self.graph_distillers_posterior_accu)

            # define_encoder_posterior_accu
            self.sentence_encoder_posterior_A = BertLayer(m=True)
            self.sentence_attention_posterior_A = ContextAttention(config=config, input_dim=config.net.bert_size)

            # context_generator_posterior_accu
            self.Transformer_posterior_A = TransformerFeatureWithLabel(feature_dim=self.hidden_size, 
                                                                       nhead=4, dropout=config.GraphDistill.dropout)
            self.matching_law_posterior_A = nn.Linear(self.hidden_size, self.hidden_size)
            self.matching_fact_posterior_A = nn.Linear(config.net.bert_size * 2, self.hidden_size)
            self.context_s_posterior_A = nn.Linear(self.hidden_size, self.hidden_size)

        self.posterior_mask = nn.Parameter(torch.zeros([1, self.hidden_size]), requires_grad=False)
        self.posterior_maskF = nn.Parameter(torch.zeros([1, self.hidden_size]), requires_grad=False)
    
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
    
    def get_basic_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            token_type_ids: torch.Tensor, max_length):
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
        
        if name == "Posterior":
            batch_size, word_num, feature_dim = fact_sentence_level.shape
            key_num, feature_dim = key_list.shape
            matching_law_prior:torch.Tensor = law_Dense(key_list)    # part of source input
            matching_fact_prior = fact_Dense(fact_sentence_level)
            label_mask = torch.ones([batch_size, key_num], dtype=torch.float).to(sentence_mask_1.device)
            matching_mask = torch.cat([sentence_mask_1, label_mask], axis=-1)
            matching_mask = torch.reshape(matching_mask, shape=(-1, word_num + key_num))
            
            label_input = matching_law_prior.unsqueeze(dim=0).repeat(batch_size, 1, 1)
            
            Transformer_input = torch.cat([matching_fact_prior, label_input], dim=1)
            # CLS_output, label_output, fact_output = Transformer(CLS_input, label_input, matching_fact_prior, mask=matching_mask)
            Transformer_out = Transformer(Transformer_input, mask=matching_mask)
            
            CLS_output = Transformer_out[:, 0, :].unsqueeze(dim=1)
            fact_output = Transformer_out[:, 1:word_num, :]
            law_output: torch.Tensor = Transformer_out[:, word_num:, :]

            scale = feature_dim ** (-0.5)
            group_pred_scores: torch.Tensor = torch.bmm(CLS_output, law_output.transpose(1, 2)) * scale
            group_pred_scores = group_pred_scores.squeeze(dim=1)
            # group_pred_scores = torch.cosine_similarity(fact_output, law_output, dim=-1) * 10.0 # batch_size, node_num
            group_pred = torch.softmax(group_pred_scores, dim=-1)
            re_context = group_pred @ context_list
        else:
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
        # 'EncodingLaw'
        distilled_law_posterior, aggregate_law_posterior = law_inputs_posterior, law_inputs_posterior
        for i in range(self.num_distill_layers):
            distilled_law_posterior, aggregate_law_posterior = \
                self.graph_distillers_posterior[i](features=distilled_law_posterior, 
                                                   key_features=aggregate_law_posterior, 
                                                   adj_matrix=adj_matrix_posterior)
    
        if warming_up:
            distilled_law_posterior *= self.posterior_mask
            aggregate_law_posterior *= self.posterior_mask
            distilled_law_posterior = distilled_law_posterior.detach()
            aggregate_law_posterior = aggregate_law_posterior.detach()

        context_list_posterior = distilled_law_posterior
        
        # EncodeFact
            # BaseEncodingFact' # [batch_size, sentence_num, sentence_len] --> [batch_size, sentence_num, feature_dim]
        word_embedding_fact = self.PLM_model(fact_inputs_ids, fact_attention_mask, fact_token_type_ids)[0]
        # [batch_size, sentence_num, feature_dim]: get the word embeddings from the BERT model.
        fact_word_level = self.sentence_encoder(word_embedding_fact, sentence_mask)
        fact_base, score_s_base = self.sentence_attention(fact_word_level, masks=real_mask_fact)
                
            # PriorEncodingFact
        fact_prior, fact_prior_word, group_pred_prior, score_s_prior = \
            self.re_encoding_fact(inputs=[fact_base, None, context_list_prior, word_embedding_fact, None],
                                    dense_funcs=[None, None],
                                    context_funcs=[self.group_chosen, None, self.context_s_prior],
                                    encoder_funcs=[self.sentence_encoder_prior, self.sentence_attention_prior],
                                    masks=[sentence_mask, None, real_mask_fact], name='Prior')
            
            # PosteriorEncodingFact_Law
        fact_sentence = torch.cat([fact_word_level, fact_prior_word], axis=-1)
        new_sentence_mask = sentence_mask
        fact_posterior, fact_posterior_word, group_pred_posterior, score_s_posterior = \
            self.re_encoding_fact(inputs=[None, law_inputs_posterior, context_list_posterior, word_embedding_fact, fact_sentence],
                                    dense_funcs=[self.matching_law_posterior, self.matching_fact_posterior],
                                    context_funcs=[None, self.Transformer_posterior, self.context_s_posterior],
                                    encoder_funcs=[self.sentence_encoder_posterior, self.sentence_attention_posterior],
                                    masks=[sentence_mask, new_sentence_mask, real_mask_fact], name='Posterior')
                
        if warming_up:
            fact_posterior: torch.Tensor = fact_posterior * self.posterior_maskF
            fact_posterior = fact_posterior.detach()
        
        if self.accu_relation is not None:
            accu_inputs_posterior, accu_adj_matrix_posterior = accu_information
            # GeneratePosteriorGroupInformation_A
            distilled_accu_posterior, aggregate_accu_posterior = accu_inputs_posterior, accu_inputs_posterior
            for i in range(self.num_distill_layers):
                distilled_accu_posterior, aggregate_accu_posterior = \
                    self.graph_distillers_posterior_accu[i](features=distilled_accu_posterior, 
                                                            key_features=aggregate_accu_posterior, 
                                                            adj_matrix=accu_adj_matrix_posterior)
            if warming_up:
                distilled_accu_posterior: torch.Tensor = distilled_accu_posterior * self.posterior_mask
                aggregate_accu_posterior: torch.Tensor = aggregate_accu_posterior * self.posterior_mask
                distilled_accu_posterior = distilled_accu_posterior.detach()
                aggregate_accu_posterior = aggregate_accu_posterior.detach()
            
            context_list_posterior_A = self.dropout(distilled_accu_posterior)
            
            # 'PosteriorEncodingFact_Accu'
            fact_sentence = torch.cat([fact_word_level, fact_prior_word], axis=-1)
            new_sentence_mask = sentence_mask
            fact_posterior_A, fact_posterior_sentence_A, group_pred_posterior_A,  score_s_posterior_A = \
                self.re_encoding_fact(inputs=[None, accu_inputs_posterior, context_list_posterior_A, word_embedding_fact, fact_sentence],
                                        dense_funcs=[self.matching_law_posterior_A, self.matching_fact_posterior_A],
                                        context_funcs=[None, self.Transformer_posterior_A, self.context_s_posterior_A],
                                        encoder_funcs=[self.sentence_encoder_posterior_A, self.sentence_attention_posterior_A],
                                        masks=[sentence_mask, new_sentence_mask, real_mask_fact], name='Posterior')
            if warming_up:
                fact_posterior_A: torch.Tensor = fact_posterior_A * self.posterior_maskF
                fact_posterior_A = fact_posterior_A.detach()
                group_pred_posterior = group_pred_posterior.detach()
                group_pred_posterior_A = group_pred_posterior_A.detach()
            
            fact_rep = torch.cat([fact_base, fact_prior, fact_posterior, fact_posterior_A], axis=-1)
            law_rep = law_base
            
            return [fact_rep, law_rep, group_pred_prior, group_pred_posterior, group_pred_posterior_A,
                    score_s_base, score_s_prior, score_s_posterior, score_s_posterior_A]
            
        fact_rep = torch.cat([fact_base, fact_prior, fact_posterior], axis=-1)
        law_rep = law_base
        return [fact_rep, law_rep, group_pred_prior, group_pred_posterior,
                score_s_base, score_s_prior, score_s_posterior]
    

class dLadan_criminal(nn.Module):
    def __init__(self, config, group_num, accu_relation=1, **kwargs):
        super(dLadan_criminal, self).__init__(**kwargs)
        
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
        
        # 'define_distiller_posterior'
        self.graph_distillers_posterior = []
        self.hidden_size = config.net.hidden_size
        graph_input_posterior = config.net.hidden_size
        for i in range(self.num_distill_layers):
            distill_layer = GraphDistillOperatorWithEdgeWeight(config, input_dim=graph_input_posterior)  # with edge weight
            self.graph_distillers_posterior.append(distill_layer)
            graph_input_posterior = distill_layer.out_dim
        self.graph_distillers_posterior = nn.ModuleList(self.graph_distillers_posterior)
                
        # 'context_generator_prior'
        self.group_chosen_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.group_chosen = nn.Linear(self.hidden_size, self.group_num)
        self.context_s_prior = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # 'define_encoder_prior'
        self.sentence_encoder_prior = BertLayer(m=True)
        self.sentence_attention_prior = ContextAttention(config=config, input_dim=config.net.bert_size)

        # define_encoder_posterior'
        self.sentence_encoder_posterior = BertLayer(m=True)
        self.sentence_attention_posterior = ContextAttention(config=config, input_dim=config.net.bert_size)
            
        # context_generator_posterior
        self.Transformer_posterior = TransformerFeatureWithLabel(feature_dim=self.hidden_size,
                                                                 nhead=4, dropout=config.GraphDistill.dropout)
        self.matching_law_posterior = nn.Linear(self.hidden_size, self.hidden_size)
        self.matching_fact_posterior = nn.Linear(config.net.bert_size * 2, self.hidden_size)
        self.context_s_posterior = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout=nn.Dropout(config.GraphDistill.dropout)

        self.posterior_mask = nn.Parameter(torch.zeros([1, self.hidden_size]), requires_grad=False)
        self.posterior_maskF = nn.Parameter(torch.zeros([1, self.hidden_size]), requires_grad=False)
    
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
        
        if name == "Posterior":
            batch_size, word_num, feature_dim = fact_sentence_level.shape
            key_num, feature_dim = key_list.shape
            matching_law_prior:torch.Tensor = law_Dense(key_list)    # part of source input
            matching_fact_prior = fact_Dense(fact_sentence_level)
            label_mask = torch.ones([batch_size, key_num], dtype=torch.float).to(sentence_mask_1.device)
            matching_mask = torch.cat([sentence_mask_1, label_mask], axis=-1)
            matching_mask = torch.reshape(matching_mask, shape=(-1, word_num + key_num))
            
            label_input = matching_law_prior.unsqueeze(dim=0).repeat(batch_size, 1, 1)
            
            Transformer_input = torch.cat([matching_fact_prior, label_input], dim=1)
            # CLS_output, label_output, fact_output = Transformer(CLS_input, label_input, matching_fact_prior, mask=matching_mask)
            Transformer_out = Transformer(Transformer_input, mask=matching_mask)
            
            CLS_output = Transformer_out[:, 0, :].unsqueeze(dim=1)
            fact_output = Transformer_out[:, 1:word_num, :]
            law_output: torch.Tensor = Transformer_out[:, word_num:, :]

            scale = feature_dim ** (-0.5)
            group_pred_scores: torch.Tensor = torch.bmm(CLS_output, law_output.transpose(1, 2)) * scale
            group_pred_scores = group_pred_scores.squeeze(dim=1)
            # group_pred_scores = torch.cosine_similarity(fact_output, law_output, dim=-1) * 10.0 # batch_size, node_num
            group_pred = torch.softmax(group_pred_scores, dim=-1)
            re_context = group_pred @ context_list
        else:
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
        
        # 'EncodingLaw'
        distilled_law_posterior, aggregate_law_posterior = accu_inputs_posterior, accu_inputs_posterior
        for i in range(self.num_distill_layers):
            distilled_law_posterior, aggregate_law_posterior = \
                self.graph_distillers_posterior[i](features=distilled_law_posterior, 
                                                   key_features=aggregate_law_posterior, 
                                                   adj_matrix=adj_matrix_posterior)
    
        if warming_up:
            distilled_law_posterior *= self.posterior_mask
            aggregate_law_posterior *= self.posterior_mask
            distilled_law_posterior = distilled_law_posterior.detach()
            aggregate_law_posterior = aggregate_law_posterior.detach()

        context_list_posterior = distilled_law_posterior
        
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
            
            # PosteriorEncodingFact_Law
        fact_sentence = torch.cat([fact_word_level, fact_prior_word], axis=-1)
        new_sentence_mask = sentence_mask
        fact_posterior, fact_posterior_word, group_pred_posterior, score_s_posterior = \
            self.re_encoding_fact(inputs=[None, accu_inputs_posterior, context_list_posterior, word_embedding_fact, fact_sentence],
                                    dense_funcs=[self.matching_law_posterior, self.matching_fact_posterior],
                                    context_funcs=[None, self.Transformer_posterior, self.context_s_posterior],
                                    encoder_funcs=[self.sentence_encoder_posterior, self.sentence_attention_posterior],
                                    masks=[sentence_mask, new_sentence_mask, real_mask_fact], name='Posterior')
                
        if warming_up:
            fact_posterior: torch.Tensor = fact_posterior * self.posterior_maskF
            fact_posterior = fact_posterior.detach()
            
        fact_rep = torch.cat([fact_base, fact_prior, fact_posterior], axis=-1)
        law_rep = law_base
        return [fact_rep, law_rep, group_pred_prior, group_pred_posterior,
                score_s_base, score_s_prior, score_s_posterior]
