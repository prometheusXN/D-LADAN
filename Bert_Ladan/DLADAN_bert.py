import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch, ipdb
from Dladan_component import dLadan, dLadan_full, dLadan_criminal
from GraphRecorder import MemoryMomentum
from CosineClassifier import CosClassifier
from transformers.tokenization_utils_base import BatchEncoding

class DLADAN_Bert_full(nn.Module):
    def __init__(self, config, group_num, law_input:BatchEncoding, law_adj_matrix:torch.Tensor, group_indexes:torch.Tensor, 
                 accu_relation=None, decoder_mode='MTL', keep_coefficient=0.9, mode='small', **kwargs):
        super(DLADAN_Bert_full, self).__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.net.hidden_size
        if mode == "small":
            self.law_num = config.num_class_small.law_num
            self.accu_num = config.num_class_small.accu_num
            self.time_num = config.num_class_small.time_num
        else:
            self.law_num = config.num_class_large.law_num
            self.accu_num = config.num_class_large.accu_num
            self.time_num = config.num_class_large.time_num
            
        print("law_num:", self.law_num, ", charge_num:", self.accu_num, ", time_num:", self.time_num)
        
        self.warming_up = False
        self.momentum_flag = False
        self.synchronize_memory = False
        self.keep_coefficient = keep_coefficient
        
        self.group_num = group_num
        self.decoder_mode = decoder_mode
        self.fact_sentence_len = config.train.fact_sentence_len
        self.fact_sentence_num = config.train.fact_sentence_num
        
        law_input_ids, law_attention_mask, law_token_type_ids = \
            law_input['input_ids'], law_input['attention_mask'], law_input['token_type_ids']
        self.register_buffer('law_input_ids', torch.LongTensor(law_input_ids))
        self.register_buffer('law_attention_mask', torch.LongTensor(law_attention_mask))
        self.register_buffer('law_token_type_ids', torch.LongTensor(law_token_type_ids))
        self.register_buffer('law_adj_matrix', torch.LongTensor(law_adj_matrix))
        self.register_buffer('group_indexes', torch.LongTensor(group_indexes))
        print('law_information loaded')
        
        self.accu_relation = accu_relation
        
        # 'define_FeatureExtracter'
        self.LadanPPK = dLadan_full(config=config, group_num=group_num, accu_relation=accu_relation)
        
        self.Law_memory = MemoryMomentum(self.law_num, self.hidden_size)
        
        self.Accu_memory = MemoryMomentum(self.accu_num, self.hidden_size)

        self.Law_decoder_m = nn.Linear(self.hidden_size *4, self.hidden_size)
        self.Accu_decoder_m = nn.Linear(self.hidden_size *4, self.hidden_size)
        self.Time_decoder_m = nn.Linear(self.hidden_size *4, self.hidden_size)

        # self.Law_decoder_mm = nn.Linear(self.hidden_size *2, self.hidden_size)
        # self.Accu_decoder_mm = nn.Linear(self.hidden_size *2, self.hidden_size)
        # self.Time_decoder_mm = nn.Linear(self.hidden_size *2, self.hidden_size)
            
        # 'define_Decoder'
        self.Law_decoder = CosClassifier(self.law_num, self.hidden_size, with_proto=False)
        self.Accu_decoder = CosClassifier(self.accu_num, self.hidden_size, with_proto=False)
        self.Time_decoder = CosClassifier(self.time_num, self.hidden_size, with_proto=False)

        self.Law_de = nn.Sequential(self.Law_decoder_m, self.Law_decoder)
        self.Accu_de = nn.Sequential(self.Accu_decoder_m, self.Accu_decoder)
        self.Time_de = nn.Sequential(self.Law_decoder_m, self.Time_decoder)

        # self.Law_de = nn.Sequential(self.Law_decoder_m, nn.ReLU(), 
        #                             self.Law_decoder_mm, self.Law_decoder)
        # self.Accu_de = nn.Sequential(self.Accu_decoder_m, nn.ReLU(), 
        #                             self.Accu_decoder_mm, self.Accu_decoder)
        # self.Time_de = nn.Sequential(self.Law_decoder_m, nn.ReLU(), 
        #                             self.Law_decoder_mm, self.Time_decoder)
        
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def init_multi_gpu(self, device):
        self.LadanPPK.PLM_model = nn.DataParallel(self.LadanPPK.PLM_model, device_ids=device)
        self.LadanPPK.sentence_encoder = nn.DataParallel(self.LadanPPK.sentence_encoder, device_ids=device)
        self.LadanPPK.sentence_encoder_prior = nn.DataParallel(self.LadanPPK.sentence_encoder_prior, device_ids=device)
        self.LadanPPK.sentence_encoder_posterior = nn.DataParallel(self.LadanPPK.sentence_encoder_posterior, device_ids=device)
        self.LadanPPK.sentence_encoder_posterior_A = nn.DataParallel(self.LadanPPK.sentence_encoder_posterior_A, device_ids=device)
        self.LadanPPK.Transformer_posterior = nn.DataParallel(self.LadanPPK.Transformer_posterior, device_ids=device)
        self.Law_de = nn.DataParallel(self.Law_de, device_ids=device)
        self.Accu_de = nn.DataParallel(self.Accu_de, device_ids=device)
        self.Time_de = nn.DataParallel(self.Time_de, device_ids=device)

        if self.accu_relation is not None:
            self.LadanPPK.Transformer_posterior_A = nn.DataParallel(self.LadanPPK.Transformer_posterior_A, device_ids=device)
    
    @staticmethod
    def matrix_computation(node_features: torch.Tensor):
        node_num, feature_dim = node_features.shape
        feature_1 = node_features.unsqueeze(dim=1)
        feature_2 = node_features.unsqueeze(dim=0)
        cosine_matrix = torch.cosine_similarity(feature_1, feature_2, dim=-1)
        adj_matrix = cosine_matrix - torch.eye(node_num, dtype=torch.float).to(cosine_matrix.device)
        return adj_matrix
        
    def forward(self, inputs, labels, model_type='train', synchronize_memory=False, warming_up=False, momentum_flag=False):
        fact_input_ids, fact_attention_mask, fact_token_type_ids = inputs
        # compute_mask
        
        law_memory = self.Law_memory(inputs=self.Law_decoder.proto, keep_coefficient=self.keep_coefficient, 
                                     synchronize_memory=synchronize_memory, momentum_flag=momentum_flag,
                                     warming_up=warming_up)
        
        law_adj_matrix_posterior = self.matrix_computation(law_memory)
        
        if self.accu_relation is not None:
            accu_memory = self.Accu_memory(self.Accu_decoder.proto, keep_coefficient=self.keep_coefficient, 
                                           synchronize_memory=synchronize_memory, momentum_flag=momentum_flag, 
                                           warming_up=warming_up)
            
            accu_adj_matrix_posterior = self.matrix_computation(accu_memory)
            accu_information = [accu_memory, accu_adj_matrix_posterior]
        else:
            accu_information = None
            
        time_information = None
        
        # 'model_process' 
        
        if self.accu_relation is not None:
            law_labels, accu_labels, time_labels, group_labels = labels
            group_posterior_labels = law_labels
            group_posterior_accu_labels = accu_labels
            [fact_rep, law_rep, group_pred_prior, group_pred_posterior, group_pred_posterior_accu,
             score_s_base, score_s_prior, score_s_posterior, score_s_posterior_A] = \
                self.LadanPPK(inputs=[fact_input_ids, fact_token_type_ids],
                              law_information=[self.law_input_ids, self.law_token_type_ids, self.law_adj_matrix,
                                               self.group_indexes, law_memory, law_adj_matrix_posterior],
                              fact_attention_mask=fact_attention_mask, sentence_mask=fact_attention_mask,
                              time_information=time_information, accu_information=accu_information,
                              law_attention_mask=self.law_attention_mask, warming_up=warming_up)
            
            output_law = self.Law_de(fact_rep)
            output_accu = self.Accu_de(fact_rep)
            output_time = self.Time_de(fact_rep)

            # law_hidden = self.Law_decoder_m(fact_rep)
            # accu_hidden = self.Accu_decoder_m(fact_rep)
            # time_hidden = self.Time_decoder_m(fact_rep)
            
            # output_law, _ = self.Law_decoder(law_hidden)
            # output_accu, _ = self.Accu_decoder(accu_hidden)
            # output_time, _ = self.Time_decoder(time_hidden)

            if model_type=="train":
                law_pred_loss = self.loss(output_law, law_labels)
                charge_pred_loss = self.loss(output_accu, accu_labels)
                time_pred_loss = self.loss(output_time, time_labels)

                GroupSelection_pri_loss = self.loss(group_pred_prior, group_labels)
                GroupSelection_post_loss = self.loss(group_pred_posterior, group_posterior_labels)
                GroupSelection_post_loss_A = self.loss(group_pred_posterior_accu, group_posterior_accu_labels)
                if warming_up:
                    loss = law_pred_loss + charge_pred_loss + time_pred_loss + 0.1 * GroupSelection_pri_loss
                else:
                    loss = law_pred_loss + charge_pred_loss + time_pred_loss + \
                        0.1 * (GroupSelection_pri_loss + GroupSelection_post_loss + GroupSelection_post_loss_A)
                    # loss = law_pred_loss + charge_pred_loss + time_pred_loss
                return {"loss_law": law_pred_loss,
                        "loss_charge": charge_pred_loss,
                        "loss_time": time_pred_loss,
                        "loss_graph_pri": GroupSelection_pri_loss,
                        "loss_graph_post": GroupSelection_post_loss,
                        "loss_graph_post_a": GroupSelection_post_loss_A,
                        "loss": loss}
            
            else:
                law_pred = torch.softmax(output_law, dim=-1)
                charge_pred = torch.softmax(output_accu, dim=-1)
                time_pred = torch.softmax(output_time, dim=-1)
                group_pred_pri = torch.softmax(group_pred_prior, dim=-1)
                group_pred_post = torch.softmax(group_pred_posterior, dim=-1)
                group_pred_post_accu = torch.softmax(group_pred_posterior_accu, dim=-1)
                return {"law_pred": law_pred,
                        "charge_pred": charge_pred,
                        "time_pred": time_pred,
                        "group_pred_pri": group_pred_pri,
                        "group_pred_post": group_pred_post,
                        "group_pred_post_a": group_pred_post_accu}
            

class DLADAN_Bert_C(nn.Module):
    def __init__(self, config, group_num, law_input:BatchEncoding, law_adj_matrix:torch.Tensor, group_indexes:torch.Tensor, 
                 accu_relation=None, decoder_mode='MTL', keep_coefficient=0.9, mode='small', **kwargs):
        super(DLADAN_Bert_C, self).__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.net.hidden_size
        self.accu_num = config.num_class
            
        print("charge_num:", self.accu_num)
        
        self.warming_up = False
        self.momentum_flag = False
        self.synchronize_memory = False
        self.keep_coefficient = keep_coefficient
        
        self.group_num = group_num
        self.decoder_mode = decoder_mode
        self.fact_sentence_len = config.train.fact_sentence_len

        
        law_input_ids, law_attention_mask, law_token_type_ids = \
            law_input['input_ids'], law_input['attention_mask'], law_input['token_type_ids']
        self.register_buffer('law_input_ids', torch.LongTensor(law_input_ids))
        self.register_buffer('law_attention_mask', torch.LongTensor(law_attention_mask))
        self.register_buffer('law_token_type_ids', torch.LongTensor(law_token_type_ids))
        self.register_buffer('law_adj_matrix', torch.LongTensor(law_adj_matrix))
        self.register_buffer('group_indexes', torch.LongTensor(group_indexes))
        print('law_information loaded')
        
        self.accu_relation = accu_relation
        
        # 'define_FeatureExtracter'
        self.LadanPPK = dLadan_criminal(config=config, group_num=group_num, accu_relation=accu_relation) 
        self.Accu_memory = MemoryMomentum(self.accu_num, self.hidden_size)
        
        self.Accu_decoder_m = nn.Linear(self.hidden_size *3, self.hidden_size)
        self.Accu_decoder = CosClassifier(self.accu_num, self.hidden_size, with_proto=False)
        self.Accu_de = nn.Sequential(self.Accu_decoder_m, self.Accu_decoder)
        
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def init_multi_gpu(self, device):
        self.LadanPPK.PLM_model = nn.DataParallel(self.LadanPPK.PLM_model, device_ids=device)
        self.LadanPPK.sentence_encoder = nn.DataParallel(self.LadanPPK.sentence_encoder, device_ids=device)
        self.LadanPPK.sentence_encoder_prior = nn.DataParallel(self.LadanPPK.sentence_encoder_prior, device_ids=device)
        self.LadanPPK.sentence_encoder_posterior = nn.DataParallel(self.LadanPPK.sentence_encoder_posterior, device_ids=device)
        self.LadanPPK.Transformer_posterior = nn.DataParallel(self.LadanPPK.Transformer_posterior, device_ids=device)
        self.Accu_de = nn.DataParallel(self.Accu_de, device_ids=device)
    
    @staticmethod
    def matrix_computation(node_features: torch.Tensor):
        node_num, feature_dim = node_features.shape
        feature_1 = node_features.unsqueeze(dim=1)
        feature_2 = node_features.unsqueeze(dim=0)
        cosine_matrix = torch.cosine_similarity(feature_1, feature_2, dim=-1)
        adj_matrix = cosine_matrix - torch.eye(node_num, dtype=torch.float).to(cosine_matrix.device)
        return adj_matrix
        
    def forward(self, inputs, labels, model_type='train', synchronize_memory=False, warming_up=False, momentum_flag=False):
        fact_input_ids, fact_attention_mask, fact_token_type_ids = inputs
        # compute_mask
        

        accu_memory = self.Accu_memory(self.Accu_decoder.proto, keep_coefficient=self.keep_coefficient, 
                                        synchronize_memory=synchronize_memory, momentum_flag=momentum_flag, 
                                        warming_up=warming_up)
        
        accu_adj_matrix_posterior = self.matrix_computation(accu_memory)
        accu_information = [accu_memory, accu_adj_matrix_posterior]
            
        
        
        accu_labels, group_labels = labels
        group_posterior_labels = accu_labels
        group_posterior_accu_labels = accu_labels
        [fact_rep, law_rep, group_pred_prior, group_pred_posterior,
            score_s_base, score_s_prior, score_s_posterior] = \
            self.LadanPPK(inputs=[fact_input_ids, fact_token_type_ids],
                            law_information=[self.law_input_ids, self.law_token_type_ids, self.law_adj_matrix,
                                            self.group_indexes, accu_memory, accu_adj_matrix_posterior],
                            fact_attention_mask=fact_attention_mask, sentence_mask=fact_attention_mask,
                            law_attention_mask=self.law_attention_mask, warming_up=warming_up)

        output_accu = self.Accu_de(fact_rep)

        if model_type=="train":
            charge_pred_loss = self.loss(output_accu, accu_labels)
            GroupSelection_pri_loss = self.loss(group_pred_prior, group_labels)
            GroupSelection_post_loss = self.loss(group_pred_posterior, group_posterior_labels)
            if warming_up:
                loss = charge_pred_loss + 0.1 * GroupSelection_pri_loss
            else:
                loss = charge_pred_loss + 0.1 * (GroupSelection_pri_loss + GroupSelection_post_loss)
            return {
                    "loss_charge": charge_pred_loss,
                    "loss_graph_pri": GroupSelection_pri_loss,
                    "loss_graph_post": GroupSelection_post_loss,
                    "loss": loss}
        
        else:
            charge_pred = torch.softmax(output_accu, dim=-1)
            group_pred_pri = torch.softmax(group_pred_prior, dim=-1)
            group_pred_post = torch.softmax(group_pred_posterior, dim=-1)

            return {
                    "charge_pred": charge_pred,
                    "group_pred_pri": group_pred_pri,
                    "group_pred_post": group_pred_post}
