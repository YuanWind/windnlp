import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from modules.nn.base_model import BaseModel, get_parameter_names
from scripts.utils import get_tensor_device

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 3, self.dim_e)
        # self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 1, self.dim_e)

    def forward(self, edge, node1, node2):
        batch, seq, seq, edge_dim = edge.shape
        node = torch.cat([node1, node2], dim=-1)

        edge_diag = torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
        edge_i = edge_diag.unsqueeze(1).expand(batch, seq, seq, edge_dim)
        edge_j = edge_i.permute(0, 2, 1, 3).contiguous()
        edge = self.W(torch.cat([edge, edge_i, edge_j, node], dim=-1))

        # edge = self.W(torch.cat([edge, node], dim=-1))

        return edge


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = RefiningStrategy(gcn_dim, self.edge_dim, self.dep_embed_dim, dropout_ratio=0.5)

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)
    
        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim)

        weight_prob_softmax += self_loop
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs)
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)

        return node_outputs, edge_outputs


class Biaffine(nn.Module):
    def __init__(self, args, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.args = args
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                    out_features=self.linear_output_size,
                                    bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.args.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.args.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine


class EMCGCN(torch.nn.Module):
    def __init__(self, args):
        super(EMCGCN, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        self.dropout_output = torch.nn.Dropout(args.emb_dropout)

        self.post_emb = torch.nn.Embedding(args.post_size, args.class_num, padding_idx=0)
        self.deprel_emb = torch.nn.Embedding(args.deprel_size, args.class_num, padding_idx=0)
        self.postag_emb  = torch.nn.Embedding(args.postag_size, args.class_num, padding_idx=0)
        self.synpost_emb = torch.nn.Embedding(args.synpost_size, args.class_num, padding_idx=0)
        
        self.triplet_biaffine = Biaffine(args, args.gcn_dim, args.gcn_dim, args.class_num, bias=(True, True))
        self.ap_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim)
        self.op_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim)

        self.dense = nn.Linear(args.bert_feature_dim, args.gcn_dim)
        self.num_layers = args.num_layers
        self.gcn_layers = nn.ModuleList()

        self.layernorm = LayerNorm(args.bert_feature_dim)

        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(args.device, args.gcn_dim, 5*args.class_num, args.class_num, args.pooling))

    def forward(self, tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost):
        features = self.bert(tokens, masks)
        bert_feature = features[0]
        bert_feature = self.dropout_output(bert_feature) 

        batch, seq = masks.shape
        tensor_masks = masks.unsqueeze(1).expand(batch, seq, seq).unsqueeze(-1)
        
        # * multi-feature
        word_pair_post_emb = self.post_emb(word_pair_position)
        word_pair_deprel_emb = self.deprel_emb(word_pair_deprel)
        word_pair_postag_emb = self.postag_emb(word_pair_pos)
        word_pair_synpost_emb = self.synpost_emb(word_pair_synpost)
        
        # BiAffine
        ap_node = F.relu(self.ap_fc(bert_feature))
        op_node = F.relu(self.op_fc(bert_feature))
        biaffine_edge = self.triplet_biaffine(ap_node, op_node)
        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        weight_prob_list = [biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, word_pair_postag_emb, word_pair_synpost_emb]
        
        biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1) * tensor_masks
        word_pair_post_emb_softmax = F.softmax(word_pair_post_emb, dim=-1) * tensor_masks
        word_pair_deprel_emb_softmax = F.softmax(word_pair_deprel_emb, dim=-1) * tensor_masks
        word_pair_postag_emb_softmax = F.softmax(word_pair_postag_emb, dim=-1) * tensor_masks
        word_pair_synpost_emb_softmax = F.softmax(word_pair_synpost_emb, dim=-1) * tensor_masks

        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        self_loop = torch.stack(self_loop).to(self.args.device).unsqueeze(1).expand(batch, 5*self.args.class_num, seq, seq) * tensor_masks.permute(0, 3, 1, 2).contiguous()
        
        weight_prob = torch.cat([biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, \
            word_pair_postag_emb, word_pair_synpost_emb], dim=-1)
        weight_prob_softmax = torch.cat([biaffine_edge_softmax, word_pair_post_emb_softmax, \
            word_pair_deprel_emb_softmax, word_pair_postag_emb_softmax, word_pair_synpost_emb_softmax], dim=-1)

        for _layer in range(self.num_layers):
            gcn_outputs, weight_prob = self.gcn_layers[_layer](weight_prob_softmax, weight_prob, gcn_outputs, self_loop)  # [batch, seq, dim]
            weight_prob_list.append(weight_prob)

        return weight_prob_list

class Extractor(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = EMCGCN(config)
        
    def forward(self, **kwargs):
        tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost = \
            kwargs.pop('tokens'), kwargs.pop('masks'), kwargs.pop('word_pair_position'), \
            kwargs.pop('word_pair_deprel'), kwargs.pop('word_pair_pos'), kwargs.pop('word_pair_synpost')
        tags, tags_symmetry = kwargs.pop('tags'), kwargs.pop('tags_symmetry')
        tags_flatten = tags.reshape([-1])
        tags_symmetry_flatten = tags_symmetry.reshape([-1])
        # label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],dtype=torch.float, device=get_tensor_device(tokens))
        weight_prob_list = self.model(tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost)
        if self.config.relation_constraint:
            predictions = weight_prob_list
            biaffine_pred, post_pred, deprel_pred, postag, synpost, final_pred = predictions[0], predictions[1], predictions[2], predictions[3], predictions[4], predictions[5]
            l_ba = 0.10 * F.cross_entropy(biaffine_pred.reshape([-1, biaffine_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
            l_rpd = 0.01 * F.cross_entropy(post_pred.reshape([-1, post_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
            l_dep = 0.01 * F.cross_entropy(deprel_pred.reshape([-1, deprel_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
            l_psc = 0.01 * F.cross_entropy(postag.reshape([-1, postag.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
            l_tbd = 0.01 * F.cross_entropy(synpost.reshape([-1, synpost.shape[3]]), tags_symmetry_flatten, ignore_index=-1)

            if self.config.symmetry_decoding:
                l_p = F.cross_entropy(final_pred.reshape([-1, final_pred.shape[3]]), tags_symmetry_flatten, weight=weight, ignore_index=-1)
            else:
                l_p = F.cross_entropy(final_pred.reshape([-1, final_pred.shape[3]]), tags_flatten, weight=weight, ignore_index=-1)

            loss = l_ba + l_rpd + l_dep + l_psc + l_tbd + l_p
        else:
            preds = weight_prob_list[-1]
            preds_flatten = preds.reshape([-1, preds.shape[3]])
            if self.config.symmetry_decoding:
                loss = F.cross_entropy(preds_flatten, tags_symmetry_flatten, weight=weight, ignore_index=-1)
            else:
                loss = F.cross_entropy(preds_flatten, tags_flatten, weight=weight, ignore_index=-1)

        return {'loss': loss, 'logits':weight_prob_list[-1]}
    
    
    def optimizer_grouped_parameters(self, weight_decay):
        """为模型参数设置不同的优化器超参，比如分层学习率等，此处默认为hf的初始化方式
        """
        total_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in total_parameters if "bias" not in name]
        an_lr_keywords = []
        another_lr_parameters = []
        for name in total_parameters:
            if 'bert' not in name:
                another_lr_parameters.append(name)
            for keyw in an_lr_keywords:
                if keyw in name:
                    another_lr_parameters.append(name)

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters and n in another_lr_parameters],
                "weight_decay": weight_decay,
                "learing_rate": self.config.non_pretrain_lr
            },
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters and n not in another_lr_parameters],
                "weight_decay": weight_decay 
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters and n in another_lr_parameters],
                "weight_decay": 0.0,
                "learing_rate": self.config.non_pretrain_lr
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters and n not in another_lr_parameters],
                "weight_decay": 0.0
            }
        ]
        return optimizer_grouped_parameters