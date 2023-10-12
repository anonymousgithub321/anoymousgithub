
import math, os, pickle, random, sys, argparse
from time import time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from function import setup_seed, csr_to_user_dict, pad_sequences
from evaluate import *

# 定义一个层复制函数，将每一层的结构执行深拷贝，并返回list形式
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 先对输入值x进行reshape一下，然后交换在维度1,2进行交换
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

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
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
 
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
 
    def __init__(self, d_model, d_ff=2048, d_output=768, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_output)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Model(nn.Module):
    def __init__(self, params, summary, review, critic_review):
        super(Model, self).__init__()
        self.params = params
        # self.pt_model = AutoModel.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = params.device
        self.n_users = params.n_users
        self.n_items = params.n_items
        self.n_critics = params.n_critics
        self.embedding_size = params.embedding_size

        # parameters of VAE
        self.anneal = 0.0
        self.dims = [self.n_items, 600, self.embedding_size]
        self.input_dropout = nn.Dropout(1-self.params.keep_prob)
        self.decoder = nn.Embedding(self.n_items, self.dims[-1])
        self.decoder_b = nn.Embedding(self.n_items, self.dims[-1])
        self.adaptfeat_u = nn.Embedding(self.n_users, self.dims[-1])
        # self.fc_adapt = nn.Linear(self.embedding_size*3, 1)
        self.fc_adapt = nn.Linear(self.embedding_size*3, self.embedding_size)
        self.fc_adapt2 = nn.Linear(self.embedding_size, 1)
        self.vae_weights = nn.ParameterDict()
        for k in range(1, len(self.dims)):
            if k != len(self.dims) - 1:
                self.vae_weights['W_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k-1], self.dims[k])).to(self.device)
                self.vae_weights['b_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k])).to(self.device)
            else:
                self.vae_weights['W_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k-1], 2*self.dims[k])).to(self.device)
                self.vae_weights['b_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(2*self.dims[k])).to(self.device)
            nn.init.xavier_normal_(self.vae_weights['W_encoder_%d' % k])
            nn.init.trunc_normal_(self.vae_weights['b_encoder_%d' % k], std=0.001, a=-0.002, b=0.002)
        self.vae_weights_w = nn.ParameterDict()
        for k in range(1, len(self.dims)):
            if k != len(self.dims) - 1:
                self.vae_weights_w['W_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k-1], self.dims[k])).to(self.device)
                self.vae_weights_w['b_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k])).to(self.device)
            else:
                self.vae_weights_w['W_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(self.dims[k-1], 2*self.dims[k])).to(self.device)
                self.vae_weights_w['b_encoder_%d' % k] = nn.Parameter(torch.FloatTensor(2*self.dims[k])).to(self.device)
            nn.init.xavier_normal_(self.vae_weights_w['W_encoder_%d' % k])
            nn.init.trunc_normal_(self.vae_weights_w['b_encoder_%d' % k], std=0.001, a=-0.002, b=0.002)

        # parameters of Text
        self.feat_summary = nn.Embedding(summary.shape[0], summary.shape[1])
        self.feat_summary.weight.data.copy_(summary)
        self.feat_summary.weight.requires_grad = False
        self.feat_review = nn.Embedding(review.shape[0], review.shape[1])
        self.feat_review.weight.data.copy_(review)
        self.feat_review.weight.requires_grad = False
        self.feat_critic_review = nn.Embedding(critic_review.shape[0], critic_review.shape[1])
        self.feat_critic_review.weight.data.copy_(critic_review)
        self.feat_critic_review.weight.requires_grad = False
        
        self.hidden_size = params.hidden_size
        self.attn_user_review = MultiHeadedAttention(8, self.hidden_size, dropout=self.params.drop)
        self.attn_user_review_w = MultiHeadedAttention(8, self.hidden_size, dropout=self.params.drop)
        self.attn_item_user_review = MultiHeadedAttention(8, self.hidden_size, dropout=self.params.drop)
        self.attn_item_critic_review = MultiHeadedAttention(8, self.hidden_size, dropout=self.params.drop)
        self.ff_user_review = PositionwiseFeedForward(self.hidden_size, d_ff=2048, d_output=self.embedding_size, dropout=self.params.drop)
        self.ff_user_review_w = PositionwiseFeedForward(self.hidden_size, d_ff=2048, d_output=self.embedding_size, dropout=self.params.drop)
        self.ff_item_user_review = PositionwiseFeedForward(self.hidden_size, d_ff=2048, d_output=self.embedding_size, dropout=self.params.drop)
        self.ff_item_critic_review = PositionwiseFeedForward(self.hidden_size, d_ff=2048, d_output=self.embedding_size, dropout=self.params.drop)
        
    def forward(self, input_ph, critic_input_ph, users, user_review_items, mask_ui, item_review_users, mask_iu, item_review_critics, mask_ic, is_train=1):
        # Multi-VAE model
        self.vae_i_emb = self.decoder(torch.arange(params.n_items, dtype=torch.long, device=self.device))
        self.vae_u_emb, KL_u = self.create_VAE_embed(input_ph, self.vae_weights)
                
        self.vae_b_emb = self.decoder_b(torch.arange(params.n_items, dtype=torch.long, device=self.device))
        self.vae_w_emb, KL_w = self.create_VAE_embed(input_ph, self.vae_weights_w)
        
        # Text model
        self.summary_emb = self.feat_summary(torch.arange(params.n_items, dtype=torch.long, device=self.device))
        user_review_emb = self.feat_review(user_review_items)
        item_user_review_emb = self.feat_review(item_review_users)
        item_critic_review_emb = self.feat_critic_review(item_review_critics)
        
        item_user_review_emb = torch.cat((self.summary_emb.unsqueeze(1), item_user_review_emb), dim=1)
        item_critic_review_emb = torch.cat((self.summary_emb.unsqueeze(1), item_critic_review_emb), dim=1)
        
        attn_u_emb, attn_w_emb, attn_iu_emb, attn_ic_emb = user_review_emb, user_review_emb, item_user_review_emb, item_critic_review_emb
        attn_u_emb = self.attn_user_review(user_review_emb, user_review_emb, user_review_emb, mask_ui.unsqueeze(-2).expand(mask_ui.size(0), mask_ui.size(1), mask_ui.size(1)))
        attn_u_emb = self.ff_user_review(attn_u_emb)
        attn_u_emb = attn_u_emb * mask_ui.unsqueeze(-1)
        self.text_u_emb = torch.sum(attn_u_emb, dim=1)/torch.sum(mask_ui, dim=1, keepdim=True)
        attn_w_emb = self.attn_user_review_w(user_review_emb, user_review_emb, user_review_emb, mask_ui.unsqueeze(-2).expand(mask_ui.size(0), mask_ui.size(1), mask_ui.size(1)))
        attn_w_emb = self.ff_user_review_w(attn_w_emb)
        attn_w_emb = attn_w_emb * mask_ui.unsqueeze(-1)
        self.text_w_emb = torch.sum(attn_w_emb, dim=1)/torch.sum(mask_ui, dim=1, keepdim=True)
        attn_iu_emb = self.attn_item_user_review(item_user_review_emb, item_user_review_emb, item_user_review_emb, mask_iu.unsqueeze(-2).expand(mask_iu.size(0), mask_iu.size(1), mask_iu.size(1)))
        attn_iu_emb = self.ff_item_user_review(attn_iu_emb)
        attn_iu_emb = attn_iu_emb * mask_iu.unsqueeze(-1)
        self.text_iu_emb = torch.sum(attn_iu_emb, dim=1)/torch.sum(mask_iu, dim=1, keepdim=True)
        attn_ic_emb = self.attn_item_critic_review(item_critic_review_emb, item_critic_review_emb, item_critic_review_emb, mask_ic.unsqueeze(-2).expand(mask_ic.size(0), mask_ic.size(1), mask_ic.size(1)))
        attn_ic_emb = self.ff_item_critic_review(attn_ic_emb)
        attn_ic_emb = attn_ic_emb * mask_ic.unsqueeze(-1)
        self.text_ic_emb = torch.sum(attn_ic_emb, dim=1)/torch.sum(mask_ic, dim=1, keepdim=True)
        
        #生成最终的用户和物品的embedding
        # self.u_emb = (self.vae_u_emb + self.text_u_emb)/2
        # self.w_emb = (self.vae_w_emb + self.text_w_emb)/2
        # self.i_emb = (self.vae_i_emb + self.text_ic_emb)/2
        # self.b_emb = (self.vae_b_emb + self.text_iu_emb)/2
        self.u_emb = torch.cat((self.vae_u_emb, self.text_u_emb), dim=-1)
        self.w_emb = torch.cat((self.vae_w_emb, self.text_w_emb), dim=-1)
        self.i_emb = torch.cat((self.vae_i_emb, self.text_ic_emb), dim=-1)
        self.b_emb = torch.cat((self.vae_b_emb, self.text_iu_emb), dim=-1)
        
        # 计算预测分数
        logits_u = torch.mm(self.u_emb, ((self.i_emb + self.b_emb)/2).t())
        logits_w = torch.mm(self.w_emb, self.b_emb.t())

        adapt_u = self.adaptfeat_u(users)
        adapt_u = torch.cat((adapt_u, self.text_u_emb, self.text_w_emb), dim=-1)
        # lambda_adapt = F.sigmoid(self.fc_adapt(adapt_u))
        lambda_adapt = F.sigmoid(self.fc_adapt2(F.relu(self.fc_adapt(adapt_u))))
        logits = ((1 - lambda_adapt) * logits_u + lambda_adapt * logits_w)
        
        self.vae_c_emb, KL_c = self.create_VAE_embed(critic_input_ph, self.vae_weights)
        logits_critic = torch.mm(self.vae_c_emb, self.vae_i_emb.t())
        log_softmax_var_critic = F.log_softmax(logits_critic)
                 
        # 计算预测Loss
        log_softmax_var = F.log_softmax(logits)
        log_softmax_var_w = F.log_softmax(logits_w)
        self.rating_loss = - torch.mean(torch.sum(log_softmax_var * input_ph, 1))
        self.rating_w_loss = - torch.mean(torch.sum(log_softmax_var_w * input_ph, 1))
        self.rating_c_loss = - torch.mean(torch.sum(log_softmax_var_critic * critic_input_ph, 1))
        
        self.loss = self.rating_loss + self.anneal * KL_u
        self.loss = self.loss + self.params.eta * (self.rating_w_loss + self.anneal * KL_w)
        self.loss = self.loss + self.params.eta2 * (self.anneal * KL_c + self.rating_c_loss)
        self.loss = self.loss + self.params.reg * (torch.norm(self.i_emb, dim=-1).mean() + torch.norm(self.b_emb, dim=-1).mean())
       
        return self.loss, self.rating_loss, KL_u, self.rating_w_loss, KL_w
    
    def get_prediction(self, input_ph, users, user_review_items, mask_ui, item_review_users, mask_iu, item_review_critics, mask_ic):
        vae_u_emb, _ = self.create_VAE_embed(input_ph, self.vae_weights)
        vae_w_emb, _ = self.create_VAE_embed(input_ph, self.vae_weights_w)
        
        user_review_emb = self.feat_review(user_review_items)
        item_user_review_emb = self.feat_review(item_review_users)
        item_critic_review_emb = self.feat_critic_review(item_review_critics)
        
        item_user_review_emb = torch.cat((self.summary_emb.unsqueeze(1), item_user_review_emb), dim=1)
        item_critic_review_emb = torch.cat((self.summary_emb.unsqueeze(1), item_critic_review_emb), dim=1)
        
        attn_u_emb = self.attn_user_review(user_review_emb, user_review_emb, user_review_emb, mask_ui.unsqueeze(-2).expand(mask_ui.size(0), mask_ui.size(1), mask_ui.size(1)))
        attn_u_emb = self.ff_user_review(attn_u_emb)
        attn_u_emb = attn_u_emb * mask_ui.unsqueeze(-1)
        text_u_emb = torch.sum(attn_u_emb, dim=1)/torch.sum(mask_ui, dim=1, keepdim=True)
        attn_w_emb = self.attn_user_review_w(user_review_emb, user_review_emb, user_review_emb, mask_ui.unsqueeze(-2).expand(mask_ui.size(0), mask_ui.size(1), mask_ui.size(1)))
        attn_w_emb = self.ff_user_review_w(attn_w_emb)
        attn_w_emb = attn_w_emb * mask_ui.unsqueeze(-1)
        text_w_emb = torch.sum(attn_w_emb, dim=1)/torch.sum(mask_ui, dim=1, keepdim=True)
        attn_iu_emb = self.attn_item_user_review(item_user_review_emb, item_user_review_emb, item_user_review_emb, mask_iu.unsqueeze(-2).expand(mask_iu.size(0), mask_iu.size(1), mask_iu.size(1)))
        attn_iu_emb = self.ff_item_user_review(attn_iu_emb)
        attn_iu_emb = attn_iu_emb * mask_iu.unsqueeze(-1)
        text_iu_emb = torch.sum(attn_iu_emb, dim=1)/torch.sum(mask_iu, dim=1, keepdim=True)
        attn_ic_emb = self.attn_item_critic_review(item_critic_review_emb, item_critic_review_emb, item_critic_review_emb, mask_ic.unsqueeze(-2).expand(mask_ic.size(0), mask_ic.size(1), mask_ic.size(1)))
        attn_ic_emb = self.ff_item_critic_review(attn_ic_emb)
        attn_ic_emb = attn_ic_emb * mask_ic.unsqueeze(-1)
        text_ic_emb = torch.sum(attn_ic_emb, dim=1)/torch.sum(mask_ic, dim=1, keepdim=True)
        
        # u_emb = (vae_u_emb + text_u_emb)/2
        # w_emb = (vae_w_emb + text_w_emb)/2
        # i_emb = (self.vae_i_emb + text_ic_emb)/2
        # b_emb = (self.vae_b_emb + text_iu_emb)/2
        u_emb = torch.cat((vae_u_emb, text_u_emb), dim=-1)
        w_emb = torch.cat((vae_w_emb, text_w_emb), dim=-1)
        i_emb = torch.cat((self.vae_i_emb, text_ic_emb), dim=-1)
        b_emb = torch.cat((self.vae_b_emb, text_iu_emb), dim=-1)
        
        logits_u = torch.mm(u_emb, ((i_emb + b_emb)/2).t())
        logits_w = torch.mm(w_emb, b_emb.t())
        
        adapt_u = self.adaptfeat_u(users)
        adapt_u = torch.cat((adapt_u, text_u_emb, text_w_emb), dim=-1)
        # lambda_adapt = F.sigmoid(self.fc_adapt(adapt_u))
        lambda_adapt = F.sigmoid(self.fc_adapt2(F.relu(self.fc_adapt(adapt_u))))
        logits = ((1 - lambda_adapt) * logits_u + lambda_adapt * logits_w)
        
        return logits.cpu().numpy()
    
    def create_VAE_embed(self, input_ph, vae_weights):
        h = F.normalize(input_ph, p=2, dim=1)
        h = self.input_dropout(h)
        for k in range(1, len(self.dims)):
            h = torch.mm(h, vae_weights['W_encoder_%d' % k]) + vae_weights['b_encoder_%d' % k]
            if k != len(self.dims) - 1:
                h = F.tanh(h)
            else:
                mu_q = h[:, :self.dims[-1]]
                logvar_q = h[:, self.dims[-1]:]
                std_q = torch.exp(0.5 * logvar_q)
                KL_loss = -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=1).mean()
        
        epsilon = torch.randn_like(std_q)
        users = mu_q + self.training * epsilon * std_q
        return users, KL_loss


parser = argparse.ArgumentParser()
parser.add_argument('-dev', action='store', dest='dev', default='2')
parser.add_argument('-print', action='store', dest='print', default=10, type=int)
parser.add_argument('-keep_prob', type=float, default=0.5)
parser.add_argument('-drop', type=float, default=0.1)
parser.add_argument('-reg', type=float, default=1.0)
parser.add_argument('-batch_size', type=int, default=2048, help='input batch size')
parser.add_argument('-batch_size_valid', type=int, default=8196, help='valid batch size')
parser.add_argument('-hidden_size', type=int, default=384, help='hidden state size of transformers module')
parser.add_argument('-embed_dim', type=int, default=150, help='the dimension of embedding')
parser.add_argument('-epochs', type=int, default=5000, help='the number of epochs to train for')
parser.add_argument('-st_epoch', type=int, default=0, help='the start number of epochs to valid')
parser.add_argument('-load_epoch', type=int, default=1, help='the number of epoch to load')
parser.add_argument('-layers', type=int, default=3, help='the number of layers to train for')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-log', type=str, default='test')
parser.add_argument('-step', action='store', dest='step', default=20, type=int)
parser.add_argument('-eta', type=float, default=0.1)
parser.add_argument('-eta2', type=float, default=1.0)
parser.add_argument('-max_len', type=int, default=10)
args = parser.parse_args()


setup_seed(20)
class Params:
    def __init__(self):
        self.print_epoch = args.print
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embed_dim
        self.batch_size = args.batch_size
        self.keep_prob = args.keep_prob
        self.drop = args.drop
        self.reg = args.reg
        self.device = torch.device('cuda:' + args.dev if torch.cuda.is_available() else 'cpu')
        self.n_layers = args.layers
        self.eta = args.eta
        self.eta2 = args.eta2
        self.epochs = args.epochs
        self.model_name = "down-paraphrase-multilingual-MiniLM-L12-v2"
        self.n_users, self.n_items, self.n_critics = 0, 0, 0
        self.max_len = args.max_len

# 处理user数据
params = Params()
data_handle_dir = '../data/mc20231006/'
# train_list = pd.read_csv(data_handle_dir + "gametrain{}.csv".format(0), header=None, encoding='UTF-8')
# valid_list = pd.read_csv(data_handle_dir + "gameprobe{}.csv".format(0), header=None, encoding='UTF-8')
# critic_data = pd.read_csv(data_handle_dir + "game_critic_review.csv", header=0, encoding='UTF-8')
# summary = pd.read_csv(data_handle_dir + "pretrainemb_gameinfo.csv", header=None, encoding='UTF-8').values
# review = pd.read_csv(data_handle_dir + "pretrainemb_usergame.csv", header=None, encoding='UTF-8').values
# critic_review = pd.read_csv(data_handle_dir + "pretrainemb_criticgame.csv", header=None, encoding='UTF-8').values

train_list = pd.read_csv(data_handle_dir + "movietrain{}.csv".format(0), header=None, encoding='UTF-8')
valid_list = pd.read_csv(data_handle_dir + "movieprobe{}.csv".format(0), header=None, encoding='UTF-8')
critic_data = pd.read_csv(data_handle_dir + "movie_critic_review.csv", header=0, encoding='UTF-8')
summary = pd.read_csv(data_handle_dir + "pretrainemb_movieinfo.csv", header=None, encoding='UTF-8').values
review = pd.read_csv(data_handle_dir + "pretrainemb_usermovie.csv", header=None, encoding='UTF-8').values
critic_review = pd.read_csv(data_handle_dir + "pretrainemb_criticmovie.csv", header=None, encoding='UTF-8').values

# train_list = pd.read_csv(data_handle_dir + "musictrain{}.csv".format(0), header=None, encoding='UTF-8')
# valid_list = pd.read_csv(data_handle_dir + "musicprobe{}.csv".format(0), header=None, encoding='UTF-8')
# critic_data = pd.read_csv(data_handle_dir + "music_critic_review.csv", header=0, encoding='UTF-8')
# summary = pd.read_csv(data_handle_dir + "pretrainemb_musicinfo.csv", header=None, encoding='UTF-8').values
# review = pd.read_csv(data_handle_dir + "pretrainemb_usermusic.csv", header=None, encoding='UTF-8').values
# critic_review = pd.read_csv(data_handle_dir + "pretrainemb_criticmusic.csv", header=None, encoding='UTF-8').values

train_list = train_list[[0, 1, 4, 2]].values
valid_list = valid_list[[0, 1, 4, 2]].values
params.n_users, params.n_items = np.max(np.concatenate((train_list, valid_list))[:, 0:2], axis=0) + 1
valid_rating = (valid_list[:, 3].astype(np.float32))/np.max(valid_list[:, 3]).astype(np.float32)
valid_list = valid_list[valid_rating >= 0.7, :]
train_rating = (train_list[:, 3].astype(np.float32))/np.max(train_list[:, 3]).astype(np.float32)
train_matrix = sp.csr_matrix((train_rating, (train_list[:, 0], train_list[:, 1])), dtype='float32', shape=(params.n_users, params.n_items))
valid_matrix = sp.csr_matrix((np.ones_like(valid_list[:, 0]), (valid_list[:, 0], valid_list[:, 1])), dtype='float32', shape=(params.n_users, params.n_items))

user_pos_dict = csr_to_user_dict(train_matrix)
item_user_dict = csr_to_user_dict(train_matrix.transpose())
num_trainings = sum([len(item) for u, item in user_pos_dict.items()])
user_pos_dict = {u: np.array(item) for u, item in user_pos_dict.items()}
review_dict = {(train_list[i, 0], train_list[i, 1]) : i for i in range(len(train_list))}
users_list = np.arange(params.n_users)

summary = torch.FloatTensor(summary)
review = torch.FloatTensor(review)
critic_review = torch.FloatTensor(critic_review)

# 处理critic数据
critic_data['score'] = pd.to_numeric(critic_data['score'], downcast='integer')
critic_data = critic_data[['CRITIC_ID', 'ITEM_ID', 'domain', 'score']].values
n_critic_data = len(critic_data)
params.n_critics = np.max(critic_data[:, 0]) + 1
critic_rating = (critic_data[:, 3].astype(np.float32))/np.max(critic_data[:, 3]).astype(np.float32)
critic_matrix = sp.csr_matrix((critic_rating, (critic_data[:, 0], critic_data[:, 1])), dtype='float32', shape=(params.n_critics, params.n_items))
item_critic_dict = csr_to_user_dict(critic_matrix.transpose())
critic_review_dict = {(critic_data[i, 0], critic_data[i, 1]) : i for i in range(n_critic_data)}
for i in range(params.n_items):
    if i not in item_critic_dict.keys():
        item_critic_dict[i] = []
ones_tensor = torch.ones(params.n_items, 1, dtype=torch.long).to(params.device)

update_count = 0.0
model = Model(params, summary, review, critic_review).to(params.device)
optimizer = optim.Adam(model.parameters(), args.lr)
# scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

total_anneal_steps = args.step
posprobe = csr_to_user_dict(valid_matrix)
n_validusers = len(posprobe.keys())
recall, precision, map, ndcg = [], [], [], []
idxlist, idxitemlist, idxcriticlist, idxcriticidlist = list(range(params.n_users)), list(range(params.n_items)), list(range(n_critic_data)), list(range(params.n_critics))
n_batch = math.ceil(params.n_users / params.batch_size)
criticid_batch_size = math.ceil(params.n_critics / n_batch)
n_batch_valid = math.ceil(params.n_users / args.batch_size_valid)
for epoch in range(1, params.epochs + 1):
    training_start_time = time()
    model.train()
    total_loss, total_rating_loss, total_kl_loss, total_rating_w_loss, total_KL_w = 0.0, 0.0, 0.0, 0.0, 0.0
    np.random.shuffle(idxlist)
    np.random.shuffle(idxcriticlist)

    for i in range(n_batch):
        st_idx = i * params.batch_size
        end_idx = min(st_idx + params.batch_size, params.n_users)
        input_ph = torch.FloatTensor(train_matrix[idxlist[st_idx:end_idx]].toarray()).to(params.device)
        users = users_list[idxlist[st_idx:end_idx]]
        users = torch.tensor(users, dtype=torch.long).to(params.device)
        st_criticid_idx = i * criticid_batch_size
        end_criticid_idx = min(st_criticid_idx + criticid_batch_size, params.n_critics)
        critic_input_ph = torch.FloatTensor(critic_matrix[idxcriticidlist[st_criticid_idx:end_criticid_idx]].toarray()).to(params.device)
        
        user_review_items = [[review_dict[(ii, x)] for x in np.random.permutation(user_pos_dict[ii])[:params.max_len]] for ii in idxlist[st_idx:end_idx]]
        user_review_items = pad_sequences(user_review_items, max_len=params.max_len)
        mask_ui = torch.tensor((user_review_items != 0).astype(np.int64), dtype=torch.long).to(params.device)
        user_review_items = torch.tensor(user_review_items, dtype=torch.long).to(params.device)
        
        item_review_users = [[review_dict[(x, ii)] for x in np.random.permutation(item_user_dict[ii])[:params.max_len]] for ii in list(range(params.n_items))]
        item_review_users = pad_sequences(item_review_users, max_len=params.max_len)
        mask_iu = torch.tensor((item_review_users != 0).astype(np.int64), dtype=torch.long).to(params.device)
        mask_iu = torch.cat((ones_tensor, mask_iu), dim=1)
        item_review_users = torch.tensor(item_review_users, dtype=torch.long).to(params.device)
        
        item_review_critics = [[critic_review_dict[(x, ii)] for x in np.random.permutation(item_critic_dict[ii])[:params.max_len]] for ii in list(range(params.n_items))]
        item_review_critics = pad_sequences(item_review_critics, max_len=params.max_len)
        mask_ic = torch.tensor((item_review_critics != 0).astype(np.int64), dtype=torch.long).to(params.device)
        mask_ic = torch.cat((ones_tensor, mask_ic), dim=1)
        item_review_critics = torch.tensor(item_review_critics, dtype=torch.long).to(params.device)

        model.anneal = min(1.0, 1. * update_count / total_anneal_steps)
        update_count += 1
        optimizer.zero_grad()
        loss, rating_loss, kl_u, rating_w_loss, KL_w = model(input_ph, critic_input_ph, users, user_review_items, mask_ui, item_review_users, mask_iu, item_review_critics, mask_ic, is_train=1)
        
        total_loss += loss
        total_rating_loss += rating_loss
        total_kl_loss += kl_u
        total_rating_w_loss += rating_w_loss
        total_KL_w += KL_w
        
        loss.backward()
        optimizer.step()
        
    # scheduler.step(epoch=epoch)
    print('Epoch {} loss: {:.4f}, rating_loss: {:.4f}, kl_u: {:.4f}, rating_w_loss: {:.4f}, kl_w: {:.4f}, time: {} \n'.format(epoch, total_loss, total_rating_loss, total_kl_loss, total_rating_w_loss, 
        total_KL_w, time() - training_start_time))

    if epoch > args.st_epoch and epoch % params.print_epoch == 0:
        model.eval()
        test_start_time1 = time()
        recall_batch, precision_batch, map_batch, ndcg_batch = 0, 0, 0, 0
        with torch.no_grad():
            for i in range(n_batch_valid):
                st_idx = i * args.batch_size_valid
                end_idx = min(st_idx + args.batch_size_valid, params.n_users)
                users = users_list[st_idx:end_idx]
                users = torch.tensor(users, dtype=torch.long).to(params.device)
                valid_batch_matrix = train_matrix[st_idx:end_idx].toarray()
                
                user_review_items = [[review_dict[(ii, x)] for x in np.random.permutation(user_pos_dict[ii])[:params.max_len]] for ii in users_list[st_idx:end_idx]]
                user_review_items = pad_sequences(user_review_items, max_len=params.max_len)
                mask_ui = torch.tensor((user_review_items != 0).astype(np.int64), dtype=torch.long).to(params.device)
                user_review_items = torch.tensor(user_review_items, dtype=torch.long).to(params.device)
                
                epcoh_rating = model.get_prediction(torch.FloatTensor(valid_batch_matrix).to(params.device), users, user_review_items, mask_ui, item_review_users, mask_iu, 
                                    item_review_critics, mask_ic) - 99999 * valid_batch_matrix
                recall_tmp, precision_tmp, map_tmp, ndcg_tmp = evaluate12(posprobe, st_idx, end_idx, epcoh_rating, [5, 10, 20])
                recall_batch, precision_batch, map_batch, ndcg_batch = recall_tmp + recall_batch, precision_tmp + precision_batch, map_tmp + map_batch, ndcg_tmp + ndcg_batch
            recall_batch, precision_batch, map_batch, ndcg_batch = recall_batch / n_validusers, precision_batch / n_validusers, map_batch / n_validusers, ndcg_batch / n_validusers
            print(precision_batch, recall_batch, map_batch, ndcg_batch, time() - test_start_time1)

        precision.append(precision_batch)
        recall.append(recall_batch)
        map.append(map_batch)
        ndcg.append(ndcg_batch)
        evaluation = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(map), pd.DataFrame(ndcg)], axis=1)
        filename = "dumper/causal_" + args.log + "_" + str(args.lr) + "_" + str(args.keep_prob) + "_" + str(args.reg) + "_" + str(args.eta) + "_" + str(args.step) + "_" + str(args.embed_dim)
        evaluation.to_csv(filename + ".csv", header=False, index=False)







