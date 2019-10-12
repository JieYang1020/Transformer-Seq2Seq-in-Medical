#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data_read import data_prepare
import config
from torchtext import data
from torchtext.data import Iterator, BucketIterator
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import recall_score, precision_score, f1_score
import sys

def residual(sublayer_fn, x):
    return sublayer_fn(x) + x

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism"""
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        '''前向传播。scale:缩放因子（1/(根号K)），一个浮点标量。返回：上下文张量和
        attention张量'''
        attention = torch.matmul(q, k.transpose(1, 2))
        # attention.shape[batch_size*num_heads, max_len, max_len]
        # attn_mask.shape[batch_size*num_heads, max_len, max_len]
        if scale:
            attention = attention * scale
        if attn_mask:
        # 给需要mask的地方设置负无穷
        # 暂时将padding_mask去掉
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention

# 将Q,K,V拆分为2份(heads=2)，每份分别进行scaled dot-product attention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # dim_per_head:q, k, v向量的长度.Wq, Wk, Wv 的矩阵尺寸为model_dim/dim_per_head
        self.linear_k = nn.Linear(config.model_dim, config.model_dim)
        self.linear_q = nn.Linear(config.model_dim, config.model_dim)
        self.linear_v = nn.Linear(config.model_dim, config.model_dim)
        self.dot_product_attention = ScaledDotProductAttention()
        self.linear_final = nn.Linear(config.model_dim, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        # multi_head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(config.model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接，增加任意常数，求导为1，避免反向传播时的梯度消失
        residual = query
        batch_size = key.size(0)
        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        # split by heads
        key = key.view(key.size(0) * config.num_heads, -1, config.dim_per_head)
        value = value.view(value.size(0) * config.num_heads, -1, config.dim_per_head)
        query = query.view(query.size(0) * config.num_heads, -1, config.dim_per_head)
        if attn_mask:
        # repeat是为了能在dot_product_attention中做mask_fill
            attn_mask = attn_mask.repeat(config.num_heads, 1, 1)
        scale = (key.size(-1) // config.num_heads) ** -0.5
        # 调用之前的scaled dot product attention
        context, attention = self.dot_product_attention(query, key, value,
                                                        scale, attn_mask)
        # 将2个头的结果连接
        context = context.view(context.size(0) // config.num_heads, -1, config.model_dim)
        output = self.linear_final(context)
        output = self.dropout(output)
        residual = residual.squeeze(1)
        output = self.layer_norm(residual + output)
        return output, attention

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size+2, config.model_dim, padding_idx=0)
        self.attention = MultiHeadAttention()
        self.linear = nn.Linear(config.max_len, config.num_category, bias=False)
        self.softmax = nn.Softmax()
    def forward(self, content):
        #content.shape:[batch_size, max_len, model_dim]
        embedding_output = self.embedding(content.long())
        '''Multi_head(中间差一步mask)'''
        self_attention_mask = None
        attention_output, attention = self.attention(embedding_output, embedding_output,
                                                     embedding_output, self_attention_mask)
        output = torch.max(attention_output, dim=2).values
        output = self.linear(output)
        # output = self.softmax(output)
        return output

def train(train_data_loader, val_data_loader):
    model = Transformer()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.learning_rate)
    for epoch in range(config.num_epochs):
        for batch_idx, (content, label) in enumerate(train_data_loader):
            '''需将label从[batch_size, num_category]转为[num_category]'''
            label = label.to(torch.float32)
            optimizer.zero_grad()
            predicted = model(content)
            loss = torch.mean(-torch.sum(label * torch.log(F.sigmoid(predicted)), dim=1))
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [batch_idx: {} content_len: {}] Loss: {:.6f}'.
                      format(epoch, batch_idx, len(content), loss.item()))

        test_loss = 0
        for batch_idx, (content, label) in enumerate(val_data_loader):
            label = label.to(torch.float32)
            predicted = model(content)
            print(label.shape, predicted.shape, np.int32(F.sigmoid(predicted)>0.5))
            recall = recall_score(label, np.int32(F.sigmoid(predicted)>0.5),average="micro")
            precision = precision_score(label, np.int32(F.sigmoid(predicted)>0.5), average="micro")
            F1 = f1_score(label, np.int32(F.sigmoid(predicted)>0.5), average="micro")
            test_loss += torch.mean(-torch.sum(label * torch.log(F.sigmoid(predicted)), dim=1))
            print('Val Epoch: {} Loss: {:.6f} Recall: {:.2f}% Precision: {:.2f}% F1: {:.2f}%'
                  .format(epoch, test_loss, 100.0*recall, 100.0*precision, 100.0*F1))

if __name__=="__main__":
    content_data, label_data = data_prepare()

    content_data = torch.tensor(np.array(content_data))
    label_data = torch.tensor(np.array(label_data))
    deal_dataset = TensorDataset(content_data, label_data)
    train_db, val_db = random_split(deal_dataset, [1305, 400])
    train_data_loader = DataLoader(dataset=train_db, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_data_loader = DataLoader(dataset=val_db,batch_size=400,shuffle=True,num_workers=0)
    train(train_data_loader, val_data_loader)

