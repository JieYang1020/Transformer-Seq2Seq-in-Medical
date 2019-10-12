# -*- coding: utf-8 -*-

batch_size = 64
num_epochs = 200
num_layers = 1
num_heads = 4
vocab_size = 186
#Y的分类
num_category = 206
model_dim = 512
dropout = 0.1
dim_per_head = model_dim // num_heads
learning_rate = 1e-5
'''data_read中得到max_len'''
max_len = 19
#LSTM中所用超参数
hidden_size = 128
'''每个句子的开始标记'''
SOS_token = 0