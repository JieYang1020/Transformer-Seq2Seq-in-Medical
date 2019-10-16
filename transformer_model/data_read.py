#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import pandas as pd
import config

def word2id(content_data, label_data):
    '''content_data的word2id'''
    word_list = []
    for idx, sentence in enumerate(content_data):
        for word in sentence:
            if word not in word_list:
                word_list.append(word)
    ids = range(len(word_list))
    word2id = pd.Series(ids, index=word_list).to_dict()
    for sentence_idx, sentence in enumerate(content_data):
        for word_idx, word in enumerate(sentence):
            if word in word2id.keys():
                content_data[sentence_idx][word_idx] = word2id[word] + 1
    '''label_data的word2id'''
    label_list = []
    for idx, sentence in enumerate(label_data):
        for word in sentence:
            if word not in label_list:
                label_list.append(word)
    ids = range(len(label_list))
    label2id = pd.Series(ids, index=label_list).to_dict()
    for sentence_idx, sentence in enumerate(label_data):
        for word_idx, word in enumerate(sentence):
            if word in label2id.keys():
                #之后需添加开始/结尾标识，分别为1和2
                label_data[sentence_idx][word_idx] = label2id[word] + 1
    return content_data, label_data

def data_prepare():
    content_data = np.load(r'C:\Users\lenovo\Desktop\Josie\自学\哥哥的\论文实验_transformer实现多分类\data\symptoms.npy',
                           allow_pickle=True)
    label_data = np.load(r'C:\Users\lenovo\Desktop\Josie\自学\哥哥的\论文实验_transformer实现多分类\data\herbs.npy',
                         allow_pickle=True)
    content_data, label_data = word2id(content_data, label_data)
    def max_len(content_data):
        max_len = 0
        for sentence_idx, sentence in enumerate(content_data):
            if max_len < len(sentence):
                max_len = len(sentence)
        return max_len
    '''用max_len补齐content_data'''
    for sentence in content_data:
        if len(sentence) < max_len(content_data):
            sentence += [0] * (max_len(content_data) - len(sentence))
    '''label_data也不一样长，同理需补齐label_data(begin:1, end:2, 剩余:0)
    算loss值时需是三维[batch_size, max_len, num_dim]'''
    new_label_data = []
    for sentence in label_data:
        #补齐开始/结尾标识
        new_label_data.append([1] + sentence + [2])
    # label_data = new_label_data
    for sentence in label_data:
        if len(sentence) < config.num_category:
            sentence += [0] * (config.num_category - len(sentence))
    return list(content_data), list(label_data)