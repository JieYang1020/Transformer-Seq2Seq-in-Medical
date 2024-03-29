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
                content_data[sentence_idx][word_idx] = word2id[word]
    '''label_data的word2id'''
    label_list = []
    for idx, sentence in enumerate(label_data):
        # 之后需添加结尾标识(开始标识'SOS'在训练时添加)
        # sentence.insert(0, 'SOS')
        sentence.append('EOS')
        for word in sentence:
            if word not in label_list:
                label_list.append(word)
    ids = range(len(label_list))
    label2id = pd.Series(ids, index=label_list).to_dict()
    for sentence_idx, sentence in enumerate(label_data):
        for word_idx, word in enumerate(sentence):
            if word in label2id.keys():
                label_data[sentence_idx][word_idx] = label2id[word]
    return content_data, label_data

def deal_label(label_data):
    labels_total=list(label_data)
    for idx1, labels_dense in enumerate(labels_total):
        num_labels = len(labels_dense)
        num_classes = config.num_category
        labels_one_hot = np.zeros((num_labels, num_classes))
        for idx2, label_one_hot in enumerate(labels_one_hot):
            '''取值 作为indx'''
            labels_one_hot[idx2][labels_total[idx1][idx2]] = 1
            # print(labels_total[idx1][idx2])
            '''临时将最后一个值取做实际值，算loss值的时候换为0'''
            # labels_one_hot[idx2][-1] = labels_total[idx1][idx2]
            labels_one_hot[idx2][-1] = 0
        '''pad标志：999'''
        for idx2, label_one_hot in enumerate(labels_one_hot):
            if idx2!= 0 and labels_total[idx1][idx2] == 0:
                labels_one_hot[idx2][0] = 999
        labels_total[idx1] = labels_one_hot
    labels_total = np.array(labels_total)
    return labels_total

def data_prepare():
    content_data = np.load(r'C:\Users\lenovo\Desktop\Josie\自学\哥哥的\论文实验seq2seq_正式版\data\symptoms.npy',
                           allow_pickle=True)
    label_data = np.load(r'C:\Users\lenovo\Desktop\Josie\自学\哥哥的\论文实验seq2seq_正式版\data\herbs.npy',
                         allow_pickle=True)
    content_data, label_data = word2id(content_data, label_data)
    word_label = label_data
    def max_len(content_data):
        max_len = 0
        for sentence_idx, sentence in enumerate(content_data):
            if max_len < len(sentence):
                max_len = len(sentence)
        return max_len
    '''用max_len补齐content_data'''
    for sentence in content_data:
        if len(sentence) < max_len(label_data):
            sentence += [0] * (max_len(label_data) - len(sentence))
    '''label_data也不一样长，同理需补齐label_data(begin:1, end:2, 剩余:0)
    算loss值时需是三维[batch_size, max_len, num_dim]'''
    for sentence in label_data:
        if len(sentence) < max_len(label_data):
            sentence += [0] * (max_len(label_data) - len(sentence))
    label_data = deal_label(label_data)
    np.set_printoptions(threshold=np.inf)
    return list(content_data), label_data