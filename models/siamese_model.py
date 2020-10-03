#!/usr/bin/env python3
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#

import torch
from torch import nn
import random
import numpy as np
from transformers import BertConfig, BertModel, BertTokenizer

INF = 1e5

class SeqPooler(nn.Module):
    def __init__(self, hidden_size, embed_size, mode='CLS', normalize=False):
        super().__init__()
        self.dense = nn.Linear(hidden_size, embed_size)
        self.activation = nn.Tanh()
        self.mode = mode
        self.normalize = normalize

    def forward(self, hidden_states, attention_mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # TODO: to apply masking
        if self.mode == 'CLS':
            token_tensor = hidden_states[:, 0]
        elif self.mode == 'mean':
            if attention_mask:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                seq_lens = torch.sum(attention_mask, axis=1, keepdim=True)
            else:
                seq_lens = hidden_states.shape[1]
            token_tensor = torch.sum(hidden_states, axis=1)/seq_lens
        elif self.mode == 'max':
            if attention_mask:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1) \
                                - INF * torch.ones_like(hidden_states)*(~attention_mask.unsqueeze(-1))
            token_tensor = torch.max(hidden_states, axis=1)[0]
        elif self.mode == 'sum':
            if attention_mask:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            token_tensor = torch.sum(hidden_states, axis=1)
        pooled_output = self.dense(token_tensor)
        pooled_output = self.activation(pooled_output)
        if self.normalize:
            pooled_output = torch.nn.functional.normalize(pooled_output)
        return pooled_output


class BertEmbed(nn.Module):
    def __init__(self, bert=None, embedding_size=512, mode='CLS', padding_id=0, normalize=False):
        super(BertEmbed, self).__init__()
        self.bert = bert
        self.embedding_size = embedding_size
        self.pooling = SeqPooler(bert.config.hidden_size, self.embedding_size, mode, normalize=normalize)
        self.padding_id = padding_id

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = ~(input_ids.eq(self.padding_id))
        hidden_output, pool_output = self.bert(input_ids, attention_mask=attention_mask)
        return self.pooling(hidden_output)