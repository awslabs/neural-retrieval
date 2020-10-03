#!/usr/bin/env python3
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#

import json
import torch
import random
import numpy as np
import tqdm
from tqdm import tqdm
import os
from nltk.tokenize import sent_tokenize as sent_tok

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, DistributedSampler
from utils.logger_utils import *
logger = get_logger()


def file_stream(tokenizer, data_mode, passage_dir=None, answer_dir=None, query_dir=None, distributed_training=True,
                num_workers=4, padding_func=None, batch_size=5, sample_size=10, sent_keep=0.1, hard_negative_rate=0.0):
    if data_mode == 'WikiGQ':
        answer_files = [n for n in os.listdir(answer_dir) if n[-4:]=='json']
        for i in range(len(answer_files)):
            passage_file = os.path.join(passage_dir, f'p_{i}.jsonl')
            query_file = os.path.join(query_dir, f'q_{i}.json')
            answer_file = os.path.join(answer_dir, f'a_{i}.json')

            train_data = SiameseDataset(tokenizer=tokenizer, query_path=query_file,
                                            passage_path=passage_file,
                                            answer_path=answer_file, sampled_softmax_n=sample_size)

            train_sampler = DistributedSampler(train_data) if distributed_training else RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
                                              num_workers=num_workers, pin_memory=True,
                                              collate_fn=padding_func)

            logger.info('Training data %d : %d samples, %d batches', i, len(train_data), len(train_dataloader))
            yield train_dataloader, train_sampler

    elif data_mode == 'ICT':
        passage_files = [n for n in os.listdir(passage_dir) if n[-5:] == 'jsonl']
        for i in range(len(passage_files)):
            passage_file = os.path.join(passage_dir, f'p_{i}.jsonl')

            train_data = ICTDataset(tokenizer=tokenizer, passage_path=passage_file, sent_keep=sent_keep,
                                    sampled_softmax_n=sample_size, hard_negative_rate=hard_negative_rate)

            train_sampler = DistributedSampler(train_data) if distributed_training else RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
                                          num_workers=num_workers, pin_memory=True,
                                          collate_fn=padding_func)

            logger.info('Training data %d : %d samples, %d batches', i, len(train_data), len(train_dataloader))
            yield train_dataloader, train_sampler

    else:
        raise Exception('Invalid task.')

class SimpleDataset(Dataset):
    def __init__(self, data, transform=None):
        '''Simple dataset
        '''
        self.instances = data
        self.transform = transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.instances[index])
        else:
            return self.instances[index]

def padding_util(examples, padding_id, seq_len):
    length = max([len(example) for example in examples])
    length = min(length, seq_len)
    batch = np.ones((len(examples), length)) * padding_id
    for i, example in enumerate(examples):
        idx = min(len(example), length)
        batch[i, :idx] = example[:idx - 1] + [example[-1]]
    return torch.tensor(batch, dtype=torch.long)


def seq2seq_padding_util(examples, padding_id=1, src_len=100, tgt_len=10):
    length = max([len(example['0']) for example in examples])
    length = min(length, src_len)
    source = np.ones((len(examples), length)) * padding_id
    for i, example in enumerate(examples):
        idx = min(len(example['0']), length)
        source[i, :idx] = example['0'][:idx-1] + [example['0'][-1]]

    length = max([len(example.get('1','')) for example in examples])
    if length == 0:
        return torch.tensor(source, dtype=torch.long), None
    length = min(length, tgt_len)
    target = np.ones((len(examples), length)) * padding_id
    for i, example in enumerate(examples):
        idx = min(len(example['1']), length)
        target[i, :idx] = example['1'][:idx-1] + [example['1'][-1]]
    return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SiameseDataset(Dataset):
    def __init__(self, tokenizer, query_path, passage_path, answer_path, sampled_softmax_n, neg_path=None):

        self.tokenizer = tokenizer
        self.query_dict = self._get_qid_to_tok(query_path)
        self.passage_dict = self._get_pid_to_tok(passage_path)
        self.answer_list, self.answer_dict = self._load_answers(answer_path)
        self.sampled_softmax_n = sampled_softmax_n
        self.passageids_set = set(self.passage_dict.keys())
        if neg_path is not None:
            self.neg_dict = json.load(open(neg_path))
            logger.info(f'Loading negative examples from {neg_path}')
        else:
            self.neg_dict = None

    def _get_qid_to_tok(self, query_path):
        if self.tokenizer.vocab_size == 30522:
            query_path_cache = query_path + '.cache.pt'
        else:
            query_path_cache = query_path + '.cased.cache.pt'
        try:
            qid_tok_dict = torch.load(query_path_cache)
            logger.info(f'Loading from {query_path_cache}!')
        except:
            query_dict = self._load_queries(query_path)
            qid_tok_dict = {}
            for k, v in query_dict.items():
                qid_tok_dict[k] = self.tokenizer.encode(v)
            torch.save(qid_tok_dict, query_path_cache)
            logger.info(f'{query_path_cache} saved!')
        return qid_tok_dict

    def _get_pid_to_tok(self, passage_path):
        if self.tokenizer.vocab_size == 30522:
            passage_path_cache = passage_path + '.cache.pt'
        else:
            passage_path_cache = passage_path + '.cased.cache.pt'
        try:
            pid_tok_dict = torch.load(passage_path_cache)
            logger.info(f'Loading from {passage_path_cache}!')
        except:
            passage_dict = self._load_passages(passage_path)
            pid_tok_dict = {}
            for k, v in passage_dict.items():
                pid_tok_dict[k] = self.tokenizer.encode(v)  # roberta tokenizer API might be different
            torch.save(pid_tok_dict, passage_path_cache)
            logger.info(f'{passage_path_cache} saved!')
        return pid_tok_dict

    def __len__(self):
        return len(self.answer_list)

    def __getitem__(self, idx):
        """
        query_token_ids, pos_psg_token_ids, [neg_psg_0_token_ids,..., neg_psg_k_token_ids]
        """
        # with dynamic negative sampling
        answer_ids = self.answer_list[idx]
        qid = answer_ids[0]
        pid = answer_ids[1]
        negative_psg_tokens_list = []
        if self.sampled_softmax_n > 0:
            if self.neg_dict is not None:
                candidates = self.neg_dict[qid]
                npids = random.sample(candidates, k=min(len(candidates), self.sampled_softmax_n))
            else:
                npids = random.sample(self.passageids_set - set(self.answer_dict[qid]), k=self.sampled_softmax_n)
            negative_psg_tokens_list = [self.passage_dict[pid] for pid in npids]
        query_tokens = self.query_dict[qid]
        postive_psg_tokens = self.passage_dict[pid]
        return query_tokens, postive_psg_tokens, negative_psg_tokens_list

    def _load_queries(self, query_path):
        return json.load(open(query_path))

    def _load_passages(self, passage_path):
        passage_lines = open(passage_path, 'r').readlines()
        passages = {}
        for line in tqdm(passage_lines):
            psgs = json.loads(line)
            if isinstance(psgs, dict): psgs = [psgs]
            for psg in psgs:
                passages[psg['id']] = psg['body']
        return passages

    def _load_answers(self, answer_path):
        answer_dict = json.load(open(answer_path, 'r'))
        answer_list = []
        for qid in answer_dict:
            for pid in answer_dict[qid]:
                answer_list.append([qid, pid])
        return answer_list, answer_dict


def batchify(examples, padding_id=0, max_query_len=64, max_passage_len=200):
    """
    batch_size x query_length, num_passages x passage_length, batch_size x1 (labels)
    """
    batch_size = len(examples)
    query_len = max([len(example[0]) for example in examples])
    query_len = min(query_len, max_query_len)
    query_tokens = np.ones((batch_size, query_len)) * padding_id
    labels = np.zeros(batch_size)
    passages = []
    count = 0
    for i, example in enumerate(examples):
        idx = min(len(example[0]), query_len)
        query_tokens[i, :idx] = example[0][:idx - 1] + [example[0][-1]]
        pos_psg = example[1]
        if pos_psg in passages:
            labels[i] = passages.index(pos_psg)
        else:
            passages.append(pos_psg)
            labels[i] = count
            count += 1
        neg_psgs = example[2]
        for psg in neg_psgs:
            if psg not in passages:
                passages.append(psg)
                count += 1
    passage_len = max([len(psg) for psg in passages])
    passage_len = min(passage_len, max_passage_len)
    passages_tokens = np.ones((count, passage_len)) * padding_id
    for i, psg in enumerate(passages):
        idx = min(len(psg), passage_len)
        passages_tokens[i, :idx] = psg[: idx - 1] + [psg[-1]]
    return torch.tensor(query_tokens, dtype=torch.long), torch.tensor(passages_tokens, dtype=torch.long), torch.tensor(
        labels, dtype=torch.long)


class ICTDataset(Dataset):
    """
    ICT data generator
    """
    def __init__(self, tokenizer, passage_path, sent_keep=0.1, sampled_softmax_n=10, hard_negative_rate=0.0):
        self.tokenizer = tokenizer
        self.passage_sent_dict = self._sent_tokenize_passages(passage_path)
        # valid passage means it has >1 sentences
        self.valid_passage_ids = [pid for pid in self.passage_sent_dict if len(self.passage_sent_dict[pid]) > 1]
        self.sent_keep = sent_keep
        self.sampled_softmax_n = sampled_softmax_n
        self.passageids_set = set(self.passage_sent_dict.keys())
        self.hard_negative_rate = hard_negative_rate

        # This could be done on the fly to save memory
        self.passage_dict = self._get_pid_to_tok(passage_path)

    def __len__(self):
        return len(self.valid_passage_ids)

    def __getitem__(self, idx):
        """
        query_token_ids, pos_psg_token_ids, [neg_psg_0_token_ids,..., neg_psg_k_token_ids], pos_id, [neg_id0, ..., neg_idk]
        """
        pos_pid = self.valid_passage_ids[idx]
        psg = self.passage_sent_dict[pos_pid]
        assert(len(psg) > 1)

        # randomly select a sentence as the question
        qid = random.choice(range(len(psg)))
        query = psg[qid]
        query_tokens = self.tokenizer.encode(query)

        # remove the selected sentence with 1 - sent_keep probability, this psg is the positive psg.
        sent_remove_flag = random.random() > self.sent_keep
        pos_psg = " ".join(psg[:qid] + psg[qid+sent_remove_flag:])
        postive_psg_tokens = self.tokenizer.encode(pos_psg)

        # randomly sample from the rest to get negative samples.

        # extract a percentage of hard examples
        hard_k = int(self.sampled_softmax_n * self.hard_negative_rate)
        passage_set = self.passageids_set - set([pos_pid])
        hard_negatives = set([item for item in passage_set if item.split('.')[0] == pos_pid.split('.')[0]])

        # handle edge case for 
        if len(hard_negatives) >= hard_k:
            npids_hard = random.sample(hard_negatives, k=hard_k)
        else:
            npids_hard = []
            hard_k = 0

        # extract a percentage of easy examples
        rest_k = self.sampled_softmax_n - hard_k
        npids_rest = random.sample(self.passageids_set - set([pos_pid]) - set(npids_hard), k=rest_k)

        # putting it all together
        npids = npids_hard + npids_rest
        negative_psg_tokens_list = [self.passage_dict[pid] for pid in npids]

        return query_tokens, postive_psg_tokens, negative_psg_tokens_list, pos_pid, npids

    def _sent_tokenize_passages(self, passage_path):
        """Tokenize the passage text into a list of sentences"""
        passage_lines = open(passage_path, 'r').readlines()
        passages = {}
        for line in tqdm(passage_lines):
            json_line = json.loads(line)
            passages[json_line['id']] = sent_tok(json_line['body'])
        return passages

    def _get_pid_to_tok(self, passage_path):
        passage_path_cache = passage_path + '.cache.pt'
        try:
            pid_tok_dict = torch.load(passage_path_cache)
            logger.info(f'Loading from {passage_path_cache}!')
        except:
            passage_dict = self._load_passages(passage_path)
            pid_tok_dict = {}
            for k, v in passage_dict.items():
                pid_tok_dict[k] = self.tokenizer.encode(v)  # roberta tokenizer API might be different
            torch.save(pid_tok_dict, passage_path_cache)
            logger.info(f'{passage_path_cache} saved!')
        return pid_tok_dict

    def _load_passages(self, passage_path):
        passage_lines = open(passage_path, 'r').readlines()
        passages = {}
        for line in tqdm(passage_lines):
            psgs = json.loads(line)
            if isinstance(psgs, dict): psgs = [psgs]
            for psg in psgs:
                passages[psg['id']] = psg['body']
        return passages


def ICT_batchify(examples, padding_id=0, max_query_len=64, max_passage_len=200):
    """
    batch_size x query_length, num_passages x passage_length, batch_size x1 (labels)
    """
    batch_size = len(examples)
    query_len = max([len(example[0]) for example in examples])
    query_len = min(query_len, max_query_len)
    query_tokens = np.ones((batch_size, query_len)) * padding_id
    labels = np.zeros(batch_size)
    passages = []
    count = 0
    psg_ids = []
    for i, example in enumerate(examples):
        idx = min(len(example[0]), query_len)
        query_tokens[i, :idx] = example[0][:idx - 1] + [example[0][-1]]
        pos_psg = example[1]
        passages.append(pos_psg)
        psg_ids.append(example[3])
        labels[i] = count
        count += 1

    for i, example in enumerate(examples):
        neg_psgs = example[2]
        pids = example[4]
        for i, psg in enumerate(neg_psgs):
            if pids[i] not in psg_ids:
                passages.append(psg)
                psg_ids.append(pids[i])
                count += 1

    passage_len = max([len(psg) for psg in passages])
    passage_len = min(passage_len, max_passage_len)
    passages_tokens = np.ones((count, passage_len)) * padding_id
    for i, psg in enumerate(passages):
        idx = min(len(psg), passage_len)
        passages_tokens[i, :idx] = psg[: idx - 1] + [psg[-1]]
    return torch.tensor(query_tokens, dtype=torch.long), torch.tensor(passages_tokens, dtype=torch.long), torch.tensor(
        labels, dtype=torch.long)
        

