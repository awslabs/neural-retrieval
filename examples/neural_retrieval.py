#!/usr/bin/env python3
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import torch
from collections import OrderedDict
import math
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from tqdm import tqdm
import json
import argparse
import logging
from torch import nn
import random
import os
import numpy as np
import pandas as pd
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer, AutoModelWithLMHead, AutoModel
from data.data import *
from models.siamese_model import *
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from multiprocessing import Pool, cpu_count
from functools import partial
import utils.logger_utils
logger = utils.logger_utils.get_logger()

def set_seed():
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)

# configurations
model_name = 'bert-base-uncased'
tokenizer_class = BertTokenizer
model_class = BertModel
config_class = BertConfig

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--init_model', default=None, type=str)
parser.add_argument('--query_len', default=100, type=int)
parser.add_argument('--passage_len', default=100, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('--lr', default=3e-5, type=float)
parser.add_argument('--wd', default=0, type=float)
parser.add_argument('--sample_size', default=24, type=int)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--print_iter', default=100, type=int)
parser.add_argument('--eval_iter', default=1000, type=int)
parser.add_argument('--log', default=None, type=str)
parser.add_argument('--save_model', default='best_model.pt', type=str)
parser.add_argument('--passagefile', default=None, type=str)
parser.add_argument('--train_queryfile', default=None, type=str)
parser.add_argument('--train_answerfile', default=None, type=str)
parser.add_argument('--train_negativefile', default=None, type=str)
parser.add_argument('--val_passagefile', default=None, type=str)
parser.add_argument('--val_queryfile', default=None, type=str)
parser.add_argument('--val_answerfile', default=None, type=str)
parser.add_argument('--share', default=False, action='store_true')
parser.add_argument('--embed_size', default=128, type=int)
parser.add_argument('--mode', default='CLS', type=str)
parser.add_argument('--normalize', default=False, action='store_true')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--doc_embed', default=False, action='store_true')
parser.add_argument('--doc_embed_file', default=None, type=str)
parser.add_argument('--query_embed', default=False, action='store_true')
parser.add_argument('--query_embed_file', default=None, type=str)
parser.add_argument('--generate_retrieval', default=False, action='store_true')
parser.add_argument('--test_queryfile', default=None, type=str)
parser.add_argument('--topk', default=500, type=int)
parser.add_argument('--retrieval_outputfile', default=None, type=str)
parser.add_argument('--data_mode', default='WikiGQ', type=str)
parser.add_argument('--ICT_sent_keep', default=0.1, type=float)
parser.add_argument('--random', default=False, action='store_true')
parser.add_argument('--hard_negative_rate', default=0.0, type=float)
parser.add_argument('--neg_q', default=0, type=int)
parser.add_argument('--biobert', default=False, action='store_true')
# checkpoints
parser.add_argument('--ckpt_dir', default=None, type=str)
parser.add_argument('--resume', default=None, type=str, help='ckpt-epoch-file-steps')

# distributed training
parser.add_argument('--num_nodes', default=1, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--node_rank', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dist_url', default=None, type=str)


args = parser.parse_args()

set_seed()
if args.log:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler = logging.FileHandler(args.log)
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
logger.info(args)
n_gpus = args.num_gpus
num_nodes = args.num_nodes
world_size = n_gpus * num_nodes
distributed_training = True if world_size > 1 and args.local_rank > -1 else False
rank = args.node_rank * n_gpus + args.local_rank

if args.train and distributed_training:
    logger.info('setting up distributed traning... ')
    if num_nodes > 1:
        dist_url = args.dist_url
        dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)
    else:
        dist_url = "tcp://127.0.0.1:8080"
        dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)
    logger.info(
        "Initialized the distributed environment: NCCL backed on %d nodes." % (num_nodes))
    logger.info(
        "World size: %d, Node rank %d, GPU Rank %d, Local rank %d" % (
        world_size, args.node_rank, rank, args.local_rank))
    torch.cuda.set_device(args.local_rank)

device = torch.device('cuda') if args.gpu else torch.device('cpu')

# initialize tokenizer, configs, and bert model
logger.info('setting up model from configs')
if args.biobert:
    tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
    bert = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
else:
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True, cache_dir=None)
    config = config_class.from_pretrained(model_name)
    bert = model_class.from_pretrained(model_name, config=config)

if args.random:
    bert.init_weights()

doc_bert = BertEmbed(bert, embedding_size=args.embed_size, padding_id=tokenizer.pad_token_id,
                     mode=args.mode, normalize=args.normalize)

if args.share:
    logger.info('using tied weights')
    query_bert = doc_bert
else:
    logger.info('not sharing parameters')
    if args.biobert:
        tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
        bert = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
    else:
        tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True, cache_dir=None)
        config = config_class.from_pretrained(model_name)
        bert = model_class.from_pretrained(model_name, config=config)
    query_bert = BertEmbed(bert, embedding_size=args.embed_size, padding_id=tokenizer.pad_token_id,
                           mode=args.mode, normalize=args.normalize)

if args.init_model:
    try:
        missing_keys, unexpected_keys = doc_bert.load_state_dict(torch.load(args.init_model + '.doc', map_location='cpu'), strict=False)
        logger.info(f"Loading init Doc model {args.init_model}.doc")
        assert len(unexpected_keys) == 0
        assert len(missing_keys) == 0 or missing_keys == ["bert.embeddings.position_ids"]
        missing_keys, unexpected_keys = query_bert.load_state_dict(torch.load(args.init_model + '.query', map_location='cpu'), strict=False)
        logger.info(f"Loading init Query model {args.init_model}.query")
        assert len(unexpected_keys) == 0
        assert len(missing_keys) == 0 or missing_keys == ["bert.embeddings.position_ids"]
    except:
        logger.info("Loading init model failed!!!")

if args.resume:
    ckpt_dir = os.path.join(args.ckpt_dir, f'{args.resume}')
    doc_bert.load_state_dict(torch.load(os.path.join(ckpt_dir, 'ckpt.doc'), map_location='cpu'))
    logger.info(f"Resume Doc model {os.path.join(ckpt_dir, 'ckpt.doc')}")
    query_bert.load_state_dict(torch.load(os.path.join(ckpt_dir, 'ckpt.query'), map_location='cpu'))
    logger.info(f"Resume Query model {os.path.join(ckpt_dir, 'ckpt.query')}")
else:
    ckpt_dir = None

query_bert.to(device)
doc_bert.to(device)

if args.train:
    # prepare training data
    logger.info('preparing data')
    if args.data_mode == 'ICT':
        padding_func = lambda x: ICT_batchify(x, tokenizer.pad_token_id, args.query_len, args.passage_len)
    elif args.data_mode == 'WikiGQ':
        padding_func = lambda x: batchify(x, tokenizer.pad_token_id, args.query_len, args.passage_len)
    else:
        raise ValueError('Wrong Data Mode')

    if os.path.isdir(args.passagefile):
        # passage file is a directory (list of files). do streaming
        logger.info('filestream training')

        logger.info('computing number of query-answer pairs')
        num_pairs = 0
        if args.data_mode == 'wiki_gen_query':
            ans_files = [n for n in os.listdir(args.train_answerfile) if n[-4:] == 'json']
            for f in ans_files:
                ans_dict = json.load(open(os.path.join(args.train_answerfile, f), 'r'))
                num_answers = sum([len(ans) for ans in ans_dict.values()])
                num_pairs += num_answers
        elif args.data_mode == 'ICT':
            passage_files = [n for n in os.listdir(args.passagefile) if n[-5:] == 'jsonl']
            for f in passage_files:
                num_pairs += len(open(os.path.join(args.passagefile, f), 'r').readlines())

        else:
            raise Exception('Invalid task.')

        epoch_len = num_pairs // (args.batch_size * world_size)
        logger.info('Training data: %d samples, %d batches', num_pairs, epoch_len)
    else:
        logger.info('single file training')
        if args.data_mode == 'WikiGQ':
            train_data = SiameseDataset(tokenizer=tokenizer, query_path=args.train_queryfile, passage_path=args.passagefile,
                                        answer_path=args.train_answerfile, sampled_softmax_n=args.sample_size,
                                        neg_path=args.train_negativefile)

        elif args.data_mode == 'ICT':
            train_data = ICTDataset(tokenizer=tokenizer, passage_path=args.passagefile, sent_keep=args.ICT_sent_keep,
                                    sampled_softmax_n=args.sample_size, hard_negative_rate=args.hard_negative_rate)

        else:
            raise Exception('Invalid task.')

        train_sampler = DistributedSampler(train_data) if distributed_training else RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size,
                                      num_workers=args.num_workers, pin_memory=True,
                                      collate_fn=padding_func)
        logger.info('Training data: %d samples, %d batches', len(train_data), len(train_dataloader))
        train_dataloaders = [(train_dataloader, train_sampler)]
        epoch_len = len(train_dataloader)

    if args.eval_iter > 0:
        val_data = SiameseDataset(tokenizer=tokenizer, query_path=args.val_queryfile, passage_path=args.val_passagefile,
                              answer_path=args.val_answerfile, sampled_softmax_n=args.sample_size)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True,
                                    collate_fn=lambda x:
                                    batchify(x, tokenizer.pad_token_id, args.query_len, args.passage_len))
        logger.info('Validation data: %d samples, %d batches', len(val_data), len(val_dataloader))

    # prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters1 = [
        {
            "params": [p for n, p in query_bert.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in query_bert.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    lr = args.lr
    query_optimizer = AdamW(optimizer_grouped_parameters1, lr=lr, eps=1e-8)
    t_total = epoch_len * args.epochs
    num_warmup_steps = int(args.warmup * t_total)
    query_scheduler = get_linear_schedule_with_warmup(
        query_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )
    if (ckpt_dir and os.path.isfile(os.path.join(ckpt_dir, "query_optimizer.pt"))
            and os.path.isfile(os.path.join(ckpt_dir, "query_scheduler.pt"))):
        # Load in optimizer and scheduler states
        query_optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "query_optimizer.pt"), map_location='cpu'))
        query_scheduler.load_state_dict(torch.load(os.path.join(ckpt_dir, "query_scheduler.pt"), map_location='cpu'))
        logger.info(f'Load query optimizer states from {os.path.join(ckpt_dir, "query_optimizer.pt")}')


    if not args.share:
        optimizer_grouped_parameters2 = [
            {
                "params": [p for n, p in doc_bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.wd,
            },
            {
                "params": [p for n, p in doc_bert.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
        ]
        doc_optimizer = AdamW(optimizer_grouped_parameters2, lr=lr, eps=1e-8)
        doc_scheduler = get_linear_schedule_with_warmup(
            doc_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )
        if (ckpt_dir and os.path.isfile(os.path.join(ckpt_dir, "doc_optimizer.pt"))
                and os.path.isfile(os.path.join(ckpt_dir, "doc_scheduler.pt"))):
            # Load in optimizer and scheduler states
            doc_optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "doc_optimizer.pt"), map_location='cpu'))
            doc_scheduler.load_state_dict(torch.load(os.path.join(ckpt_dir, "doc_scheduler.pt"), map_location='cpu'))
            logger.info(f'Load doc optimizer states from {os.path.join(ckpt_dir, "doc_optimizer.pt")}')


    else:
        doc_bert = query_bert
        doc_optimizer = None
        doc_scheduler = None
    if args.num_gpus > 1 and not distributed_training:
        doc_bert = torch.nn.DataParallel(doc_bert)
        query_bert = torch.nn.DataParallel(query_bert)

    # multi-gpu training (should be after apex fp16 initialization)
    if distributed_training:
        logger.info('using distributed training')
        query_bert = torch.nn.parallel.DistributedDataParallel(query_bert, device_ids=[args.local_rank],
                                                               find_unused_parameters=True)
        if not args.share:
            doc_bert = torch.nn.parallel.DistributedDataParallel(doc_bert, device_ids=[args.local_rank],
                                                                 find_unused_parameters=True)
    else:
        logger.info('using single machine training')


def save_model(query_bert, doc_bert, save_model):
    query_model_to_save = query_bert.module if hasattr(query_bert, "module") else query_bert
    torch.save(query_model_to_save.state_dict(), save_model+'.query')
    doc_model_to_save = doc_bert.module if hasattr(doc_bert, "module") else doc_bert
    torch.save(doc_model_to_save.state_dict(), save_model + '.doc')


def evaluation(query_bert, doc_bert, loss_fun, val_data):
    query_bert.eval()
    doc_bert.eval()
    total_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_data)):
            batch = tuple(t.to(device) for t in batch)
            query_tokens, passages_tokens, labels = batch
            query_embeds = query_bert(query_tokens)
            passages_embeds = doc_bert(passages_tokens)
            scores = torch.mm(query_embeds, passages_embeds.t())
            val_loss = loss_fun(scores, labels).mean()
            total_loss += val_loss
    avg_loss = total_loss / len(val_data)
    query_bert.train()
    doc_bert.train()
    return avg_loss


def train():

    # start training
    logger.info('starting training')
    loss_fun = nn.CrossEntropyLoss()
    query_bert.train()
    doc_bert.train()
    scaler = GradScaler()
    global_step = 0
    best_loss = 1e8
    epochs_trained = 0
    files_trained_in_current_epoch = 0

    if args.resume:
        # resume: ckpt-epoch-file_number-step:
        try:
            ckpt_suffix = args.resume.split('-')[1:]
            global_step = int(ckpt_suffix[-1])
            epochs_trained = int(ckpt_suffix[0])
            files_trained_in_current_epoch = int(ckpt_suffix[1]) + 1
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d files in the first epoch", files_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    for epoch in tqdm(range(epochs_trained, args.epochs)):
        if os.path.isdir(args.passagefile):
            current_dataloaders = file_stream(tokenizer=tokenizer, data_mode=args.data_mode, passage_dir=args.passagefile,
                                            answer_dir=args.train_answerfile, query_dir=args.train_queryfile,
                                            distributed_training=distributed_training, num_workers=args.num_workers,
                                            padding_func=padding_func, batch_size=args.batch_size,
                                            sample_size=args.sample_size, sent_keep=args.ICT_sent_keep, hard_negative_rate=args.hard_negative_rate)
        else:
            current_dataloaders = train_dataloaders
        for num_file, (train_dataloader, train_sampler) in enumerate(current_dataloaders):
            if files_trained_in_current_epoch > 0:
                files_trained_in_current_epoch -= 1
                continue
            if distributed_training:
                train_sampler.set_epoch(epoch)

            for step, batch in enumerate(tqdm(train_dataloader)):
                query_bert.zero_grad()
                doc_bert.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                query_tokens, passages_tokens, labels = batch
                with autocast():
                    query_embeds = query_bert(query_tokens)
                    passages_embeds = doc_bert(passages_tokens)
                    scores = torch.mm(query_embeds, passages_embeds.t())
                    loss = loss_fun(scores, labels).mean()
                scaler.scale(loss).backward()
                scaler.step(query_optimizer)
                query_scheduler.step()  # Update learning rate schedule

                if not args.share:
                    scaler.step(doc_optimizer)
                    doc_scheduler.step()  # Update learning rate schedule
                scaler.update()

                global_step += 1
                if global_step % args.print_iter == 0:
                    logger.info(f'Global step {global_step}:  Train loss: {loss.item():.3f}')
                if args.eval_iter > 0 and global_step % args.eval_iter == 0 and (rank == 0 or not distributed_training):
                    val_loss = evaluation(query_bert, doc_bert, loss_fun,  val_dataloader)
                    logger.info(f'Global step {global_step}:  val loss: {val_loss:.3f}')
                    if val_loss < best_loss:
                        best_loss = val_loss
                        save_model(query_bert, doc_bert, args.save_model)
        # if args.ckpt_dir and os.path.isdir(args.passagefile) and (rank == 0 or not distributed_training):
        if args.ckpt_dir and (rank == 0 or not distributed_training):
            output_dir = os.path.join(args.ckpt_dir, f'ckpt-{epoch}-{num_file}-{global_step}')
            os.makedirs(output_dir, exist_ok=True)
            save_model(query_bert, doc_bert, os.path.join(output_dir, 'ckpt'))
            logger.info(f'Save checkpoint to {output_dir}')
            torch.save(query_optimizer.state_dict(), os.path.join(output_dir, "query_optimizer.pt"))
            torch.save(query_scheduler.state_dict(), os.path.join(output_dir, "query_scheduler.pt"))
            if not args.share:
                torch.save(doc_optimizer.state_dict(), os.path.join(output_dir, "doc_optimizer.pt"))
                torch.save(doc_scheduler.state_dict(), os.path.join(output_dir, "doc_scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
        if args.eval_iter <= 0 and (rank == 0 or not distributed_training):
            save_model(query_bert, doc_bert, args.save_model+str(epoch))
    if args.eval_iter > 0 and (rank == 0 or not distributed_training):
        val_loss = evaluation(query_bert, doc_bert, loss_fun, val_dataloader)
        logger.info(f'Global step {global_step}:  val loss: {val_loss:.3f}')
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(query_bert, doc_bert, args.save_model)
    logger.info(f'Training Done! Best model saved into {args.save_model}.')


def doc_embed():
    """Compute Passage Embeddings"""
    doc_bert = BertEmbed(bert, embedding_size=args.embed_size, padding_id=tokenizer.pad_token_id,
                         mode=args.mode, normalize=args.normalize)
    missing_keys, unexpected_keys = doc_bert.load_state_dict(torch.load(args.save_model + '.doc'), strict=False)
    logger.info(f'Load Passages Embedding model from {args.save_model}.doc')
    assert len(unexpected_keys) == 0
    assert len(missing_keys) == 0 or missing_keys == ["bert.embeddings.position_ids"]
    doc_bert.to(device)
    if args.num_gpus <=0:
        args.num_gpus = torch.cuda.device_count()

    if args.num_gpus > 1:
        doc_bert = torch.nn.DataParallel(doc_bert)
    doc_bert.eval()
    logger.info('Generate Passages Embeddings')
    passage_embedding_dict = {}

    with open(args.passagefile) as fp:
        for line in fp:
            items = json.loads(line.strip())
            if isinstance(items, dict): items = [items]
            for item in items:
                passage_embedding_dict[item['id']] = item['body']

    pids = list(passage_embedding_dict.keys())
    psgs = list(passage_embedding_dict.values())
    psg_data = SimpleDataset(psgs, transform=lambda x: tokenizer.encode(x, max_length=args.passage_len, truncation=True))
    num_psgs = len(psgs)
    psg_embeds = np.zeros((num_psgs, args.embed_size))

    padding_func = lambda x: padding_util(x, tokenizer.pad_token_id, args.passage_len)
    sampler = SequentialSampler(psg_data)
    psg_dataloader = DataLoader(psg_data, sampler=sampler, batch_size=args.batch_size, collate_fn=padding_func)
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(psg_dataloader, desc='Embedding Passages')):
            batch = batch.to(device)
            bz = batch.shape[0]
            embed = doc_bert(batch).data.cpu().numpy()
            psg_embeds[count:count+bz, :] = embed
            count += bz

    psg_df = pd.DataFrame(psg_embeds, index=pids)
    psg_df.to_csv(args.doc_embed_file)
    logger.info(f'Passage Embeddings saved into {args.doc_embed_file}.')


def generate_retrieval(retrieve=True):
    """Compute Query Embeddings and Retrieve passages"""
    try:
        query_df = pd.read_csv(args.query_embed_file, index_col=0)
        logger.info(f'Load Query Embeddings from {args.query_embed_file}.')
    except:
        missing_keys, unexpected_keys = query_bert.load_state_dict(torch.load(args.save_model + '.query'), strict=False)
        logger.info(f'Load Query Embedding model from {args.save_model}.query')
        assert len(unexpected_keys) == 0
        assert len(missing_keys) == 0 or missing_keys == ["bert.embeddings.position_ids"]
        query_bert.to(device)
        query_bert.eval()
        test_queries = json.load(open(args.test_queryfile))
        if isinstance(test_queries, list):
            test_queries = {q['query_id']: q['query'] for q in test_queries}
        qids = list(test_queries.keys())
        query_data = SimpleDataset(list(test_queries.values()), transform=tokenizer.encode)
        query_embeds = np.zeros((len(query_data), query_bert.embedding_size))
        sampler = SequentialSampler(query_data)
        padding_func = lambda x: padding_util(x, tokenizer.pad_token_id, args.query_len)
        query_dataloader = DataLoader(query_data, sampler=sampler, batch_size=args.batch_size, collate_fn=padding_func)
        if args.topk == 0: args.topk = len(psg_df)
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(query_dataloader, desc='Embedding Queries')):
                batch = batch.to(device)
                bz = batch.shape[0]
                embed = query_bert(batch).data.cpu().numpy()
                query_embeds[count:count + bz, :] = embed
                count += bz
        query_df = pd.DataFrame(query_embeds, index=qids)
        query_df.to_csv(args.query_embed_file)
        logger.info(f'Query Embeddings saved into {args.query_embed_file}')
    if retrieve:
        try:
            psg_df = pd.read_csv(args.doc_embed_file, index_col=0)
            logger.info(f'Load Passages Embeddings from {args.doc_embed_file}.')
        except:
            raise ValueError('Load Passage Embedding failed.')


        try:
            query_scores = psg_df.dot(query_df.to_numpy().transpose())
            results = {}
            for i, qid in enumerate(tqdm(query_scores.keys())):
                score = query_scores[qid].nlargest(args.topk)
                results[qid] = score.to_dict()

        except Exception as e:
            logger.info(e)
            # for very large files we will chunk the file for retrieval.
            results = {}
            for query_df in tqdm(pd.read_csv(args.query_embed_file, index_col=0, chunksize=1000)):
                query_scores = psg_df.dot(query_df.transpose())
                for i, qid in enumerate(tqdm(query_scores.keys())):
                    score = query_scores[qid].nlargest(args.topk)
                    results[qid] = score.to_dict()

        json.dump(results, open(args.retrieval_outputfile, 'w'))
        logger.info(f'Retrieval results saved into {args.retrieval_outputfile}')


if __name__== "__main__":
    if args.train:
        train()
    if args.doc_embed:
        doc_embed()
    if args.query_embed:
        generate_retrieval(args.generate_retrieval)
