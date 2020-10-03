#!/usr/bin/env python3
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import torch
from models import bart_model
from  data.data import SimpleDataset, seq2seq_padding_util
import math
from transformers import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import argparse
import pandas as pd
import json
from utils.logger_utils import *
logger = get_logger()

bart_large_config = {
    "activation_dropout": 0.0,
    "vocab_size": 50265,
    "pad_token_id": 1,
    "d_model": 1024,
    "encoder_ffn_dim": 4096,
    "encoder_layers": 12,
    "encoder_attention_heads": 16,
    "decoder_ffn_dim": 4096,
    "decoder_layers": 12,
    "decoder_attention_heads": 16,
    "encoder_layerdrop": 0.0,
    "decoder_layerdrop": 0.0,
    "attention_dropout": 0.1,
    "dropout": 0.1,
    "max_position_embeddings": 1024,
    "init_std": 0.02,
    "pretrained_path": "pytorch_model.bin", # download model file from https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/pytorch_model.bin
}

def read_corpus_jsonl(corpusfile):
    """
    read corpus file in jsonl format {'id': , 'body': }
    :param corpusfile:
    :return: a dict
    """
    corpus = {}
    with open(corpusfile) as fp:
        for line in fp:
            item = json.loads(line.strip())
            qid = item['id']
            text = item['body']
            corpus[str(qid)] = text
    return corpus

def read_corpus(corpusfile):
    """
    read corpus file or query files in TSV format
    :param corpusfile:
    :return: a dict
    """
    df = pd.read_csv(corpusfile, sep='\t', header=None, names=['qid', 'text'])
    corpus = {}
    for item in df.itertuples():
        qid = item.qid
        text = item.text
        corpus[str(qid)] = text
    return corpus


def read_qrels(qrelsfile):
    """
    read qrel files in TSV format
    :param qrelsfile:
    :return: a dict
    """
    qrels = pd.read_csv(qrelsfile, sep='\t', header=None, names=['qid', 'unused', 'id', 'label'])
    positive_pairs = []
    for item in qrels.itertuples():
        query_id = str(item.qid)
        id = str(item.id)
        label = item.label
        if label > 0:
            positive_pairs.append((query_id, id))
    return positive_pairs


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def load_train_data(tokenizer, corpus, queries, pairs):
    src_tgt = []
    for qid, pid in pairs:
        src = corpus[pid]
        # bpe_tokens, src_bpe_ids = tokenizer.encode(src)
        src_bpe_ids = tokenizer.encode(src)

        tgt = queries[qid]
        # bpe_tokens, tgt_bpe_ids = tokenizer.encode(tgt)
        tgt_bpe_ids = tokenizer.encode(tgt)
        src_tgt.append({'0': src_bpe_ids, '1': tgt_bpe_ids})

    dataset = SimpleDataset(src_tgt)
    return dataset


def load_test_data(tokenizer, corpus):
    source = []
    for pid in corpus:
        src = corpus[pid]
        # bpe_tokens, bpe_ids = tokenizer.encode(src)
        bpe_ids = tokenizer.encode(src)
        source.append({'0': bpe_ids})
    return SimpleDataset(source)

def generate(model, dataloader, args):
    model.eval()
    question_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch[0]
            questions = model.generate(input_ids=input_ids.to(device),
                                       beam_size=args.num_beams, max_decode_step=args.max_output_len
            )
            dec = [tokenizer.decode(g.token_ids, skip_special_tokens=True).strip() for g in questions]
            if len(dec) != len(batch[0]) * args.num_beams:
                import pdb; pdb.set_trace()
            assert len(dec) == len(batch[0]) * args.num_beams

            question_list += dec

    return question_list

def do_sample(model, dataloader, args):
    model.eval()
    question_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch[0]
            questions = model.sampling(input_ids=input_ids.to(device), num_samples=args.num_samples, top_p=args.top_p, top_k=args.top_k, max_decode_step=args.max_output_len)
            dec = [tokenizer.decode(g.token_ids, skip_special_tokens=True).strip() for g in questions]
            if len(dec) != len(batch[0]) * args.num_samples:
                import pdb; pdb.set_trace()
            assert len(dec) == len(batch[0]) * args.num_samples
            question_list += dec

    return question_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--corpusfile', default=None, type=str)
    parser.add_argument('--train_queryfile', default=None, type=str)
    parser.add_argument('--train_qrelfile', default=None, type=str)
    parser.add_argument('--valid_queryfile', default=None, type=str)
    parser.add_argument('--valid_qrelfile', default=None, type=str)
    parser.add_argument('--max_input_len', default=100, type=int)
    parser.add_argument('--max_output_len', default=100, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--warmup', default=0.1, type=float)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--print_iter', default=100, type=int)
    parser.add_argument('--eval_iter', default=1000, type=int)
    parser.add_argument('--log', default=None, type=str)
    parser.add_argument('--save_model', default='best_model.pt', type=str)
    parser.add_argument('--model_path', default='best_model.pt', type=str)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_corpusfile', default=None, type=str)
    parser.add_argument('--test_outputfile', default=None, type=str)
    parser.add_argument('--generated_queriesfile', default=None, type=str)
    parser.add_argument('--generated_answersfile', default=None, type=str)
    parser.add_argument('--num_beams', default=5, type=int)
    parser.add_argument('--num_samples', default=5, type=int)
    parser.add_argument('--do_sample', default=False, action='store_true')
    parser.add_argument('--top_p', default=1, type=float)
    parser.add_argument('--top_k', default=0, type=int)

    args = parser.parse_args()
    if args.log:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fhandler = logging.FileHandler(args.log)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    logger.info(args)
    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    # Init the model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', add_prefix_space=True)
    model = bart_model.TransformerModel(config=bart_large_config, tokenizer=tokenizer)
    seq2seq_padding_utilc = lambda x: seq2seq_padding_util(x, tokenizer.pad_token_id, args.max_input_len, args.max_output_len)
    if args.test:
        try:
            model_params = torch.load(args.model_path, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(model_params, strict=False)
            logger.info(f'Loading model from {args.model_path} succeeded!')
        except:
            raise ValueError(f'Load model from {args.model_path} failed!')
        model.to(device)
        # prepare test data
        if args.test_corpusfile.endswith('.jsonl'):
            test_corpus = read_corpus_jsonl(args.test_corpusfile)
        else:
            test_corpus = read_corpus(args.test_corpusfile)
        test_data = load_test_data(tokenizer, corpus=test_corpus)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size, collate_fn=seq2seq_padding_utilc) # right now only support batch size 1
        logger.info(f'Test data: {len(test_data)} samples.')
        if args.do_sample:
            logger.info(f'Sampling')
            questions = do_sample(model, test_dataloader, args)
        else:
            logger.info(f'Beam Search')
            questions = generate(model, test_dataloader, args)

        assert (len(test_data) * args.num_samples == len(questions))
        with open(args.test_outputfile, 'w') as fp:
            for i in tqdm(range(len(test_data))):
                qs = questions[i * args.num_samples: (i + 1) * args.num_samples]
                q = '[SEP]'.join(qs)
                fp.write(q + '\n')

        # model.eval()
        # with open(args.test_outputfile, 'w') as fp:
        #     with torch.no_grad():
        #         for step, batch in enumerate(tqdm(test_dataloader)):
        #             input_ids = batch[0].to(device)
        #             _, hypos = model(input_ids=input_ids)
        #             # question = tokenizer.decode(hypos[0].token_ids)
        #             # fp.write(question + '\n')
        #             # questions = [tokenizer.decode(hypo.token_ids) for hypo in hypos]
        #             questions = [tokenizer.decode(hypo.token_ids, skip_special_tokens=True).strip() for hypo in hypos]
        #             fp.write("[SEP]".join(questions) + '\n')
        logger.info(f'{len(questions)} questions written into file {args.test_outputfile}.')
        qid_to_questions = {}
        answers_dict = {}
        questions_to_qid = {}  # to trace duplicated questions
        count = 0
        psg_ids = list(test_corpus.keys())
        for i, did in enumerate(tqdm(psg_ids)):
            qs = questions[i * args.num_samples: (i + 1) * args.num_samples]
            for q in qs[:args.num_beams]:
                if q not in questions_to_qid:
                    qid = 'FQ' + str(count)
                    questions_to_qid[q] = qid
                    qid_to_questions[qid] = q
                    answers_dict[qid] = [did]
                    count += 1
                else:
                    qid = questions_to_qid[q]
                    if did not in answers_dict[qid]:
                        answers_dict[qid] += [did]
        assert (len(answers_dict) == len(qid_to_questions))

        json.dump(qid_to_questions, open(args.generated_queriesfile, 'w'), indent=2)
        logger.info(f'{len(qid_to_questions)} unique questions written into {args.generated_queriesfile}')

        json.dump(answers_dict, open(args.generated_answersfile, 'w'), indent=2)
        logger.info(f'labels written into {args.generated_answersfile}')

    else:
        if args.gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

        # prepare training data
        corpus = read_corpus(args.corpusfile)
        logger.info(f'Read corpus: {args.corpusfile}')
        train_query = read_corpus(args.train_queryfile)
        logger.info(f'Read train queries: {args.train_queryfile}')
        train_pair = read_qrels(args.train_qrelfile)
        logger.info(f'Read train qrels: {args.train_qrelfile}')
        train_data = load_train_data(tokenizer, corpus=corpus, queries=train_query, pairs=train_pair)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size,
                                      collate_fn=seq2seq_padding_utilc)
        logger.info('Training data: %d samples, %d batches', len(train_data), len(train_dataloader))

        if args.eval_iter > 0:
            val_query = read_corpus(args.valid_queryfile)
            logger.info(f'Read valid queries: {args.valid_queryfile}')
            val_pair = read_qrels(args.valid_qrelfile)
            logger.info(f'Read valid qrels: {args.valid_qrelfile}')
            val_data = load_train_data(tokenizer, corpus=corpus, queries=val_query, pairs=val_pair)
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size,
                                        collate_fn=seq2seq_padding_utilc)
            logger.info('Validation data: %d samples, %d batches', len(val_data), len(val_dataloader))

        def evaluation(model, val_data):
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_data):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, output_ids = batch
                    val_loss, _ = model(input_ids=input_ids, target_ids=output_ids)
                    total_loss += val_loss.mean()
            model.train()
            avg_loss = total_loss/len(val_data)
            return avg_loss, math.exp(avg_loss)

        # prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.wd,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        lr = args.lr
        t_total = len(train_dataloader)*args.epochs
        num_warmup_steps = int(args.warmup * t_total)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )
        # start training
        model.train()
        global_step = 0
        best_ppl = 1e8
        for _ in tqdm(range(args.epochs)):
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, output_ids = batch
                loss, _ = model(input_ids=input_ids, target_ids=output_ids)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if global_step % args.print_iter == 0:
                    logger.info(f'Global step {global_step}:  Train loss: {loss.item():.3f}')
                if args.eval_iter > 0 and global_step % args.eval_iter == 0:
                    val_loss, val_ppl = evaluation(model, val_dataloader)
                    logger.info(f'Global step {global_step}:  val loss: {val_loss:.3f} PPL: {val_ppl:.2f}')
                    if val_ppl < best_ppl:
                        best_ppl = val_ppl
                        model_to_save = model.module if hasattr(model, "module") else model
                        torch.save(model_to_save.state_dict(), args.save_model)
        if args.eval_iter > 0:
            val_loss, val_ppl = evaluation(model, val_dataloader)
            logger.info(f'Global step {global_step}:  val loss: {val_loss:.3f} PPL: {val_ppl:.2f}')
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), args.save_model)
        else:
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), args.save_model)
        logger.info(f'Training Done! Best model saved into {args.save_model}.')



