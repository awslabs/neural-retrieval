#!/usr/bin/env python3
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# This script processes the official Natural question dataset into the format we want
import gzip
import json
import sys
import tqdm
from tqdm import tqdm
import argparse
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', default='v1.0-simplified-train.jsonl.gz')
    parser.add_argument('--devfile', default='v1.0-simplified-dev.jsonl.gz')
    parser.add_argument('--passagefile', default='all_passages.jsonl')
    parser.add_argument('--queries_trainfile', default='train_queries.json')
    parser.add_argument('--answers_trainfile', default='train_answers.json')
    parser.add_argument('--queries_devfile', default='dev_queries.json')
    parser.add_argument('--answers_devfile', default='dev_answers.json')
    parser.add_argument('--qrelsfile', default='all_qrels.txt')

    args = parser.parse_args()

    traindata = []

    with gzip.open(args.trainfile, 'rb') as fp:
        for i, line in enumerate(tqdm(fp, total=307373, desc='Reading trainset')):
            item = json.loads(line.strip())
            eid = item['example_id']
            doc = item.pop('document_text')
            item.pop('long_answer_candidates')
            pids = []
            paras = []
            for ans in item.pop('annotations'):
                lans = ans['long_answer']
                pid = lans['candidate_index']
                st = lans['start_token']
                et = lans['end_token']
                if pid not in pids:
                    pids.append(pid)
                    para = doc.split(" ")[st:et]
                    paras.append(para)
            if len(pids) > 1:
                print(eid)
            item['paragraphs'] = paras
            traindata.append(item)

    train_questions = {}
    passages = []
    train_answer = {}
    count = 0
    for example in tqdm(traindata, desc='filtering'):
        eid = example['example_id']
        ans = []
        for para in example['paragraphs']:
            if len(para) == 0:
                continue
            if para[0] != '<P>' or para[-1] != '</P>':
                continue
            if para not in passages:
                passages.append(para)
                pid = 'D' + str(count)
                count += 1
            else:
                pid = passages.index(para)
                pid = 'D' + str(pid)
            ans.append(pid)

        if len(ans) > 0:
            train_questions[eid] = example['question_text']
            train_answer[eid] = ans
        assert (count == len(passages))
        assert (len(train_answer) == len(train_questions))
    json.dump(train_questions, open(args.queries_trainfile, 'w'), indent=2)
    json.dump(train_answer, open(args.answers_trainfile, 'w'), indent=2)
    logger.info(f'{len(train_questions)} queries and answers in training set written into '
                f'{args.queries_trainfile} and {args.answers_trainfile}')


    devdata = []
    with gzip.open(args.devfile, 'rb') as fp:
        for i, line in enumerate(tqdm(fp, total=7830, desc='Reading devset')):
            item = json.loads(line.strip())
            eid = item['example_id']
            doc = item.pop('document_text')
            item.pop('long_answer_candidates')
            pids = []
            paras = []
            for ans in item.pop('annotations'):
                lans = ans['long_answer']
                pid = lans['candidate_index']
                st = lans['start_token']
                et = lans['end_token']
                if pid not in pids:
                    pids.append(pid)
                    para = doc.split(" ")[st:et]
                    paras.append(para)
            item['paragraphs'] = paras
            devdata.append(item)

    dev_questions = {}
    dev_answer = {}
    for example in tqdm(devdata, desc='filtering'):
        eid = example['example_id']
        ans = []
        for para in example['paragraphs']:
            if len(para) == 0:
                continue
            if para[0] != '<P>' or para[-1] != '</P>':
                continue
            if para not in passages:
                passages.append(para)
                pid = 'D' + str(count)
                count += 1
            else:
                pid = passages.index(para)
                pid = 'D' + str(pid)
            ans.append(pid)

        if len(ans) > 0:
            dev_questions[eid] = example['question_text']
            dev_answer[eid] = ans
    assert(count == len(passages))
    assert(len(dev_answer) == len(dev_questions))

    json.dump(dev_questions, open(args.queries_devfile, 'w'), indent=2)
    json.dump(dev_answer, open(args.answers_devfile, 'w'), indent=2)
    logger.info(f'{len(dev_questions)} queries and answers in dev set written into '
                f'{args.queries_devfile} and {args.answers_devfile}')

    with open(args.passagefile, 'w') as fp:
        for i, psg in enumerate(tqdm(passages, desc='writing passages')):
            item = {'id': 'D'+str(i), 'body': " ".join(psg[1:-1])}
            fp.write(json.dumps(item) + '\n')
    logger.info(f'{len(passages)} passages written into {args.passagefile}')

    with open(args.qrelsfile, 'w') as fp:
        for q in train_answer:
            for d in train_answer[q]:
                line = " ".join([str(q), 'Q0', str(d), str(1)])
                fp.write(line + '\n')
        for q in dev_answer:
            for d in dev_answer[q]:
                line = " ".join([str(q), 'Q0', str(d), str(1)])
                fp.write(line + '\n')
    logger.info(f'answers written into qrels file {args.qrelsfile}')
