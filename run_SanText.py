import argparse
import json

import torch
import random
import numpy as np
import logging
import os
import math

logger = logging.getLogger(__name__)
from tqdm import tqdm
from scipy.special import softmax
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from utils import get_vocab_SST2, get_vocab_QQP, get_vocab_CoLA, get_vocab_CliniSTS, get_vocab_QNLI, word_normalize,get_vocab_gossipcop
from spacy.lang.en import English
from transformers import BertTokenizer, BertForMaskedLM
from SanText import SanText_plus, SanText_plus_init


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def write_to_tsv(file_path, texts, labels):
    with open(file_path, 'w', encoding='utf-8') as out_file:
        out_file.write("origin_text\tlabel\n")
        for text, label in zip(texts, labels):
            origin_text = text.replace("\n", " ")
            out_file.write(f"{origin_text}\t{label}\n")


# def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
#     distance = euclidean_distances(word_embed_1, word_embed_2)
#     sim_matrix = -distance
#     prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
#     return prob_matrix

def cal_probability(word_embed, sensitive_word_embed, epsilon, batch_size=1024):
    word_embed = torch.tensor(word_embed, dtype=torch.float32).cuda()
    sensitive_word_embed = torch.tensor(sensitive_word_embed, dtype=torch.float32).cuda()

    N = word_embed.shape[0]
    M = sensitive_word_embed.shape[0]
    prob_matrix = []

    for i in range(0, N, batch_size):
        batch = word_embed[i:i+batch_size]  # shape = [batch_size, dim]
        dist = torch.cdist(batch, sensitive_word_embed, p=2)  # [batch_size, M]
        prob = torch.exp(-0.5 * epsilon * dist)
        prob = prob / prob.sum(dim=1, keepdim=True)  # softmax normalize
        prob_matrix.append(prob.cpu())  # 搬回 CPU、節省 GPU 記憶體

    prob_matrix = torch.cat(prob_matrix, dim=0)  # [N, M]
    return prob_matrix.numpy()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="./data/SST-2/",
        type=str,
        help="The input dir"
    )

    parser.add_argument(
        "--bert_model_path",
        default="./base_bert_models",
        type=str,
        help="bert model name or path. leave it bank if you are using Glove"
    )

    parser.add_argument(
        "--output_dir",
        default="./output_SanText/QNLI/",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--word_embedding_path",
        default='./data/glove.840B.300d.txt',
        type=str,
        help="The pretrained word embedding path. leave it blank if you are using BERT",
    )

    parser.add_argument(
        "--word_embedding_size",
        default=300,
        type=int,
        help="The pretrained word embedding size. leave it blank if you are using BERT",
    )

    parser.add_argument(
        '--method',
        choices=['SanText', 'SanText_plus'],
        default='SanText_plus',
        help='Sanitized method'
    )

    parser.add_argument(
        '--embedding_type',
        choices=['glove', 'bert'],
        default='glove',
        help='embedding used for sanitization'
    )

    parser.add_argument('--task',
                        choices=['CliniSTS', "SST-2", "QNLI", "CoLA", "FakeNews", "QQP"],
                        default='SST-2',
                        help='NLP eval tasks')

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--epsilon", type=float, default=15, help="privacy parameter epsilon")
    parser.add_argument("--p", type=float, default=0.2,
                        help="SanText+: probability of non-sensitive words to be sanitized")

    parser.add_argument("--sensitive_word_percentage", type=float, default=0.5,
                        help="SanText+: how many words are treated as sensitive")

    parser.add_argument("--threads", type=int, default=12, help="number of processors")

    args = parser.parse_args()

    set_seed(args)

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Running method: %s, task: %s,  epsilon = %s, random_seed: %d" % (
        args.method, args.task, args.epsilon, args.seed))

    if args.method == "SanText":
        args.sensitive_word_percentage = 1.0
        args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon)
    else:
        args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon,
                                       "sword_%.2f_p_%.2f" % (args.sensitive_word_percentage, args.p))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Building Vocabulary...")

    if args.embedding_type == "glove":
        tokenizer = English()
        tokenizer_type = "word"
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        tokenizer_type = "subword"
    if args.task == "SST-2":
        vocab = get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "CliniSTS":
        vocab = get_vocab_CliniSTS(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "QNLI":
        vocab = get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "FakeNews":
        vocab = get_vocab_gossipcop(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "CoLA":
        vocab = get_vocab_CoLA(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "QQP":
        vocab = get_vocab_QQP(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    else:
        raise NotImplementedError

    sensitive_word_count = int(args.sensitive_word_percentage * len(vocab))
    words = [key for key, _ in vocab.most_common()]
    sensitive_words = words[-sensitive_word_count - 1:]

    sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}
    logger.info("#Total Words: %d, #Sensitive Words: %d" % (len(words), len(sensitive_words2id)))

    sensitive_word_embed = []
    all_word_embed = []

    word2id = {}
    sword2id = {}
    sensitive_count = 0
    all_count = 0
    if args.embedding_type == "glove":
        num_lines = sum(1 for _ in open(args.word_embedding_path, encoding='utf-8'))
        logger.info("Loading Word Embedding File: %s" % args.word_embedding_path)

        with open(args.word_embedding_path, encoding='utf-8') as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for row in tqdm(f, total=num_lines - 1):
                content = row.rstrip().split(' ')
                cur_word = word_normalize(content[0])
                if cur_word in vocab and cur_word not in word2id:
                    word2id[cur_word] = all_count
                    all_count += 1
                    emb = [float(i) for i in content[1:]]
                    all_word_embed.append(emb)
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)
                assert len(word2id) == len(all_word_embed)
                assert len(sword2id) == len(sensitive_word_embed)
            f.close()
    else:
        logger.info("Loading BERT Embedding File: %s" % args.bert_model_path)
        model = BertForMaskedLM.from_pretrained(args.bert_model_path)
        embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

        for cur_word in tokenizer.vocab:
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = all_count
                emb = embedding_matrix[tokenizer.convert_tokens_to_ids(cur_word)]
                all_word_embed.append(emb)
                all_count += 1

                if cur_word in sensitive_words2id:
                    sword2id[cur_word] = sensitive_count
                    sensitive_count += 1
                    sensitive_word_embed.append(emb)
            assert len(word2id) == len(all_word_embed)
            assert len(sword2id) == len(sensitive_word_embed)

    all_word_embed = np.array(all_word_embed, dtype='f')
    sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')

    logger.info("All Word Embedding Matrix: %s" % str(all_word_embed.shape))
    logger.info("Sensitive Word Embedding Matrix: %s" % str(sensitive_word_embed.shape))

    logger.info("Calculating Prob Matrix for Exponential Mechanism...")
    prob_matrix = cal_probability(all_word_embed, sensitive_word_embed, args.epsilon)

    threads = min(args.threads, cpu_count())

    if args.task == "FakeNews":
        input_file = os.path.join(args.data_dir, 'gossipcop.json')
        output_file = os.path.join(args.output_dir, 'train.tsv')
        output_file_dev = os.path.join(args.output_dir, 'dev.tsv')
        logger.info("Processing file: %s. Will write to: %s" % (input_file, output_file))
        # 读取 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as rf:
            data = json.load(rf)

            # 初始化数据结构
            docs = []
            labels = []
            generated_tones = []
            num_lines = len(data)
            for key, value in tqdm(data.items(), total=num_lines):
                x = 0
                origin_text = value['origin_text']
                # if any(key.startswith('generated_text') for key in value) and any(key1.startswith('generated_label') for key1 in value):
                #     x = 1
                #     generated_text = [value[key] for key in value if key.startswith('generated_text')][0]
                #     label2 = [value[key] for key in value if key.startswith('generated_label')][0]
                    # generated_tone = value['generated_tone']
                label1 = value['origin_label']

                if args.embedding_type == "glove":
                    doc1 = [token.text for token in tokenizer(origin_text)]
                    # if x == 1:
                    #     doc2 = [token.text for token in tokenizer(generated_text)]
                else:
                    doc1 = tokenizer.tokenize(origin_text)
                    # if x == 1:
                    #     doc2 = tokenizer.tokenize(generated_text)

                docs.append(doc1)
                # if x == 1:
                #     docs.append(doc2)
                labels.append(label1)
                # if x == 1:
                    # labels.append(label2)
                # generated_tones.append(generated_tone)
            rf.close()
        # 使用 Pool 并行处理文档
        with Pool(threads, initializer=SanText_plus_init,
                  initargs=(prob_matrix, word2id, sword2id, words, args.p, tokenizer)) as p:
            annotate_ = partial(SanText_plus)
            results = list(
                tqdm(p.imap(annotate_, docs, chunksize=32), total=len(docs), desc="Sanitize docs using SanText"))
            p.close()

        logger.info("Saving ...")
        total_size = len(results)
        train_size = math.floor(total_size * 0.8)
        print(len(results))
        # 保存处理后的结果train
        with open(output_file, 'w', encoding='utf-8') as out_file:
            processed_data = []
            # print(results)
            out_file.write("text\tlabel\n")
            for i, predicted_text in enumerate(results[:train_size]):
                # entry = {
                #     "origin_text": predicted_text,
                #     # "generated_text": predicted_text,
                #     # "generated_tone": generated_tones[i],
                #     "label": labels[i]
                # }
                text = predicted_text.replace("\n", " ")
                if labels[i] == "fake":
                    label = 0
                else:
                    label = 1
                # processed_data.append(entry)
                out_file.write(f"{text}\t{label}\n")
            # json.dump(processed_data, out_file, indent=4, ensure_ascii=False)
        out_file.close()
        # 保存处理后的结果dev
        with open(output_file_dev, 'w', encoding='utf-8') as out_file_dev:
            out_file_dev.write("origin_text\tlabel\n")
            for i, predicted_text in enumerate(results[train_size:]):
                origin_text = predicted_text.replace("\n", " ")
                # label = labels[i]
                if labels[i] == "fake":
                    label = 0
                else:
                    label = 1
                out_file_dev.write(f"{origin_text}\t{label}\n")
        out_file_dev.close()
    else:
        for file_name in ['train.tsv', 'dev.tsv']:
            data_file = os.path.join(args.data_dir, file_name)
            out_file = open(os.path.join(args.output_dir, file_name), 'w')
            logger.info("Processing file: %s. Will write to: %s" % (data_file, os.path.join(args.output_dir, file_name)))

            num_lines = sum(1 for _ in open(data_file))
            with open(data_file, 'r') as rf:
                # header
                if args.task == "SST-2" or args.task == "CliniSTS" or args.task == "QNLI":
                    header = next(rf)
                    out_file.write(header)
                if args.task == "QQP":
                    header = next(rf)
                    out_file.write("id\tquestion1\tquestion2\tis_duplicate\n")
                labels = []
                docs = []
                if args.task == "SST-2":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text = content[0]
                        label = int(content[1])
                        if args.embedding_type == "glove":
                            doc = [token.text for token in tokenizer(text)]
                        else:
                            doc = tokenizer.tokenize(text)
                        docs.append(doc)
                        labels.append(label)
                elif args.task == "CliniSTS":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text1 = content[7]
                        text2 = content[8]
                        label = content[-1]
                        if args.embedding_type == "glove":
                            doc1 = [token.text for token in tokenizer(text1)]
                            doc2 = [token.text for token in tokenizer(text2)]
                        else:
                            doc1 = tokenizer.tokenize(text1)
                            doc2 = tokenizer.tokenize(text2)
                        docs.append(doc1)
                        docs.append(doc2)
                        labels.append(label)
                elif args.task == "QNLI":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text1 = content[1]
                        text2 = content[2]
                        label = content[-1]
                        if args.embedding_type == "glove":
                            doc1 = [token.text for token in tokenizer(text1)]
                            doc2 = [token.text for token in tokenizer(text2)]
                        else:
                            doc1 = tokenizer.tokenize(text1)
                            doc2 = tokenizer.tokenize(text2)

                        docs.append(doc1)
                        docs.append(doc2)
                        labels.append(label)
                elif args.task == "CoLA":
                    for line in tqdm(rf, total=num_lines):
                        content = line.strip().split("\t")
                        text = content[3]
                        label = int(content[1])
                        if args.embedding_type == "glove":
                            doc = [token.text for token in tokenizer(text)]
                        else:
                            doc = tokenizer.tokenize(text)
                        docs.append(doc)
                        labels.append(label)
                elif args.task == "QQP":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text1 = content[3]
                        text2 = content[4]
                        label = content[-1]
                        if args.embedding_type == "glove":
                            doc1 = [token.text for token in tokenizer(text1)]
                            doc2 = [token.text for token in tokenizer(text2)]
                        else:
                            doc1 = tokenizer.tokenize(text1)
                            doc2 = tokenizer.tokenize(text2)

                        docs.append(doc1)
                        docs.append(doc2)
                        labels.append(label)

                rf.close()

            with Pool(threads, initializer=SanText_plus_init,
                      initargs=(prob_matrix, word2id, sword2id, words, args.p, tokenizer)) as p:
                annotate_ = partial(
                    SanText_plus,
                )
                results = list(
                    tqdm(
                        p.imap(annotate_, docs, chunksize=32),
                        total=len(docs),
                        desc="Sanitize docs using SanText",
                    )
                )
                p.close()

            logger.info("Saving ...")

            if args.task == "SST-2":
                for i, predicted_text in enumerate(results):
                    write_content = predicted_text + "\t" + str(labels[i]) + "\n"
                    out_file.write(write_content)
            elif args.task == "CliniSTS":
                assert len(results) / 2 == len(labels)
                for i in range(len(labels)):
                    predicted_text1 = results[i * 2]
                    predicted_text2 = results[i * 2 + 1]
                    write_content = str(i) + "\t" + "none\t" * 6 + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                        labels[i]) + "\n"
                    out_file.write(write_content)
            elif args.task == "QNLI":
                assert len(results) / 2 == len(labels)
                for i in range(len(labels)):
                    predicted_text1 = results[i * 2]
                    predicted_text2 = results[i * 2 + 1]
                    write_content = str(i) + "\t" + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                        labels[i]) + "\n"
                    out_file.write(write_content)
            elif args.task == "CoLA":
                for i, predicted_text in enumerate(results):
                    write_content = predicted_text + "\t" + str(labels[i]) + "\n"
                    out_file.write(write_content)
            elif args.task == "QQP":
                assert len(results) / 2 == len(labels)
                for i in range(len(labels)):
                    predicted_text1 = results[i * 2]
                    predicted_text2 = results[i * 2 + 1]
                    write_content = str(i) + "\t" + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                        labels[i]) + "\n"
                    out_file.write(write_content)

            out_file.close()


if __name__ == "__main__":
    main()
