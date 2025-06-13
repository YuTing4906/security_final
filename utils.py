from tqdm import tqdm
import os
import unicodedata
from collections import Counter
import json
def word_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def get_vocab_SST2(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
                next(csvfile)
                for line in tqdm(csvfile,total=num_lines-1):
                    line=line.strip().split("\t")
                    text = line[0]
                    if tokenizer_type=="subword":
                        tokenized_text = tokenizer.tokenize(text)
                    elif tokenizer_type=="word":
                        tokenized_text = [token.text for token in tokenizer(text)]
                    for token in tokenized_text:
                        vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab

def get_vocab_CliniSTS(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
            next(csvfile)
            for line in tqdm(csvfile,total=num_lines-1):
                line = line.strip().split("\t")
                text = line[7] + " " + line[8]
                if tokenizer_type=="subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type=="word":
                    tokenized_text = [token.text for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab


def get_vocab_QNLI(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
            next(csvfile)
            for line in tqdm(csvfile,total=num_lines-1):
                line = line.strip().split("\t")
                text = line[1] + " " + line[2]
                if tokenizer_type=="subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type=="word":
                    tokenized_text = [token.text for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab


def get_vocab_QQP(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir, split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
            next(csvfile)
            for line in tqdm(csvfile,total=num_lines-1):
                line = line.strip().split("\t")
                text = line[3] + " " + line[4]
                if tokenizer_type=="subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type=="word":
                    tokenized_text = [token.text for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab


def get_vocab_CoLA(data_dir, tokenizer, tokenizer_type="subword"):
    vocab = Counter()
    for split in ['train', 'dev']:
        data_file_path = os.path.join(data_dir, split + ".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
            for line in tqdm(csvfile, total=num_lines):
                line = line.strip().split("\t")
                text = line[3]
                if tokenizer_type == "subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type == "word":
                    tokenized_text = [token.text for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token] += 1

    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token] += 1

    return vocab

def get_vocab_gossipcop(data_file_path, tokenizer, tokenizer_type="subword"):
    vocab = Counter()
    # file_path =
    input_file = os.path.join(data_file_path, 'gossipcop.json')
    # Read JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        num_lines = len(data)

        for key, value in tqdm(data.items(), total=num_lines):
            text = value['origin_text']

            if tokenizer_type == "subword":
                tokenized_text = tokenizer.tokenize(text)
            elif tokenizer_type == "word":
                tokenized_text = [token.text for token in tokenizer(text)]

            for token in tokenized_text:
                vocab[token] += 1

    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token] += 1

    return vocab

