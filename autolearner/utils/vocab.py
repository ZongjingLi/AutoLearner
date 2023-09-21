import nltk
from nltk import WordNetLemmatizer, RegexpTokenizer, pos_tag

#nltk.data.path.append('/Users/melkor/Documents/dataset/nltk_data')
for module in ["punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f'tokenizers/{module}')
    except LookupError:
        nltk.download(module)
#nltk.download("wordnet")

class WordVocab:
    def __init__(self):
        self.words = {"<start>", "<end>", "<pad>", "<unk>"}
        self._lemmatize = WordNetLemmatizer().lemmatize
        self._tokenize = RegexpTokenizer(r'\w+').tokenize
        self.word2index = {}
        self.index2word = []

    def freeze(self):
        self.words = frozenset(sorted(self.words))
        self.word2index = {w: i for i, w in enumerate(sorted(self.words))}
        self.index2word = list(sorted(self.words))

    def update(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        for sentence in sentences:
            self.words.update([self._lemmatize(word.lower()) for word in self._tokenize(sentence)])

    @property
    def unk(self):
        return self.word2index["<unk>"]

    @property
    def start(self):
        return self.word2index["<start>"]

    @property
    def end(self):
        return self.word2index["<end>"]

    @property
    def pad(self):
        return self.word2index["<pad>"]

    @property
    def special_tokens(self):
        return [self.unk, self.start, self.end, self.pad]

    def __getitem__(self, word):
        assert len(self.word2index) > 0, "The vocab should be freezed."
        word = word.lower()
        if word == "unk":
            return self.unk
        if word == "pad":
            return self.pad
        word = self._lemmatize(word)
        assert word in self.word2index, f"Word \'{word}\' not found in vocabulary."
        return self.word2index[word]

    def __call__(self, sentence):
        return [self.start, *(self[word] for word in self._tokenize(sentence)), self.end]

    def is_noun(self, word):
        return pos_tag([word])[0][1] == "NN"

    def __len__(self):
        return len(self.words)

import torch
import torch.nn as nn

import numpy as np # linear algebra and arrays

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Yiqi Sun
# Email  : rintfd@163.com
# Date   : 6/03/2022
#
# This file is part of MecThuen.
# Unity, Percision, Perfection
# Distributed under terms of the MIT license.


def make_corpus(dataset):
    corpus = []
    for bind in dataset:
        corpus.append(bind[0])
    return corpus

SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


def tokenize(s, delim=' ',
            add_start_token=True, add_end_token=True,
            punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    token_to_count = {}
    tokenize_kwargs = {
        'delim': delim,
        'punct_to_keep': punct_to_keep,
        'punct_to_remove': punct_to_remove,
    }
    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs,
                                        add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def reverse_diction(dict):
    keys = dict.keys()
    out_dict = {}
    for key in keys:
        value = dict[key]
        out_dict[value] = key
    return out_dict