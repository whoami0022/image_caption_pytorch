import os
import nltk
import pickle
import json
from collections import Counter

class Vocab():
    def __init__(self, captions_dict, threshold):
        self.word2id = {}
        self.id2word = {}
        self.id = 0
        self.build(captions_dict, threshold)


    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.id
            self.id2word[self.id] = word
            self.id += 1

    def get_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id['<unk>']

    def get_word(self, id):
        return self.id2word[id]

    def build(self, captions_dict, threshold):
        counter = Counter()
        tokens = []

        for k, captions in captions_dict.items():
            for caption in captions:
                caption = caption[0]
                tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
        counter.update(tokens)

        words = []
        for word, count in counter.items():
            if count >= threshold:
                words.append(word)

        for word in words:
            self.add_word(word)

        self.add_word('<unk>')
        self.add_word('<start>')
        self.add_word('<end>')

    def caption2ids(self, caption):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        vec = []
        vec.append(self.get_id('<start>'))
        vec.extend([self.get_id(word) for word in tokens])
        vec.append(self.get_id('<end>'))
        return vec

