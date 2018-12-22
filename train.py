import os
import torch
import time
import pickle
import nltk
import argparse
import numpy as np
import torch.nn as nn
from network.rnn import RNN
import matplotlib.pyplot as plt
from vocab import Vocab
from network import vgg
from torchvision import transforms
from torch.autograd import Variable
from collections import Counter
import Flickr8k
from torchtext import data
from torchtext.vocab import Vectors, GloVe
import spacy
spacy_en = spacy.load('en')


def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


glove_word_file = 'embedding/glove.6B.50d.txt'

TEXT = data.Field(sequential=True)

vectors = Vectors(name=glove_word_file)

threshold = 5
embedding_dim = 50
hidden_dim = 30
learning_rate = 1e-3

dataset = Flickr8k.Flickr8k(transform=transforms.Compose([transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

captions_dict = {}

caption_vocab = []
for i, data in enumerate(dataloader, 0):
    _, caption, id = data
    captions_dict[id[0]] = caption
    caption_vocab.append(caption[0])

f = open('embedding/glove.6B.50d.txt', 'r')
data = [line.strip('\n').split() for line in f.readlines()]

vocab = Vocab(captions_dict, threshold)
vocab_size = vocab.id

embeddings = np.zeros([vocab_size, embedding_dim])
for k in data:
    if k[0] in vocab.word2id:
        embeddings[vocab.word2id[k[0]]] = list(map(float, k[1:]))

weights = embeddings
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
    print('dictionary dump')


# # Build models
encoder = vgg.vgg16()
decoder = RNN(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=1, weights=weights)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

# Train models
num_epochs = 100
save_iter = 10
for epoch in range(num_epochs):

    loss_list = []
    for _, data in enumerate(dataloader, 0):
        image, captions, id = data

        for caption in captions:
            image = torch.Tensor(image)
            caption = torch.LongTensor(vocab.caption2ids(caption[0]))
            caption_wo_end = caption[:-1]

            encoder.zero_grad()
            decoder.zero_grad()

            cnn_out = encoder(image)
            lstm_out = decoder(cnn_out, caption_wo_end)

            loss = criterion(lstm_out, caption)
            loss.backward()
            optimizer.step()
            loss_list.append(loss)

    avg_loss = torch.mean(torch.Tensor(loss_list))
    print('epoch %d avg_loss %f' % (epoch, avg_loss))
    if epoch % 10 == 0:
        torch.save(encoder.state_dict(), 'model/iter_%d_cnn.pkl' % epoch)
        torch.save(decoder.state_dict(), 'model/iter_%d_lstm.pkl' % epoch)
