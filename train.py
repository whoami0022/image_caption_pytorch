import os
import torch
import time
import pickle
import argparse
import torch.nn as nn
from network.rnn import RNN
import matplotlib.pyplot as plt
from vocab import Vocab
from network import vgg
from torchvision import transforms
from torch.autograd import Variable
from collections import Counter
import Flickr8k

glove_word_file = 'embedding/glove.6B.50d.txt'


threshold = 5
embedding_dim = 512
hidden_dim = 512
learning_rate = 1e-3

dataset = Flickr8k.Flickr8k(transform=transforms.Compose([transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

captions_dict = {}

for i, data in enumerate(dataloader, 0):
    _, caption, id = data
    captions_dict[id[0]] = caption

vocab = Vocab(captions_dict, threshold)
vocab_size = vocab.id

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
    print('dictionary dump')


# # Build models
encoder = vgg.vgg16()
decoder = RNN(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=1)

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
