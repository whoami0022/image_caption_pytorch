import torch
import torch.nn as nn
from torch.autograd import variable
import vocab


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, weights):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

        # self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        weights = torch.Tensor(weights)
        print(weights.size())
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, caption):
        seq_length = len(caption) + 1
        embeds = self.word_embeddings(caption)

        embeds = torch.cat((features, embeds), 0)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        out = self.linear(lstm_out.view(seq_length, -1))
        return out

    def greedy(self, cnn_out, seq_len=20):
        ip = cnn_out
        hidden = None
        ids_list = []
        for t in range(seq_len):
            lstm_out, hidden = self.lstm(ip.unsqueeze(1), hidden)
            # generating single word at a time
            linear_out = self.linear(lstm_out.squeeze(1))
            word_caption = linear_out.max(dim=1)[1]
            ids_list.append(word_caption)
            ip = self.word_embeddings(word_caption)
        return ids_list

