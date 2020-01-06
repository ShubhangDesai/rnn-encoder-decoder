import torch
import torch.nn as nn
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size=500, hidden_size=1000):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, 1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(input.size(0), 1, -1)
        output, hidden_state = self.gru(embedded, hidden)
        return output, hidden_state

    def first_hidden(self):
        return Variable(torch.FloatTensor(1, 1, self.hidden_size).zero_())