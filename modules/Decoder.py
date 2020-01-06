import torch.nn as nn
from torch.autograd import Variable

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size=500, hidden_size=1000):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden_state = self.gru(embedded, hidden)
        output = output.view(output.size(0), output.size(2))
        linear = self.linear(output)
        softmax = self.softmax(linear)
        return output, softmax, hidden_state
