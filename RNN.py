from modules.Encoder import *
from modules.Decoder import *

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch

class RNN(object):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        self.encoder = Encoder(input_size)
        self.decoder = Decoder(output_size)

        self.loss = nn.CrossEntropyLoss()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())

        sos, eos = torch.LongTensor(1, 1).zero_(), torch.LongTensor(1, 1).zero_()
        sos[0, 0], eos[0, 0] = 0, 1

        self.sos, self.eos = sos, eos


    def train(self, input, target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        hidden_state = self.encoder.first_hidden()

        # Encoder
        for ivec in input:
            _, hidden_state = self.encoder.forward(Variable(ivec), hidden_state)

        # Decoder
        target.insert(0, self.sos)
        target.append(self.eos)
        total_loss = 0
        for i in range(len(target) - 1):
            _, softmax, hidden_state = self.decoder.forward(target[i], hidden_state)
            total_loss += self.loss(softmax, Variable(target[i+1][0]))

        total_loss.backward()

        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        return total_loss

    def eval(self, input):
        hidden_state = self.encoder.first_hidden()

        # Encoder
        for ivec in input:
            _, hidden_state = self.encoder.forward(ivec, hidden_state)

        outputs = []
        output = self.sos
        # Decoder
        while output is not self.eos:
            output, _, hidden_state = self.decoder.forward(output, hidden_state)
            outputs += output

        return outputs