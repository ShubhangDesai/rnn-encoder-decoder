from modules.Encoder import *
from modules.Decoder import *

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch

import numpy as np

class RNN(object):
    def __init__(self, input_size, output_size, resume=False):
        super(RNN, self).__init__()

        self.encoder = Encoder(input_size)
        self.decoder = Decoder(output_size)

        self.loss = nn.CrossEntropyLoss()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())

        if resume:
            self.encoder.load_state_dict(torch.load("models/encoder.ckpt"))
            self.decoder.load_state_dict(torch.load("models/decoder.ckpt"))


    def train(self, input, target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encoder
        hidden_state = self.encoder.first_hidden()
        for ivec in input:
            _, hidden_state = self.encoder.forward(ivec, hidden_state)

        # Decoder
        total_loss, outputs = 0, []
        for i in range(len(target) - 1):
            _, softmax, hidden_state = self.decoder.forward(target[i], hidden_state)

            outputs.append(np.argmax(softmax.data.numpy(), 1)[:, np.newaxis])
            total_loss += self.loss(softmax, target[i+1].squeeze(1))

        total_loss /= len(outputs)
        total_loss.backward()

        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        return total_loss.data[0], outputs

    def eval(self, input):
        hidden_state = self.encoder.first_hidden()

        # Encoder
        for ivec in input:
            _, hidden_state = self.encoder.forward(Variable(ivec), hidden_state)

        sentence = []
        input = self.sos
        # Decoder
        while input.data[0, 0] != 1:
            output, _, hidden_state = self.decoder.forward(input, hidden_state)
            word = np.argmax(output.data.numpy()).reshape((1, 1))
            input = Variable(torch.LongTensor(word))
            sentence.append(word)

        return sentence

    def save(self):
        torch.save(self.encoder.state_dict(), "models/encoder.ckpt")
        torch.save(self.decoder.state_dict(), "models/decoder.ckpt")