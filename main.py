from LanguageLoader import *
from RNN import *

en_path = 'data/en.zip'
fr_path = 'data/fr.zip'

max_length = 20
num_batches = 7500
vocab_size = 15000

def main():
    data = LanguageLoader(en_path, fr_path, vocab_size, max_length)
    rnn = RNN(data.input_size, data.output_size)

    losses = []
    iter = 0
    for input, target in data.sentences(num_batches):
        loss = rnn.train(input, target)
        if iter % 100 is 0:
            print(loss.data[0])
        iter += 1

main()