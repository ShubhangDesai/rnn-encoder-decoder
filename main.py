from LanguageLoader import *
from RNN import *

en_path = 'data/en.zip'
fr_path = 'data/fr.zip'

max_length = 20
num_epochs = 1000
num_batches = 750
batch_size = 100
vocab_size = 15000

def main():
    data = LanguageLoader(en_path, fr_path, vocab_size, max_length)
    rnn = RNN(data.input_size, data.output_size)

    losses = []
    for epoch in range(num_epochs):
        print("=" * 50 + ("  EPOCH %i  " % epoch) + "=" * 50)
        for i, batch in enumerate(data.sentences(batch_size * num_batches, batch_size)):
            input, target = batch

            loss, outputs = rnn.train(Variable(torch.from_numpy(input).long()), Variable(torch.from_numpy(target).long()))
            losses.append(loss)

            if i % 100 is 0:
                print("Loss at step %d: %.2f" % (i, loss))
                print("Truth: \"%s\"" % data.vec_to_sentence(target))
                print("Guess: \"%s\"\n" % data.vec_to_sentence(outputs))
                rnn.save()

def translate():
    data = LanguageLoader(en_path, fr_path, vocab_size, max_length)
    rnn = RNN(data.input_size, data.output_size)

    vecs = data.sentence_to_vec("the president is here <EOS>")

    translation = rnn.eval(vecs)
    print(data.vec_to_sentence(translation))

main()
#translate()