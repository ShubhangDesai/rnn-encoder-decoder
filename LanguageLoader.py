from collections import Counter
import numpy as np
import torch
import pickle

from utils import read_data

class LanguageLoader(object):
    def __init__(self, input_path, output_path, vocab_size, max_length):
        super(LanguageLoader, self).__init__()

        self.vocab_size, self.max_length = vocab_size, max_length

        try:
            self.input_dict = pickle.load(open("data/input_dict.p", "rb"))
            self.input_vecs = pickle.load(open("data/input_vecs.p", "rb"))
            self.input_size = len(self.input_dict)

            self.output_dict = pickle.load(open("data/output_dict.p", "rb"))
            self.output_vecs = pickle.load(open("data/output_vecs.p", "rb"))
            self.output_size = len(self.output_dict)
            print("Languages found and loaded.")
        except(IOError):
            self.input_dict, self.input_vecs, self.input_size = self.init_language(input_path)
            pickle.dump(self.input_dict, open("data/input_dict.p", "wb"))
            pickle.dump(self.input_vecs, open("data/input_vecs.p", "wb"))
            print("Input language loaded.")

            self.output_dict, self.output_vecs, self.output_size = self.init_language(output_path)
            pickle.dump(self.output_dict, open("data/output_dict.p", "wb"))
            pickle.dump(self.output_vecs, open("data/output_vecs.p", "wb"))
            print("Output language loaded.")

        self.input_vecs, self.output_vecs = self.filter(self.input_vecs, self.output_vecs)
        self.pad, self.sos, self.eos = np.zeros((1, 1)) + 3, np.zeros((1, 1)), np.zeros((1, 1)) + 1

    def init_language(self, path):
        dictionary = ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]

        corpus = read_data(path)
        words = " ".join(corpus).split()
        mc = Counter(words).most_common(self.vocab_size-3)
        dictionary.extend([word for word, _ in mc])
        vectors = [[self.vectorize(word, dictionary) for word in sentence.split()] for sentence in corpus]

        return dictionary, vectors, len(dictionary)

    def sentences(self, amount, batch_size=5):
        indeces = np.random.choice(len(self.input_vecs), amount)
        in_sentences, out_sentences = [self.input_vecs[i] for i in indeces], [[self.sos] + self.output_vecs[i] + [self.eos] for i in indeces]

        batches = []
        for i in range(0, amount, batch_size):
            in_batch, out_batch = in_sentences[i:i+batch_size], out_sentences[i:i+batch_size]
            max_in_len, max_out_len = max([len(sentence) for sentence in in_batch]), max([len(sentence) for sentence in out_batch])

            in_batch = np.array([sentence + [self.pad for _ in range(max_in_len - len(sentence))] for sentence in in_batch])
            in_batch = np.transpose(np.squeeze(in_batch, 3), (1, 0, 2))

            out_batch = np.array([sentence + [self.pad for _ in range(max_out_len - len(sentence))] for sentence in out_batch])
            out_batch = np.transpose(np.squeeze(out_batch, 3), (1, 0, 2))

            batches.append((in_batch, out_batch))

        return batches

    def sentence_to_vec(self, sentence):
        vectors = [self.vectorize(word, self.input_dict) for word in sentence.lower().split()]
        return vectors

    def vec_to_sentence(self, vectors, language='output'):
        dict = self.output_dict if language == 'output' else self.input_dict
        sentence = " ".join([dict[int(vec[0, 0])] for vec in vectors])
        return sentence

    def vectorize(self, word, list):
        #vec = torch.LongTensor(1, 1).zero_()
        vec = np.zeros((1, 1), dtype=np.int32)
        index = list.index("<UNK>") if word not in list else list.index(word)
        vec[0][0] = index
        return vec

    def filter(self, input_vecs, output_vecs):
        i = 0
        for _ in input_vecs:
            if len(input_vecs[i]) > self.max_length or len(output_vecs[i]) > self.max_length:
                input_vecs.pop(i)
                output_vecs.pop(i)
            else:
                i += 1

        return input_vecs, output_vecs