import numpy


# Initialize NLP parameters
class w2vFeatures:

    def __init__(self, word=None):
        self.word = word

    def get_vect(self, word, model):
        try:
            return model.wv[word]
        except KeyError:
            return numpy.zeros((model.vector_size,))

    def sum_vectors(self, phrase, model):
        return sum(self.get_vect(w, model) for w in phrase)

    def word2vec_features(self, X, model):
        feats = numpy.vstack([self.sum_vectors(p, model) for p in X])
        return feats


if __name__ == '__init__':
    pass
