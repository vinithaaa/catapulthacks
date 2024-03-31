from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec
import gensim
import numpy as np

class Similarity:

    def __init__(self):
        self.model = Word2Vec.load('brown.embedding')
        self.dataset = np.load('vec_rep.npy').item()


    def calc_sim(self, input_words):
        word_list = gensim.utils.simple_preprocess(input_words)
        word_vecs = [self.model.wv[word] for word in word_list if word in model.wv]
        avg_vec = sum(word_vecs) / len(word_vecs)

        scores = {}
        for key in self.dataset.keys():
            vec = self.dataset.get(key)
            scores[key] = util.cosine_similarity([avg_vec], [vec])[0][0]
            
        scores_s = ((k,v) for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True))
        return (scores_s[0], scores_s[1], scores_s[2])    

        





