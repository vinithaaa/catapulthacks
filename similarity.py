#from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim
import numpy as np

class Similarity:

    def __init__(self):
        self.model = Word2Vec.load('brown.embedding')
        self.dataset = np.load('vec_rep.npy', allow_pickle=True).item()


    def calc_sim(self, input_words):
        word_list = gensim.utils.simple_preprocess(input_words)
        bools = [True if word in self.model.wv else False for word in word_list]
        print(bools)
        word_vecs = np.array([self.model.wv[word] for word in word_list if word in self.model.wv])
        #print(word_vecs)
        avg_vec = np.sum(word_vecs, axis=0) / sum(bools)

        scores = {}
        for key in self.dataset.keys():
            vec = self.dataset.get(key)
            #print(vec)
            #print(avg_vec)
            scores[key] = cosine_similarity([avg_vec], [vec])[0][0]
            #print(scores[key])
            
        #scores_s = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        scores_s = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        #return (scores_s[0], scores_s[1], scores_s[2])
        return scores_s

        



sim = Similarity()
print(sim.calc_sim('earnings ($million) Sport'))

