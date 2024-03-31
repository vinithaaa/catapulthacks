#from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim
from nltk.data import find
import numpy as np
import nltk
nltk.download('word2vec_sample')

class Similarity:

    def __init__(self):
        #self.model = Word2Vec.load('brown.embedding')
        self.model = gensim.models.KeyedVectors.load_word2vec_format(str(find('models/word2vec_sample/pruned.word2vec.txt')), binary=False)
        self.dataset = np.load('vec_rep.npy', allow_pickle=True).item()
        self.weight = 3

    def calc_sim(self, title, input_words):
        title += ' '
        title *= self.weight
        #print(title)
        #print(title + ' ' + input_words)
        word_list = gensim.utils.simple_preprocess(title + ' ' + input_words)
        bools = [True if word in self.model else False for word in word_list]
        #print(bools)
        word_vecs = np.array([self.model[word] for word in word_list if word in self.model])
        #print(word_vecs)
        avg_vec = np.sum(word_vecs, axis=0) / sum(bools)

        scores = {}
        for key in self.dataset.keys():
            vec = self.dataset.get(key)
            #print(vec)
            #print(avg_vec)
            scores[key] = cosine_similarity([avg_vec], [vec])[0][0]
            #scores[key] = self.model.wv.similarity([avg_vec], [vec])
            #print(scores[key])
            # print(key)
            # print("cosine similarity", cosine_similarity(["statistic data science"], [vec])[0][0])
        #scores_s = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        scores_s = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        #return (scores_s[0], scores_s[1], scores_s[2])
        return scores_s
# print(Similarity().calc_sim('statistic data science', 'data science'))



