from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec
import gensim
import os
import pandas as pd

from nltk.test.gensim_fixt import setup_module
setup_module()

from nltk.data import find
import nltk
import numpy as np

vec_reps = {}



# Function to calculate Word2Vec representation
def calculate_word2vec(file_path, weight):
    # Read CSV file
    df = pd.read_csv(file_path)
    # Combine title and column names
    text = file_path.split('/')[-1].split('.')[0].replace('_', ' ')  # Extract title from file name
    title = text
    text *= weight
    text += ' '.join(df.columns)
    # Calculate average Word2Vec representation
    word_list = gensim.utils.simple_preprocess(text)
    word_vectors = [model.wv[word] for word in word_list if word in model.wv]
    avg_vector = sum(word_vectors) / (len(word_vectors) + weight)
    #similarity = util.cosine_similarity([avg_vector], [input_vector])[0][0]
    vec_reps[title] = avg_vector
    #return similarity

# Set path to the folder containing CSV files
folder_path = 'datasets/'

# Load Word2Vec model (you need to train or load a pre-trained model)
model = Word2Vec.load('brown.embedding')
# Set weight for title
weight = 1  # You can tune this value
#heap = []
# Iterate through all CSV files in the folder

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        calculate_word2vec(file_path, weight)
        #heap.append((similarity, file_name))
        #print(f"Word2Vec representation for {file_name}: {similarity}")


np.save('vec_rep.npy', vec_reps)


#heap.sort(reverse=True)
#print(heap)




# TODO add ranking percentage for match scores absed on similarity
# TODO see if you can get metadata without downloading
# TODO for downloading the dataset, display relevant info about the dataset and then let them choose
