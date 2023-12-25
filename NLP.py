from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import WordPunctTokenizer


class TextImprovement(object): # object is the list with two names of the files(text and the standart phrases)
    def __init__(self, object): 
        self.text_path = object[0]  #
        self.phrases_path = object[1]

    def compute(self, n): # n - vector size for the model W2V
        tokenizer = WordPunctTokenizer()
        data = [] # creating a new list for the data in file
        with open(self.text_path) as file:
            for line in file: 
                line = line.strip() #or some other preprocessing
                data.append(line)

        self.data_tok = [tokenizer.tokenize(x.lower()) for x in data]
        self.phrases = pd.read_csv(self.phrases_path)

        self.model = Word2Vec(self.data_tok, # building the model
                 vector_size=n,
                 window=5).wv

        a = pd.read_csv(self.phrases_path) # from csv file ...
        s = a.squeeze().values.tolist()   #  ... to list of phrases

        self.new = [] # string for the simularity score and the recommended replacement
        for i in range(len(s)):
            x = s[i].split()
            synonym = self.model.most_similar(positive=x)[0][0] 

            if self.model.most_similar(positive=x)[0][1] > 0.9: # threshold for defining synonyms(current 0.9)
                for i in range(len(data)): # loop through the array to find 
                    mas = data[i].split() # dividing each line into words
                    for j in range(len(mas)):
                        if mas[j] == synonym:
                            data[i] = data[i].replace(mas[j], synonym) 
                            x_sim, y_sim = self.model.get_vector(mas[j]), self.model.get_vector(synonym) #converting words to numeric vectors
                            cos_sim = dot(x_sim, y_sim)/(norm(x_sim)*norm(y_sim))  #calculating cosine similarity  
                            self.new.append([str(mas[j])  + "--" + "synonym:" + " " + synonym + "," + " " + "simularity:" + " " + str(cos_sim)])
        return data[50], self.new
