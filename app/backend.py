import pandas as pd
import numpy as np
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import json

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary
from nltk.probability import FreqDist
from nltk.util import ngrams
from matplotlib import pyplot as plt
from collections import Counter
from wordcloud import WordCloud

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# %%
def text_filtering(text):
    text = text.lower()     # mengubah review menjadi lowercase
    text = re.sub(r'https?:\/\/\S+','',text)        # menghilangkan url
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ",text).split())        # menghilangkan mention & hashtag
    text = re.sub(r'(b\'{1,2})',"", text)       # menghilangkan karakter byte
    text = re.sub('[^a-zA-Z]', ' ', text)       # menghilangkan karakter bukan huruf
    text = re.sub(r'\d+', '', text)       # menghilangkan digit angka
    text = text.translate(str.maketrans("","",string.punctuation))      #menghilangkan tanda baca
    text = re.sub(r'\s+', ' ', text).strip()        # menghilangkan whitespace
    return text

# %%
def stopstem(text):
    with open("app/kamus_stopwords_v2.txt") as kamus:
        word = kamus.readlines()
        list_stopword = [line.replace('\n',"") for line in word]
    dictionary = ArrayDictionary(list_stopword)
    stopword = StopWordRemover(dictionary)
    text = stopword.remove(text)
    #stem
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    text = stemmer.stem(text)
    return text 

# %%
def word_tokenize_wrapper(text):
    return word_tokenize(text)

# %%
#pengubahan kata slang
kamusslang = pd.read_csv("app/kamus_slangwords.csv")
kata_pembakuan_dict = {}

for index, row in kamusslang.iterrows():
    if row[0] not in kata_pembakuan_dict:
        kata_pembakuan_dict[row[0]] = row[1] 


# %%
def pembakuan_kata(document):
    return [kata_pembakuan_dict[term] if term in kata_pembakuan_dict else term for term in document]

# %%
#penggabungan token pembakuan menjadi kalimat
def join_token(document):
    for i in range(len(document)):
        document[i] = ' '.join(document[i])
    return document

# %%
#penghapusan stopwords akhir
def stop(text):
    with open("app/kamus_stopwords_v2.txt") as kamus:
        word = kamus.readlines()
        list_stopword = [line.replace('\n',"") for line in word]
    dictionary = ArrayDictionary(list_stopword)
    stopword = StopWordRemover(dictionary)
    text = stopword.remove(text)
    return text

# %%
def temptoken(document):
    return document.apply(lambda x: word_tokenize(str(x)))

# %%
#Perhitungan Frekuensi Kata
def hitung_kata(df,column='temp_token', preprocess=None, min_freq=2):
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)
    counter = Counter()
    df[column].map(update)
    freq_df = pd.DataFrame.from_dict(counter, orient= 'index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'
    return freq_df.sort_values('freq', ascending=False)

# %%
def wordcloud(doc):
    text = ' '.join(doc.tolist())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title("Wordcloud")
    plt.axis('off')
    plt.show()

# %%
def ngrams(tokens, n=2, sep=' ', stopwords=set()):
    return [sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)]) if len([t for t in ngram if t in stopwords])==0]

# %%
def convertdf_tocsv(df):
    return df.to_csv().encode('utf-8')

# %%
def vectorizzer(df):
    tfidf = TfidfVectorizer(min_df= 10, ngram_range=(1,1))
    x_tf = tfidf.transform(df.apply(lambda x: np.str(x)))
    return x_tf


