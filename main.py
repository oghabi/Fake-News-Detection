import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os


import collections
import operator
from collections import defaultdict

import re
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import LatentDirichletAllocation
import xgboost as xgb

from keras.utils import np_utils

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping




pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.height', 1000)
pd.set_option('display.width', 5000)


# To keep randomness the same
np.random.seed(0)


df_train_text = pd.read_csv('./train_bodies.csv', encoding="ISO-8859-1")  # Shape is (1683, 2)
df_train_stances = pd.read_csv('./train_stances.csv', encoding="ISO-8859-1") # Shape is (49972, 3)

df_test_text = pd.read_csv('./competition_test_bodies.csv', encoding="ISO-8859-1") 
df_test_stances = pd.read_csv('./competition_test_stances.csv', encoding="ISO-8859-1")



stemmer = nltk.stem.SnowballStemmer('english')
WORD_RE = re.compile(r"(?u)\b\w\w+\b", flags = re.UNICODE)
stopwords = nltk.corpus.stopwords.words('english')

# Apply this to both the news headline and the article body
def preprocess(text, stemming=True, stopword_removal=True):
	# Tokenize & Clean the strings, some words are like "Londonâ€™s"   (remove non-alphanumeric using Regex)
	# Remove non-ascii characters
	tokens = [t.encode("ascii", errors="ignore").decode().lower() for t in WORD_RE.findall(text)]
	# Stem the tokens
	if stemming:
		tokens = [stemmer.stem(t) for t in tokens]
	# Remove Stopwords
	if stopword_removal:
		tokens = [t for t in tokens if t not in stopwords]
	return tokens


# Pre-process, stem, and remove stopwords
df_train_stances['Headline_stemmed'] = df_train_stances['Headline'].apply( preprocess )	
df_train_text['articleBody_stemmed'] = df_train_text['articleBody'].apply( preprocess )
df_test_stances['Headline_stemmed'] = df_test_stances['Headline'].apply( preprocess )	
df_test_text['articleBody_stemmed'] = df_test_text['articleBody'].apply( preprocess )


# Pre-process but don't do stemming (for Word2Vec)
df_train_stances['Headline_nonstemmed'] = df_train_stances['Headline'].apply( lambda x: preprocess(x, stemming=False) )	
df_train_text['articleBody_nonstemmed'] = df_train_text['articleBody'].apply( lambda x: preprocess(x, stemming=False) )
df_test_stances['Headline_nonstemmed'] = df_test_stances['Headline'].apply( lambda x: preprocess(x, stemming=False)  )	
df_test_text['articleBody_nonstemmed'] = df_test_text['articleBody'].apply( lambda x: preprocess(x, stemming=False)  )


# Pre-process for Word2Vec pre-trained on Google News (remove words that don't have a pre-trained embedding)
word_vectors = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 
df_train_stances['Headline_Word2Vec_Pretrained'] = df_train_stances['Headline_nonstemmed'].apply( lambda x: [t if t in word_vectors.wv.vocab else '<_UNK_>' for t in x ] )	
df_train_text['articleBody_Word2Vec_Pretrained'] = df_train_text['articleBody_nonstemmed'].apply( lambda x: [t if t in word_vectors.wv.vocab else '<_UNK_>' for t in x ] )
df_test_stances['Headline_Word2Vec_Pretrained'] = df_test_stances['Headline_nonstemmed'].apply( lambda x: [t if t in word_vectors.wv.vocab else '<_UNK_>' for t in x ]  )	
df_test_text['articleBody_Word2Vec_Pretrained'] = df_test_text['articleBody_nonstemmed'].apply( lambda x: [t if t in word_vectors.wv.vocab else '<_UNK_>' for t in x ]   )



# Natural join to have all in one dataframe
df_train = df_train_stances.join(df_train_text.set_index('Body ID'), on='Body ID')
df_test = df_test_stances.join(df_test_text.set_index('Body ID'), on='Body ID')


def labelling(x):
	if x == 'unrelated':
		return 0
	elif x == 'discuss':
		return 1
	elif x == 'agree':
		return 2
	elif x == 'disagree':
		return 3


# Apply numeric labels for machine learning models
df_train['stance_labels'] = df_train['Stance'].apply( lambda x: labelling(x)  )
df_test['stance_labels'] = df_test['Stance'].apply( lambda x: labelling(x) )


num_train = df_train.shape[0]
df_train_stats = df_train['Stance'].value_counts()/num_train

# Statistics of 4 classes
# unrelated    73.130953
# discuss      17.827984
# agree         7.360122
# disagree      1.680941


# Split train into train and cv (but keep imbalanced statistics)
# Pick from:
# unrelated 73.3% chance
# discuss   17.8% chance
# agree     7.36% chance
# disagree  1.68

labels = list(df_train_stats.index)
probs = list(df_train_stats.values)

# 10% of training data
num_cv = int(num_train * 0.1)

# Initialize df_cv
df_cv = pd.DataFrame( index=range(0,num_cv), columns=list(df_train.columns) )  # Shape is (4997, 4)

for i in range(0, num_cv):
	selected_label = np.random.choice(labels, p=probs)
	# Now for that label, uniformly pick a sample from the training data
	# df_train.groupby(['Stance']).groups returns a dictionary with df indexes of rows belonging to the 4 labels
	sample_indices = df_train.groupby(['Stance']).groups[ selected_label ]
	selected_index = np.random.choice(list(sample_indices), p=None) # Uniform sampling
	# Add sample to df_cv
	df_cv.iloc[ i ] = df_train.iloc[ selected_index  ]
	# Remove sample from df_train (cuz its now part of df_cv)
	df_train.drop( selected_index, inplace=True )
	# Reset the index to start from 0
	df_train = df_train.reset_index(drop=True)


# Make sure df_cv has the same ratio of the 4 classes
df_cv_stats = df_cv['Stance'].value_counts()/num_c    v


# This is similar to document collection
Articles = pd.read_csv('./train_bodies.csv', encoding="ISO-8859-1") 
Articles['articleBody_stemmed'] = Articles['articleBody'].apply( preprocess )  
Articles['articleBody_nonstemmed'] = Articles['articleBody'].apply( lambda x: preprocess(x, stemming=False) )  


# Build the term frequency vocab (from stemmed and removed stopwords)
tf_vocab_stemmed = collections.defaultdict(float)

# Build inverted index for IDF
inverted_index = collections.defaultdict(set)

for index,article in enumerate(np.array(Articles['articleBody_stemmed'])):
	for word in article:
		tf_vocab_stemmed[ word ] += 1.0
		inverted_index[ word ].add(index)  # Add the index of the article this word appeared in



# Build the term frequency vocab (from non-stemmed and removed stopwords)
tf_vocab_nonstemmed = collections.defaultdict(float)

for article in np.array(Articles['articleBody_nonstemmed']):
	for word in article:
		tf_vocab_nonstemmed[ word ] += 1.0


stemmed_vocab_words = list(tf_vocab_stemmed.keys())  # 15639
num_words_in_collection = sum(tf_vocab_stemmed.values())
num_vocab = len(stemmed_vocab_words)
nonstemmed_vocab_words = list(tf_vocab_nonstemmed.keys())  # 22950



# Cosine Similarity
def cosine_sim(a,b):
	# Calculate inner product between rows of arrays a and b
	numer = np.sum(a*b, axis=1)
	deno = np.sqrt(np.sum(a**2, axis=1)) * np.sqrt(np.sum(b**2, axis=1))
	return numer/deno

############################## cosine similarity - Manually ################################

# Vector representation of the headlines & articles
df_train_headlines_BoW = pd.DataFrame(0, index=range(0,df_train.shape[0]), columns=stemmed_vocab_words ) 
df_train_articles_BoW = pd.DataFrame(0, index=range(0,df_train.shape[0]), columns=stemmed_vocab_words ) 
df_cv_headlines_BoW = pd.DataFrame(0, index=range(0,df_cv.shape[0]), columns=stemmed_vocab_words ) 
df_cv_articles_BoW = pd.DataFrame(0, index=range(0,df_cv.shape[0]), columns=stemmed_vocab_words ) 
df_test_headlines_BoW = pd.DataFrame(0, index=range(0,df_test.shape[0]), columns=stemmed_vocab_words ) 
df_test_articles_BoW = pd.DataFrame(0, index=range(0,df_test.shape[0]), columns=stemmed_vocab_words ) 

# For the headlines and for test set articles and bodies, check if the word exists in the stemmed_vocab_words cuz these words
# Weren't included when building the dictionary

for i in range(0, df_train.shape[0]):
	df_train_headlines_BoW.iloc[i][ [w for w in df_train.iloc[i]['Headline_stemmed'] if w in stemmed_vocab_words ] ] = 1
	df_train_articles_BoW.iloc[i][ df_train.iloc[i]['articleBody_stemmed'] ] = 1
	
	# If you want term frequency counts instead of binary (1,0): use below
	# for wrd in df_train.iloc[i]['articleBody_stemmed']:
	# 	df_cv_articles_BoW.iloc[1][ wrd ] += 1

for i in range(0, df_cv.shape[0]):
	df_cv_headlines_BoW.iloc[i][ [w for w in df_cv.iloc[i]['Headline_stemmed'] if w in stemmed_vocab_words ] ] = 1
	df_cv_articles_BoW.iloc[i][ df_cv.iloc[i]['articleBody_stemmed'] ] = 1

for i in range(0, df_test.shape[0]):
	df_test_headlines_BoW.iloc[i][ [w for w in df_test.iloc[i]['Headline_stemmed'] if w in stemmed_vocab_words ] ] = 1
	df_test_articles_BoW.iloc[i][ [w for w in df_test.iloc[i]['articleBody_stemmed'] if w in stemmed_vocab_words ] ] = 1


train_cosine_sim = cosine_sim( np.array(df_train_headlines_BoW) ,  np.array(df_train_articles_BoW) )
cv_cosine_sim = cosine_sim( np.array(df_cv_headlines_BoW) ,  np.array(df_cv_articles_BoW) )
test_cosine_sim = cosine_sim( np.array(df_test_headlines_BoW) ,  np.array(df_test_articles_BoW) )

##############################################################################################



############################## cosine similarity - Library ################################

articles_sklearn = [ ' '.join( Articles['articleBody_stemmed'][i] ) for i in range(0, Articles.shape[0]) ]

# If binary is False, then it uses the Term Frequencies as the count
vectorizer = CountVectorizer(binary=False)
# Learn the vocabulary dictionary and return term-document matrix
articles_BoW_vectors = vectorizer.fit_transform(articles_sklearn)
articles_BoW_vectors.toarray()  # (1683, 15619)
# vectorizer.get_feature_names()

# Transform train set
train_articles_sklearn = [ ' '.join( df_train['articleBody_stemmed'][i] ) for i in range(0, df_train.shape[0]) ]
train_headlines_sklearn = [ ' '.join( df_train['Headline_stemmed'][i] ) for i in range(0, df_train.shape[0]) ]
train_articles_BoW =  vectorizer.transform(train_articles_sklearn).toarray()
train_headlines_BoW =  vectorizer.transform(train_headlines_sklearn).toarray()

# Transform cv set
cv_articles_sklearn = [ ' '.join( df_cv['articleBody_stemmed'][i] ) for i in range(0, df_cv.shape[0]) ]
cv_headlines_sklearn = [ ' '.join( df_cv['Headline_stemmed'][i] ) for i in range(0, df_cv.shape[0]) ]
cv_articles_BoW =  vectorizer.transform(cv_articles_sklearn).toarray()
cv_headlines_BoW =  vectorizer.transform(cv_headlines_sklearn).toarray()

# Transform test set
test_articles_sklearn = [ ' '.join( df_test['articleBody_stemmed'][i] ) for i in range(0, df_test.shape[0]) ]
test_headlines_sklearn = [ ' '.join( df_test['Headline_stemmed'][i] ) for i in range(0, df_test.shape[0]) ]
test_articles_BoW =  vectorizer.transform(test_articles_sklearn).toarray()
test_headlines_BoW =  vectorizer.transform(test_headlines_sklearn).toarray()


train_cosine_sim = cosine_sim(train_headlines_BoW, train_articles_BoW)
cv_cosine_sim = cosine_sim(cv_headlines_BoW, cv_articles_BoW)
test_cosine_sim = cosine_sim(test_headlines_BoW, test_articles_BoW)


# Replace inf with zero (inf occurs cuz of division by zero),  i!=i is for NaN
train_cosine_sim = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in train_cosine_sim])
cv_cosine_sim = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in cv_cosine_sim])
test_cosine_sim = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in test_cosine_sim])

##############################################################################################


############## cosine similarity - Word2Vec (Using pre-trained model on Google News) ###############

# pre-trained Google News corpus (3 billion running words) word vector model (3 million 300-dimension English word vectors)
word_vectors = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  # C binary format

# 3,000,000 words
word_vectors.wv.vocab
word_vectors.wv.vocab.keys()

# Word embedding for <UNK> token
num_embedding_dim = 300
unk_embedding = np.array( [random.uniform(-1,1) for i in range(num_embedding_dim)] )

# multiply the word vectors (performs better than addition) for the headline and article body to get 2 300-d vectors to perform cosine similarity
train_headlines_Word2Vec = np.array( [  np.prod([ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_train['Headline_Word2Vec_Pretrained'][i]  ], axis=0)  for i in range(0, df_train.shape[0])  ]  )
train_articles_Word2Vec = np.array( [  np.prod([ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_train['articleBody_Word2Vec_Pretrained'][i]  ], axis=0)  for i in range(0, df_train.shape[0])  ]  )
cv_headlines_Word2Vec = np.array( [  np.prod([ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_cv['Headline_Word2Vec_Pretrained'][i]  ], axis=0)  for i in range(0, df_cv.shape[0])  ]  )
cv_articles_Word2Vec = np.array( [  np.prod([ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_cv['articleBody_Word2Vec_Pretrained'][i]  ], axis=0)  for i in range(0, df_cv.shape[0])  ]  )
test_headlines_Word2Vec = np.array( [  np.prod([ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_test['Headline_Word2Vec_Pretrained'][i]  ], axis=0)  for i in range(0, df_test.shape[0])  ]  )
test_articles_Word2Vec = np.array( [  np.prod([ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_test['articleBody_Word2Vec_Pretrained'][i]  ], axis=0)  for i in range(0, df_test.shape[0]) ] )


train_cosine_sim_word2vec = cosine_sim(train_headlines_Word2Vec, train_articles_Word2Vec)
cv_cosine_sim_word2vec  = cosine_sim(cv_headlines_Word2Vec, cv_articles_Word2Vec)
test_cosine_sim_word2vec  = cosine_sim(test_headlines_Word2Vec, test_articles_Word2Vec)

# Replace inf with zero (inf occurs cuz of division by zero),  i!=i is for NaN
train_cosine_sim_word2vec = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in train_cosine_sim_word2vec])
cv_cosine_sim_word2vec = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in cv_cosine_sim_word2vec])
test_cosine_sim_word2vec = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in test_cosine_sim_word2vec])

##############################################################################################



############## cosine similarity - Word2Vec (training it on training Articles) ###############

sentences = [['first', 'sentence'], ['second', 'sentence']]

# Parameters:
# min_count (default 5): prune internal dictionary, words that appear once or twice are probably typos/meaningless and not enough data to train the embeddings from it 
# size (default 100): # dimensions of word embedding
# output is matrix vocab x size
model = gensim.models.Word2Vec(sentences, min_count=1, size=300, workers=4)

print(model.wv.vocab)

model.save('/tmp/mymodel')
new_model = gensim.models.Word2Vec.load('/tmp/mymodel')  # You can continute training after loading (because the weights are saved)

model.wv['computer']  # numpy vector of a word
model['computer'] 
##############################################################################################



############## cosine similarity - Word2Vec (model pre-trained on Google News but train on corpus Articles as well) ###############

# This is not possible with Google News cuz we don't have the model weights and vocab
model = gensim.models.Word2Vec.load(temp_path)
more_sentences = [['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']]
model.build_vocab(more_sentences, update=True)
model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)

##############################################################################################



############################## TF-IDF  ############################################

# Scikit Learn uses Log_e (Total # docs/ # docs which term t occurs)
# For scikit learn, set smoothing to false and then subtract 1 from the result

# corpus = ["This is very strange","This is very nice",'man orange nice']
# vectorizer = TfidfVectorizer(min_df=1, smooth_idf=False)
# X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_
# print (dict(zip(vectorizer.get_feature_names(), idf)))


# Build IDF (# docs in which each vocab word appears in)
# Now for IDF, get the # docs each term in vocab appears in using len(indexes)
IDF_stemmed = collections.defaultdict(float)
num_articles = Articles.shape[0]
for w in list(tf_vocab_stemmed.keys()):
	#IDF is N/n_t  (# total docs/#docs in which t appears in)
	n_t = len( inverted_index[w] )
	IDF_stemmed[w] = np.log10( num_articles/float(n_t) )

sorted_IDF_stemmed = sorted(IDF_stemmed.items(), key=operator.itemgetter(1))


# tf * idf
train_tfidf = np.array( [  sum([ df_train['articleBody_stemmed'][i].count(t) * IDF_stemmed[t] for t in set(df_train['Headline_stemmed'][i]).intersection( df_train['articleBody_stemmed'][i] )  ])  for i in range(0, df_train.shape[0])  ] )
cv_tfidf = np.array( [  sum([ df_cv['articleBody_stemmed'][i].count(t) * IDF_stemmed[t] for t in set(df_cv['Headline_stemmed'][i]).intersection( df_cv['articleBody_stemmed'][i] )  ])  for i in range(0, df_cv.shape[0])  ] )
test_tfidf =  np.array( [  sum([ df_test['articleBody_stemmed'][i].count(t) * IDF_stemmed[t] for t in set(df_test['Headline_stemmed'][i]).intersection( df_test['articleBody_stemmed'][i] )  ])  for i in range(0, df_test.shape[0])  ] )

##############################################################################################


############################## SVD on TF-IDF  ############################################

# # Build IDF (# docs in which each vocab word appears in)
# # Now for IDF, get the # docs each term in vocab appears in using len(indexes)
IDF_stemmed = collections.defaultdict(float)
num_articles = Articles.shape[0]
for w in list(tf_vocab_stemmed.keys()):
	#IDF is N/n_t  (# total docs/#docs in which t appears in)
	n_t = len( inverted_index[w] )
	IDF_stemmed[w] = np.log10( num_articles/float(n_t) )

sorted_IDF_stemmed = sorted(IDF_stemmed.items(), key=operator.itemgetter(1))


train_tfidf_svd = np.array( [  [ df_train['articleBody_stemmed'][i].count(t) * IDF_stemmed[t] for t in set(df_train['Headline_stemmed'][i]).intersection( df_train['articleBody_stemmed'][i] )  ]  for i in range(0, df_train.shape[0])  ] )
cv_tfidf_svd = np.array( [  [ df_cv['articleBody_stemmed'][i].count(t) * IDF_stemmed[t] for t in set(df_cv['Headline_stemmed'][i]).intersection( df_cv['articleBody_stemmed'][i] )  ]  for i in range(0, df_cv.shape[0])  ] )
test_tfidf_svd =  np.array( [  [ df_test['articleBody_stemmed'][i].count(t) * IDF_stemmed[t] for t in set(df_test['Headline_stemmed'][i]).intersection( df_test['articleBody_stemmed'][i] )  ]  for i in range(0, df_test.shape[0])  ] )


# since we didn't sum the tf-idf for each (headline, article) pair, we need to re-size the array and pad with zeros
max_len = len(max(train_tfidf_svd, key=len)) # This finds the longest list in the numpy array
train_tfidf_svd = np.array( [i if len(i)==max_len else i+([0]*(max_len-len(i))) for i in train_tfidf_svd] )
cv_tfidf_svd = np.array( [i if len(i)==max_len else i+([0]*(max_len-len(i))) for i in cv_tfidf_svd] )
test_tfidf_svd = np.array( [i if len(i)==max_len else i+([0]*(max_len-len(i))) for i in test_tfidf_svd] )

# Apply SVD
svd = TruncatedSVD(n_components=3, n_iter=10, random_state=42)
train_tfidf_svd = svd.fit_transform( train_tfidf_svd )  
cv_tfidf_svd = svd.transform( cv_tfidf_svd )
test_tfidf_svd = svd.transform( test_tfidf_svd )
##############################################################################################


############################## Language Models & KL-Divergence  ############################################

# Build language model for each article (using article ID)
def create_article_lm(df):
	counts = collections.defaultdict(lambda: collections.defaultdict(float))
	for index, article in np.array( df[['Body ID','articleBody_stemmed']] ):
		for word in article:
			counts[ index ][ word ] += 1.0
		# Add total number of words in this article
		counts[index][ "__NumWordsArticle__" ] = len( article )
	return counts

	
train_cv_articles_lm = create_article_lm(Articles)

test_articles =  pd.read_csv('./competition_test_bodies.csv', encoding="ISO-8859-1") 
test_articles['articleBody_stemmed'] = test_articles['articleBody'].apply( preprocess )  
test_articles['articleBody_nonstemmed'] = test_articles['articleBody'].apply( lambda x: preprocess(x, stemming=False) )  

test_articles_lm = create_article_lm(test_articles)



# Build language model for each headline/query (using headline index in df_train, cv, or test)
def create_headline_lm(df):
	counts = collections.defaultdict(lambda: collections.defaultdict(float))
	for index, headline in enumerate(np.array(df['Headline_stemmed'])):
		for word in headline:
			counts[ index ][ word ] += 1.0
		# Add total number of words in this headline
		counts[index][ "__NumWordsHeadline__" ] = len( headline )
	return counts

train_headlines_lm = create_headline_lm(df_train)
cv_headlines_lm = create_headline_lm(df_cv)
test_headlines_lm = create_headline_lm(df_test)


# Article language model with dirichlet smoothing
def calc_article_lm(article_lm, index_, term_, smoothing='no'):
	N = article_lm[index_]["__NumWordsArticle__"] 
	mu = 2000
	coeff_a =  N/(N+mu)  # This lambda for the article language model
	coeff_b =  mu/(N+mu)  # This is (1-lambda) for the collection language model

	article_model = article_lm[index_][term_]/article_lm[index_]["__NumWordsArticle__"] 
	# Using the entire training collection as the language model (number of times word appears in collection / number of words in collection)
	interpolated_model = tf_vocab_stemmed[term_] / num_words_in_collection

	if smoothing == 'yes':
		value = (coeff_a * article_model) + (coeff_b * interpolated_model)
		if value == 0:
			return 0.0 # cuz log(0) will give inf
		else:
			return np.log( value )
	# No smoothing
	else:
		if article_model == 0:
			return 0.0 # cuz log(0) will give inf
		else:
			return np.log( article_model )



def calc_headline_lm(headline_lm, index_, term_):
	return headline_lm[index_][term_]/headline_lm[index_]["__NumWordsHeadline__"]


# KL-Divergence Calculation for train, cv, and test set
# For all words in the query (not intersection between query and article)
train_KL =  np.array( [  -sum([  calc_headline_lm(train_headlines_lm, i, t) * calc_article_lm(train_cv_articles_lm,  df_train['Body ID'][i] , t)    for t in set(df_train['Headline_stemmed'][i])  ])  for i in range(0, df_train.shape[0])  ]  )
cv_KL =   np.array( [  -sum([  calc_headline_lm(cv_headlines_lm, i, t) * calc_article_lm(train_cv_articles_lm,  df_cv['Body ID'][i] , t)    for t in set(df_cv['Headline_stemmed'][i])  ])  for i in range(0, df_cv.shape[0])  ]  )
test_KL = np.array( [  -sum([  calc_headline_lm(test_headlines_lm, i, t) * calc_article_lm(test_articles_lm,  df_test['Body ID'][i] , t)    for t in set(df_test['Headline_stemmed'][i])  ])  for i in range(0, df_test.shape[0])  ]  )


# Replace inf with zero (inf occurs cuz of division by zero),  i!=i is for NaN
train_KL = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in train_KL])
cv_KL = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in cv_KL])
test_KL = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in test_KL])

############################################################################################################



############################## BM25  ############################################

# Only use the complete train set articles
n_articles = Articles.shape[0] 
# Average document length: Sum training (Article) document lengths / number of training articles
avg_dl = sum( [ len(j) for j in np.array(Articles['articleBody_stemmed']) ] ) / n_articles  

def calc_BM25(headline, query_term, article_lm, index_):
	# docs term occurs in
	n_i = len( inverted_index[ query_term ] )
	bm25_p1 = (  ( (0+0.5)/(0-0+0.5) )/( (n_i -0+0.5)/(n_articles - n_i - 0 + 0 + 0.5) )     )

	k_1 = 1.2
	k_2 = 100
	b = 0.75
	f_i = article_lm[index_][ query_term ] # number of times term occurs in the Article
	qf_i = headline.count( query_term ) # number of times term occurs in the headline
	dl = article_lm[index_][ "__NumWordsArticle__" ]  # Document length

	K = k_1 * ((1-b) + (b* (dl/avg_dl) )  )
	bm25_p2 = ( (k_1+1) * f_i  ) / (K + f_i)
	bm25_p3 = ( (k_2+1) * qf_i )/(k_2+qf_i) 

	return np.log(bm25_p1) * bm25_p2 * bm25_p3



train_BM25 = np.array( [  sum([ calc_BM25(df_train['Headline_stemmed'][i], t, train_cv_articles_lm , df_train['Body ID'][i] )  for t in set(df_train['Headline_stemmed'][i])  ])  for i in range(0, df_train.shape[0])  ]  )
cv_BM25 = np.array( [  sum([ calc_BM25(df_cv['Headline_stemmed'][i], t, train_cv_articles_lm , df_cv['Body ID'][i] )  for t in set(df_cv['Headline_stemmed'][i])  ])  for i in range(0, df_cv.shape[0])  ]  )
test_BM25 = np.array( [  sum([ calc_BM25(df_test['Headline_stemmed'][i], t, test_articles_lm , df_test['Body ID'][i] )  for t in set(df_test['Headline_stemmed'][i])  ])  for i in range(0, df_test.shape[0])  ]  )


train_BM25 = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in train_BM25])
cv_BM25 = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in cv_BM25])
test_BM25 = np.array([0 if (i == math.inf or i == -math.inf or i!=i) else i for i in test_BM25])

##################################################################################


# FNC Baseline features

############################## Word Overlap Features ################################

train_wordoverlap = np.array( [ len(set( df_train['Headline_stemmed'][i] ).intersection(  df_train['articleBody_stemmed'][i] )) / float(len(set( df_train['Headline_stemmed'][i] ).union(  df_train['articleBody_stemmed'][i] )))  for i in range(0, df_train.shape[0])  ] )
cv_wordoverlap = np.array( [ len(set( df_cv['Headline_stemmed'][i] ).intersection(  df_cv['articleBody_stemmed'][i] )) / float(len(set( df_cv['Headline_stemmed'][i] ).union(  df_cv['articleBody_stemmed'][i] )))  for i in range(0, df_cv.shape[0])  ] )
test_wordoverlap =  np.array( [ len(set( df_test['Headline_stemmed'][i] ).intersection(  df_test['articleBody_stemmed'][i] )) / float(len(set( df_test['Headline_stemmed'][i] ).union(  df_test['articleBody_stemmed'][i] )))  for i in range(0, df_test.shape[0])  ] )

######################################################################################


############################## Refuting Features ################################
refuting_words = ['fake','fraud', 'hoax', 'false', 'deny', 'denies','not','despite','nope','doubt', 'doubts','bogus','debunk','pranks','retract']

train_refuting_features = np.array( [ [1 if word in df_train['Headline_stemmed'][i] else 0 for word in refuting_words] for i in range(0, df_train.shape[0])  ] )
cv_refuting_features = np.array( [ [1 if word in df_cv['Headline_stemmed'][i] else 0 for word in refuting_words] for i in range(0, df_cv.shape[0])  ] )
test_refuting_features =  np.array( [ [1 if word in df_test['Headline_stemmed'][i] else 0 for word in refuting_words] for i in range(0, df_test.shape[0])  ] )

######################################################################################


################################# Evaluation Metrics #######################################

labels_dict= {0: 'unrelated', 1: 'discuss', 2: 'agree', 3: 'disagree'}

# FNC scoring method: Divide the model score by the maximum possible score
# Diagram showing the scoring procedure is in the FNC-1 Website
def fnc_scorer(true_labels, predicted_labels):
	model_score = 0.0
	max_score = 0.0
	for i in range(0, predicted_labels.shape[0]):
		# Prediction correct
		if predicted_labels[i] == true_labels[i]:
			# Is it unrelated
			if predicted_labels[i] == 0:
				model_score += 0.25
			# Is it related (agree, disagree, discuss)
			if predicted_labels[i] == 1 or predicted_labels[i] == 2 or predicted_labels[i] == 3:
				model_score += (0.75+0.25)  # the 0.25 is for the unrelated portion
		# Prediction False
		else:
			# Did we at least guess that it was related or unrelated?
			if true_labels[i] == 1 or true_labels[i] == 2 or true_labels[i] == 3:
				if predicted_labels[i] == 1 or predicted_labels[i] == 2 or predicted_labels[i] == 3:
					model_score += 0.25

		# Calc max score
		if true_labels[i] == 0:
			max_score += 0.25
		if true_labels[i] == 1 or true_labels[i] == 2 or true_labels[i] == 3:
			max_score += (0.75+0.25)

	return model_score, max_score



def plot_confusion_matrix(true_labels, predicted_labels):
	# initialize confusion matrix with string indexes
	conf_matrix = pd.DataFrame(0, index=range(0,4), columns=labels ) 
	for i in range(0, 4):
		conf_matrix.rename(index={i: labels[i]}, inplace=True)
	# Assign the values
	for i in range(0, predicted_labels.shape[0]):
		conf_matrix[  labels_dict[ predicted_labels[i] ]  ][  labels_dict[ true_labels[i] ]  ] += 1  # Dataframe, you index the column first then the row
	return conf_matrix.T

# Precision, Recall, and F1 for each class
def precision_recall(conf_matrix):
	conf_matrix = np.array( conf_matrix )
	for i in range(0, len(conf_matrix)):
		# Diagonal has the related predictions / how many predictions I made for that class (row in confusion matrix)
		precision = np.diagonal(conf_matrix)[i] / np.sum(conf_matrix[i,:])
		# Recall deno is the number of samples with that class
		recall = np.diagonal(conf_matrix)[i] / np.sum( conf_matrix[:, i] )
		F1 = 2 * ( (precision*recall)/(precision+recall) )
		print (labels_dict[i], 'Precision: ', precision, 'Recall: ', recall, 'F1: ', F1)

######################################################################################


################################# Machine Learning Model ###########################################


# Horizantally Concatenate the features
# stacked_train_features = np.hstack((train_cosine_sim.reshape(-1,1), train_cosine_sim_lda.reshape(-1,1), train_tfidf.reshape(-1,1) , train_KL.reshape(-1,1), train_BM25.reshape(-1,1) ))
# stacked_cv_features = np.hstack(( cv_cosine_sim.reshape(-1,1), cv_cosine_sim_lda.reshape(-1,1), cv_tfidf.reshape(-1,1), cv_KL.reshape(-1,1), cv_BM25.reshape(-1,1) ))
# stacked_test_features =  np.hstack(( test_cosine_sim.reshape(-1,1), test_cosine_sim_lda.reshape(-1,1), test_tfidf.reshape(-1,1), test_KL.reshape(-1,1), test_BM25.reshape(-1,1) ))


stacked_train_features = np.hstack((train_cosine_sim.reshape(-1,1),  train_tfidf.reshape(-1,1) , train_KL.reshape(-1,1), train_BM25.reshape(-1,1) ))
stacked_cv_features = np.hstack(( cv_cosine_sim.reshape(-1,1),  cv_tfidf.reshape(-1,1), cv_KL.reshape(-1,1), cv_BM25.reshape(-1,1) ))
stacked_test_features =  np.hstack(( test_cosine_sim.reshape(-1,1),  test_tfidf.reshape(-1,1), test_KL.reshape(-1,1), test_BM25.reshape(-1,1) ))


# Replace nan with zero and inf with large finite numbers
stacked_train_features = np.nan_to_num(stacked_train_features)
stacked_cv_features = np.nan_to_num(stacked_cv_features)
stacked_test_features = np.nan_to_num(stacked_test_features)

# Scale each feature (0 mean unit std)
# scaled_features.mean(axis=0)  and  scaled_features.std(axis=0)
scaler = StandardScaler().fit(stacked_train_features)
stacked_train_features = scaler.transform(stacked_train_features)
stacked_cv_features = scaler.transform(stacked_cv_features)
stacked_test_features = scaler.transform(stacked_test_features)

# Append FNC baseline features (word overlap and refuting features)
stacked_train_features = np.hstack(( stacked_train_features,   train_wordoverlap.reshape(-1,1),  train_refuting_features ))
stacked_cv_features = np.hstack(( stacked_cv_features,   cv_wordoverlap.reshape(-1,1), cv_refuting_features ))
stacked_test_features =  np.hstack((stacked_test_features,   test_wordoverlap.reshape(-1,1), test_refuting_features ))


# (LR and XGBoost don't need categorical one-hot encoded labels)
train_labels = np.array(df_train['stance_labels'])
cv_labels = np.array(df_cv['stance_labels'])
test_labels = np.array(df_test['stance_labels'])


# Logistic Regression
# multi_class = One vs Rest (ovr), can also be 'multinomial' (Performs better)
model_LR = LogisticRegression(C=0.01, penalty="l2", class_weight='balanced' ,max_iter=500, multi_class='ovr', solver='lbfgs')
model_LR.fit(stacked_train_features, train_labels)
predicted_LR_cv = model_LR.predict(stacked_cv_features)
predicted_LR_test = model_LR.predict(stacked_test_features)

print ('CV Accuracy LR: ', sum(predicted_LR_cv == cv_labels)/cv_labels.shape[0])
print ('Test Accuracy LR: ', sum(predicted_LR_test == test_labels)/test_labels.shape[0])
cv_eval_score, cv_max_score = fnc_scorer(cv_labels, predicted_LR_cv)
test_eval_score, test_max_score = fnc_scorer(test_labels, predicted_LR_test)
print ('CV FNC Scorer: ', cv_eval_score/cv_max_score)
print ('Test FNC Scorer: ', test_eval_score/test_max_score)
print ('CV confusion matrix: ', plot_confusion_matrix(cv_labels, predicted_LR_cv) )
print ('Test confusion matrix: ', plot_confusion_matrix(test_labels, predicted_LR_test) )

cv_labels1 = np_utils.to_categorical(cv_labels, num_classes)
predicted_LR_cv1 = np_utils.to_categorical(predicted_LR_cv, num_classes)
print ('CV F1 Score: ', metrics.f1_score(cv_labels1, predicted_LR_cv1, average='weighted'))

test_labels1 = np_utils.to_categorical(test_labels, num_classes)
predicted_LR_test1 = np_utils.to_categorical(predicted_LR_test, num_classes)
print ('Test F1 Score: ', metrics.f1_score(test_labels1, predicted_LR_test1, average='weighted'))

# Precision and Recall for the different classes
precision_recall( plot_confusion_matrix(cv_labels, predicted_LR_cv)  )
precision_recall( plot_confusion_matrix(test_labels, predicted_LR_test)  )

# XG-Boost
param = {} # Parameters
param['objective'] = 'multi:softmax'
param['objective'] = 'multi:softprob'  #This is the same but outputs probabilities
param['max_depth'] = 6
model_xgb = xgb.XGBClassifier(param, max_depth=3, n_estimators=300, learning_rate=0.1, objective='multi:softmax')
model_xgb = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1, objective='multi:softmax')
model_xgb.fit(stacked_train_features, train_labels)
predicted_xgb_cv = model_xgb.predict(stacked_cv_features)
predicted_xgb_test = model_xgb.predict(stacked_test_features)

print ('CV Accuracy XGBoost: ', sum(predicted_xgb_cv == cv_labels)/cv_labels.shape[0])
print ('Test Accuracy XGBoost: ', sum(predicted_xgb_test == test_labels)/test_labels.shape[0])
cv_eval_score, cv_max_score = fnc_scorer(cv_labels, predicted_xgb_cv)
test_eval_score, test_max_score = fnc_scorer(test_labels, predicted_xgb_test)
print ('CV FNC Scorer: ', cv_eval_score/cv_max_score)
print ('Test FNC Scorer: ', test_eval_score/test_max_score)
print ('CV confusion matrix: ', plot_confusion_matrix(cv_labels, predicted_xgb_cv) )
print ('Test confusion matrix: ', plot_confusion_matrix(test_labels, predicted_xgb_test) )


test_labels1 = np_utils.to_categorical(test_labels, num_classes)
predicted_xgb_test1 = np_utils.to_categorical(predicted_xgb_test, num_classes)
print ('Test F1 Score: ', metrics.f1_score(test_labels1, predicted_xgb_test1, average='weighted'))

# Precision and Recall for the different classes
precision_recall( plot_confusion_matrix(cv_labels, predicted_xgb_cv)   )
precision_recall( plot_confusion_matrix(test_labels, predicted_xgb_test)  )


##########################################################################################################



################################# Deep Learning Model + SVD/Topic Modelling Features ###########################################



############################## Count-Vectorizer for 5000 words ################################

# If binary is False, then it uses the Term Frequencies as the count
# min_df: ignore terms that appear in less than 5 documents
vectorizer_DL = CountVectorizer(binary=False, max_features=5000)
# Learn the vocabulary dictionary and return term-document matrix
articles_BoW_vectors_DL = vectorizer_DL.fit_transform(articles_sklearn)
articles_BoW_vectors_DL.toarray().shape  # (1683, 5728)
# vectorizer.get_feature_names()

# Transform train set
train_articles_sklearn_DL = [ ' '.join( df_train['articleBody_stemmed'][i] ) for i in range(0, df_train.shape[0]) ]
train_headlines_sklearn_DL = [ ' '.join( df_train['Headline_stemmed'][i] ) for i in range(0, df_train.shape[0]) ]
train_articles_BoW_DL =  vectorizer_DL.transform(train_articles_sklearn_DL).toarray()
train_headlines_BoW_DL =  vectorizer_DL.transform(train_headlines_sklearn_DL).toarray()

# Transform cv set
cv_articles_sklearn_DL = [ ' '.join( df_cv['articleBody_stemmed'][i] ) for i in range(0, df_cv.shape[0]) ]
cv_headlines_sklearn_DL = [ ' '.join( df_cv['Headline_stemmed'][i] ) for i in range(0, df_cv.shape[0]) ]
cv_articles_BoW_DL =  vectorizer_DL.transform(cv_articles_sklearn_DL).toarray()
cv_headlines_BoW_DL =  vectorizer_DL.transform(cv_headlines_sklearn_DL).toarray()

# Transform test set
test_articles_sklearn_DL = [ ' '.join( df_test['articleBody_stemmed'][i] ) for i in range(0, df_test.shape[0]) ]
test_headlines_sklearn_DL = [ ' '.join( df_test['Headline_stemmed'][i] ) for i in range(0, df_test.shape[0]) ]
test_articles_BoW_DL =  vectorizer_DL.transform(test_articles_sklearn_DL).toarray()
test_headlines_BoW_DL =  vectorizer_DL.transform(test_headlines_sklearn_DL).toarray()

train_cosine_sim_DL = cosine_sim(train_headlines_BoW_DL, train_articles_BoW_DL)
cv_cosine_sim_DL = cosine_sim(cv_headlines_BoW_DL, cv_articles_BoW_DL)
test_cosine_sim_DL = cosine_sim(test_headlines_BoW_DL, test_articles_BoW_DL)
##############################################################################################################################



############################## Cosine Sim between LDA Vectors (Topic Modelling) ################################

# We already have the TF count vectorizers above

# 300 Topics
lda = LatentDirichletAllocation(n_components = 300, learning_method='online', max_iter=10)
lda.fit(articles_BoW_vectors_DL)

# Get topics for the train, cv, and test set from the TF-BoW
train_headlines_lda = lda.transform(train_headlines_BoW_DL)
train_articles_lda = lda.transform(train_articles_BoW_DL)

cv_headlines_lda = lda.transform(cv_headlines_BoW_DL)
cv_articles_lda = lda.transform(cv_articles_BoW_DL)

test_headlines_lda = lda.transform(test_headlines_BoW_DL)
test_articles_lda = lda.transform(test_articles_BoW_DL)


# Cosine Similarity betwene number of topics
train_cosine_sim_lda = cosine_sim(train_headlines_lda, train_articles_lda)
cv_cosine_sim_lda = cosine_sim(cv_headlines_lda, cv_articles_lda)
test_cosine_sim_lda = cosine_sim(test_headlines_lda, test_articles_lda)

##############################################################################################


############################## Cosine Sim on SVD of TF-BoW ################################

# SVD on the TF-BoW Vectors
svd = TruncatedSVD(n_components=300, n_iter=10, random_state=42)
train_articles_BoW_svd = svd.fit_transform( train_articles_BoW_DL ) 
train_headlines_BoW_svd = svd.transform( train_headlines_BoW_DL )
cv_articles_BoW_svd = svd.transform( cv_articles_BoW_DL )
cv_headlines_BoW_svd = svd.transform( cv_headlines_BoW_DL )
test_articles_BoW_svd = svd.transform( test_articles_BoW_DL )
test_headlines_BoW_svd = svd.transform( test_headlines_BoW_DL )

# Cosine Similarity on SVD of TF-BoW vectors
train_cosine_sim_svd = cosine_sim(train_headlines_BoW_svd, train_articles_BoW_svd)
cv_cosine_sim_svd = cosine_sim(cv_headlines_BoW_svd, cv_articles_BoW_svd)
test_cosine_sim_svd = cosine_sim(test_headlines_BoW_svd, test_articles_BoW_svd)

##############################################################################################



######################### MLP #########################################################

train_labels = np.array(df_train['stance_labels'])
cv_labels = np.array(df_cv['stance_labels'])
test_labels = np.array(df_test['stance_labels'])

# Categorical labels (LR and XGBoost don't need categorical one-hot encoded labels, only for MLP Keras)
num_classes = len(labels)
Y_train = np_utils.to_categorical(train_labels, num_classes)
Y_cv = np_utils.to_categorical(cv_labels, num_classes)
Y_test = np_utils.to_categorical(test_labels, num_classes)


num_features = stacked_train_features.shape[1]
num_train_samples = stacked_train_features.shape[0]

model_MLP = Sequential()
model_MLP.add(Dense(128, activation='relu', input_dim=num_features, kernel_initializer='he_uniform'))
model_MLP.add(Dropout(0.5))
model_MLP.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_MLP.add(Dropout(0.5))
model_MLP.add(Dense(num_classes, activation='softmax'))
model_MLP.summary()


model_MLP.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer='adam',
			  metrics=['accuracy'])

model_MLP.fit(stacked_train_features, Y_train,
			batch_size=512,
			epochs=20,
			shuffle=True,
			validation_data = (stacked_cv_features ,Y_cv),
			verbose=1)


predicted_MLP_labels = np.argmax(model_MLP.predict(stacked_test_features), axis=1)
# predicted_MLP_probs = model_MLP.predict(stacked_test_features)  # Gives classes probabilities

print (metrics.classification_report(test_labels, predicted_MLP_labels))
print ('Test Accuracy MLP: ', sum(predicted_MLP_labels == test_labels)/test_labels.shape[0])
test_eval_score, test_max_score = fnc_scorer(test_labels, predicted_MLP_labels)
print ('Test FNC Scorer: ', test_eval_score/test_max_score)
print ('Test confusion matrix: ', plot_confusion_matrix(test_labels, predicted_MLP_labels) )

precision_recall( plot_confusion_matrix(test_labels, predicted_MLP_labels)  )


##########################################################################################################


##################################### Equally Weighted Ensemble (MLP + XGB) ################

predicted_xgb = model_xgb.predict_proba(stacked_test_features)
predicted_mlp = model_MLP.predict(stacked_test_features)

ensemble_predictions = (predicted_xgb+predicted_mlp)/2  # Weighted equally
ensemble_predictions_labels = np.argmax(ensemble_predictions, axis=1)

print ('Test Accuracy MLP: ', sum(ensemble_predictions_labels == test_labels)/test_labels.shape[0])
test_eval_score, test_max_score = fnc_scorer(test_labels, ensemble_predictions_labels)
print ('Test FNC Scorer: ', test_eval_score/test_max_score)
print ('Test confusion matrix: ', plot_confusion_matrix(test_labels, ensemble_predictions_labels) )

precision_recall( plot_confusion_matrix(test_labels, ensemble_predictions_labels)  )

print ('Test F1 Score: ', metrics.f1_score(test_labels, ensemble_predictions_labels, average='weighted'))

##########################################################################################################


######################################### LSTM Neural Model ############################################

# pre-trained Google News corpus (3 billion running words) word vector model (3 million 300-dimension English word vectors)
word_vectors = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  # C binary format

# 3,000,000 words
word_vectors.wv.vocab
word_vectors.wv.vocab.keys()

# Word embedding for <UNK> token
num_embedding_dim = 300
unk_embedding = np.array( [random.uniform(-1,1) for i in range(num_embedding_dim)] )
pad_embedding = np.zeros(num_embedding_dim)

# The word vectors for each word in the headline and article ( Headline limit is 20 words and Article limit is 200 words)

train_headlines_Word2Vec = np.array( [  [ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_train['Headline_Word2Vec_Pretrained'][i]  ]  for i in range(0, df_train.shape[0])  ]  )
train_articles_Word2Vec = np.array( [  [ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_train['articleBody_Word2Vec_Pretrained'][i]  ]  for i in range(0, df_train.shape[0])  ]  )
cv_headlines_Word2Vec = np.array( [  [ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_cv['Headline_Word2Vec_Pretrained'][i]  ]  for i in range(0, df_cv.shape[0])  ]  )
cv_articles_Word2Vec = np.array( [  [ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_cv['articleBody_Word2Vec_Pretrained'][i]  ]  for i in range(0, df_cv.shape[0])  ]  )
test_headlines_Word2Vec = np.array( [  [ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_test['Headline_Word2Vec_Pretrained'][i]  ] for i in range(0, df_test.shape[0])  ]  )
test_articles_Word2Vec = np.array( [  [ word_vectors.wv[t] if t !='<_UNK_>' else  unk_embedding for t in df_test['articleBody_Word2Vec_Pretrained'][i]  ]  for i in range(0, df_test.shape[0]) ] )


# Find maximum headline and article length to figure which ones have to be padded or tuncated
max_len_headline = len(max(train_headlines_Word2Vec, key=len)) 
max_len_article = len(max(train_articles_Word2Vec, key=len)) 

# headline and article limits (All words longer will be truncated)
headline_word_limit = 20
article_word_limit = 200

# Upon truncation, only take first x tokens
train_headlines_Word2Vec = np.array( [i if len(i)==headline_word_limit else i+([ list(pad_embedding) ]*(headline_word_limit-len(i))) if len(i)<headline_word_limit  else  i[:headline_word_limit]  for i in train_headlines_Word2Vec] )
cv_headlines_Word2Vec = np.array( [i if len(i)==headline_word_limit else i+([ list(pad_embedding) ]*(headline_word_limit-len(i))) if len(i)<headline_word_limit  else  i[:headline_word_limit]  for i in cv_headlines_Word2Vec] )
test_headlines_Word2Vec = np.array( [i if len(i)==headline_word_limit else i+([ list(pad_embedding) ]*(headline_word_limit-len(i))) if len(i)<headline_word_limit  else  i[:headline_word_limit]  for i in test_headlines_Word2Vec] )


train_articles_Word2Vec = np.array( [i if len(i)==article_word_limit else i+([ list(pad_embedding) ]*(article_word_limit-len(i))) if len(i)<article_word_limit  else  i[:article_word_limit]  for i in train_articles_Word2Vec] )
cv_articles_Word2Vec = np.array( [i if len(i)==article_word_limit else i+([ list(pad_embedding) ]*(article_word_limit-len(i))) if len(i)<article_word_limit  else  i[:article_word_limit]  for i in cv_articles_Word2Vec] )
test_articles_Word2Vec = np.array( [i if len(i)==article_word_limit else i+([ list(pad_embedding) ]*(article_word_limit-len(i))) if len(i)<article_word_limit  else  i[:article_word_limit]  for i in test_articles_Word2Vec] )



# 2 Bi-LSTM model to get encoded versions of the headline and article independently from the word embeddings 
# The output of the intermediate layers (encoded headline and article) are concatenated in order to train a MLP

train_labels = np.array(df_train['stance_labels'])
cv_labels = np.array(df_cv['stance_labels'])
test_labels = np.array(df_test['stance_labels'])

# Categorical labels (LR and XGBoost don't need categorical one-hot encoded labels, only for MLP Keras)
num_classes = len(labels)
y_train = np_utils.to_categorical(train_labels, num_classes)
y_cv = np_utils.to_categorical(cv_labels, num_classes)
y_test = np_utils.to_categorical(test_labels, num_classes)

x_train = train_headlines_Word2Vec
x_cv = cv_headlines_Word2Vec
x_test = test_headlines_Word2Vec

_, timesteps, data_dim = x_train.shape

num_hidden_lstm = 64
num_hidden_units = 64
batch_size = 128
epochs = 10
model_patience = 20


#LSTM expects 3D data (batch_size, timesteps, features)
model = Sequential()
model.add(LSTM(num_hidden_lstm, input_shape=(timesteps,data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 128
model.add(LSTM(num_hidden_lstm, return_sequences=False))  # return a single vector of dimension 128
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(
              loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


# Train model to encode the headlines
H = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=(x_cv, y_cv),
            # validation_split = 0,   #15% of last x_train and y_train taken for CV before shuffling
            callbacks =[EarlyStopping(monitor='val_loss', patience= model_patience)]  #6 epochs patience
            )


headline_encoding_layer = 'lstm_8'
headline_intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(headline_encoding_layer).output)
encoded_train_headline = headline_intermediate_layer_model.predict(x_train)
encoded_cv_headline = headline_intermediate_layer_model.predict(x_cv)
encoded_test_headline = headline_intermediate_layer_model.predict(x_test)




# Train model to encode the articles
x_train = train_articles_Word2Vec
x_cv = cv_articles_Word2Vec
x_test = test_articles_Word2Vec

_, timesteps, data_dim = x_train.shape

num_hidden_lstm = 64
num_hidden_units = 64
batch_size = 64
epochs = 5
model_patience = 20;

H = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=(x_cv, y_cv),
            # validation_split = 0,   #15% of last x_train and y_train taken for CV before shuffling
            callbacks =[EarlyStopping(monitor='val_loss', patience= model_patience)]  #6 epochs patience
            )


article_encoding_layer = 'lstm_10'
article_intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(article_encoding_layer).output)
encoded_train_article= article_intermediate_layer_model.predict(x_train)
encoded_cv_article = article_intermediate_layer_model.predict(x_cv)
encoded_test_article = article_intermediate_layer_model.predict(x_test)



# Horizantally Concatenate them both and train a MLP with a 4 node softmax at the end
x_train_concat = np.hstack(( encoded_train_headline,  encoded_train_headline))
x_cv_concat = np.hstack(( encoded_cv_headline,  encoded_cv_article))
x_test_concat = np.hstack(( encoded_test_headline,  encoded_test_article))


num_features = x_train_concat.shape[1]
num_train_samples = x_train_concat.shape[0]

model_MLP = Sequential()
model_MLP.add(Dense(128, activation='relu', input_dim=num_features, kernel_initializer='he_uniform'))
model_MLP.add(Dropout(0.5))
model_MLP.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_MLP.add(Dropout(0.5))
model_MLP.add(Dense(num_classes, activation='softmax'))
model_MLP.summary()


model_MLP.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer='adam',
			  metrics=['accuracy'])

model_MLP.fit(x_train_concat, y_train,
			batch_size=512,
			epochs=20,
			shuffle=True,
			validation_data = (x_cv_concat ,y_cv),
			verbose=1)


predicted_MLP_labels = np.argmax(model_MLP.predict(x_test_concat), axis=1)

print (metrics.classification_report(test_labels, predicted_MLP_labels))
print ('Test Accuracy MLP: ', sum(predicted_MLP_labels == test_labels)/test_labels.shape[0])
test_eval_score, test_max_score = fnc_scorer(test_labels, predicted_MLP_labels)
print ('Test FNC Scorer: ', test_eval_score/test_max_score)
print ('Test confusion matrix: ', plot_confusion_matrix(test_labels, predicted_MLP_labels) )

precision_recall( plot_confusion_matrix(test_labels, predicted_MLP_labels)  )
print ('Test F1 Score: ', metrics.f1_score(test_labels, predicted_MLP_labels, average='weighted'))

##########################################################################################################










########## Plotting distribution features for each stance for training data #########


# Plot distribution of cosine similiarity of TF BoW vectors
train_cosine_sim_distribution = pd.DataFrame(np.hstack((train_cosine_sim.reshape(-1,1), np.array(df_train['Stance']).reshape(-1,1))))
train_cosine_sim_distribution.columns = ['feature_val', 'Stance']

# Find the min, max, and mean for each feature value
the_avg = np.array( train_cosine_sim_distribution.groupby(['Stance']).sum() /  train_cosine_sim_distribution.groupby(['Stance']).count() ).reshape(-1)
the_min = np.array( train_cosine_sim_distribution.groupby(['Stance']).min() ).reshape(-1)
the_max = np.array( train_cosine_sim_distribution.groupby(['Stance']).max() ).reshape(-1)


# The x-axis values
plt.xticks(np.arange(1,5),('agree', 'disagree', 'discuss', 'unrelated'))
plt.title('Distribution of Cosine Similarity TF BoW features')
plt.ylabel('Cosine Similarity of the TF BoW vectors')

# Plot a vertical line from min to max value ( for each stance label from (x1,x2) tp (y1,y2) )
plt.plot((1, 1), (the_min[0], the_max[0]), (2, 2), (the_min[1], the_max[1]), (3, 3), (the_min[2], the_max[2]), (4, 4), (the_min[3], the_max[3]), 'k-')

# Plot the average
plt.plot((0.9, 1.1), (the_avg[0], the_avg[0]), (1.9, 2.1), (the_avg[1], the_avg[1]), (2.9, 3.1), (the_avg[2], the_avg[2]), (3.9, 4.1), (the_avg[3], the_avg[3]), 'k-')

plt.show()
plt.clf()



# Plot distribution of TF-IDF
train_tfidf_distribution = pd.DataFrame(np.hstack((train_tfidf.reshape(-1,1), np.array(df_train['Stance']).reshape(-1,1))))
train_tfidf_distribution.columns = ['feature_val', 'Stance']

# Find the min, max, and mean for each feature value
the_avg = np.array( train_tfidf_distribution.groupby(['Stance']).sum() /  train_tfidf_distribution.groupby(['Stance']).count() ).reshape(-1)
the_min = np.array( train_tfidf_distribution.groupby(['Stance']).min() ).reshape(-1)
the_max = np.array( train_tfidf_distribution.groupby(['Stance']).max() ).reshape(-1)


# The x-axis values
plt.xticks(np.arange(1,5),('agree', 'disagree', 'discuss', 'unrelated'))
plt.title('Distribution of TF-IDF features')
plt.ylabel('TF-IDF')

# Plot a vertical line from min to max value ( for each stance label from (x1,x2) tp (y1,y2) )
plt.plot((1, 1), (the_min[0], the_max[0]), (2, 2), (the_min[1], the_max[1]), (3, 3), (the_min[2], the_max[2]), (4, 4), (the_min[3], the_max[3]), 'k-')

# Plot the average
plt.plot((0.9, 1.1), (the_avg[0], the_avg[0]), (1.9, 2.1), (the_avg[1], the_avg[1]), (2.9, 3.1), (the_avg[2], the_avg[2]), (3.9, 4.1), (the_avg[3], the_avg[3]), 'k-')

plt.show()
plt.clf()




# Plot distribution of BM25
train_BM25_distribution = pd.DataFrame(np.hstack((train_BM25.reshape(-1,1), np.array(df_train['Stance']).reshape(-1,1))))
train_BM25_distribution.columns = ['feature_val', 'Stance']

# Find the min, max, and mean for each feature value
the_avg = np.array( train_BM25_distribution.groupby(['Stance']).sum() /  train_BM25_distribution.groupby(['Stance']).count() ).reshape(-1)
the_min = np.array( train_BM25_distribution.groupby(['Stance']).min() ).reshape(-1)
the_max = np.array( train_BM25_distribution.groupby(['Stance']).max() ).reshape(-1)


# The x-axis values
plt.xticks(np.arange(1,5),('agree', 'disagree', 'discuss', 'unrelated'))
plt.title('Distribution of BM25 features')
plt.ylabel('BM25')

# Plot a vertical line from min to max value ( for each stance label from (x1,x2) tp (y1,y2) )
plt.plot((1, 1), (the_min[0], the_max[0]), (2, 2), (the_min[1], the_max[1]), (3, 3), (the_min[2], the_max[2]), (4, 4), (the_min[3], the_max[3]), 'k-')

# Plot the average
plt.plot((0.9, 1.1), (the_avg[0], the_avg[0]), (1.9, 2.1), (the_avg[1], the_avg[1]), (2.9, 3.1), (the_avg[2], the_avg[2]), (3.9, 4.1), (the_avg[3], the_avg[3]), 'k-')

plt.show()
plt.clf()



# Plot distribution of KL-Divergence
train_KL_distribution = pd.DataFrame(np.hstack((train_KL.reshape(-1,1), np.array(df_train['Stance']).reshape(-1,1))))
train_KL_distribution.columns = ['feature_val', 'Stance']

# Replace all -0s with large number (representing infinity)
train_KL_distribution['feature_val'] = train_KL_distribution['feature_val'].apply( lambda x: 2000 if x ==0 else x )

# Find the min, max, and mean for each feature value
the_avg = np.array( train_KL_distribution.groupby(['Stance']).sum() /  train_KL_distribution.groupby(['Stance']).count() ).reshape(-1)
the_min = np.array( train_KL_distribution.groupby(['Stance']).min() ).reshape(-1)
the_max = np.array( train_KL_distribution.groupby(['Stance']).max() ).reshape(-1)


# The x-axis values
plt.xticks(np.arange(1,5),('agree', 'disagree', 'discuss', 'unrelated'))
plt.title('Distribution of KL-Divergence features')
plt.ylabel('KL-Divergence')

# Plot a vertical line from min to max value ( for each stance label from (x1,x2) tp (y1,y2) )
plt.plot((1, 1), (the_min[0], the_max[0]), (2, 2), (the_min[1], the_max[1]), (3, 3), (the_min[2], the_max[2]), (4, 4), (the_min[3], the_max[3]), 'k-')

# Plot the average
plt.plot((0.9, 1.1), (the_avg[0], the_avg[0]), (1.9, 2.1), (the_avg[1], the_avg[1]), (2.9, 3.1), (the_avg[2], the_avg[2]), (3.9, 4.1), (the_avg[3], the_avg[3]), 'k-')

plt.show()
plt.clf()



######################################################################################



























