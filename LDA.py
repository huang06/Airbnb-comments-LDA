"""
Dataset Input
"""
import pandas as pd
df = pd.DataFrame(pd.read_csv('C:/Users/lab-jack/Desktop/new york_review.csv'))
#df = [str(i) for i in df]
df = df.drop("id", axis = 1)
df = df.drop("date", axis = 1)
df = df.drop("reviewer_name", axis = 1)
df['id'] = df.index+1
df = df[['id','listing_id','reviewer_id','comments']]
test = df.head(100)
df.head()
# dataset output
df.to_csv('out.csv',index=False,encoding='utf-8')

"""
Test Phase
"""
test = pd.DataFrame(pd.read_csv('out.csv'))
#test = test.rename(columns = {'Unnamed: 0':'id'})
import types
test = df.head(100)
# Transform to lowcase and split
test['comments'] = test['comments'].str.lower().str.split() 
# Select non-stopwords
stop = stopwords.words('english')
test['comments'] = test['comments'].apply(lambda x: [item for item in x if item not in stop and ])

test['comments'] = test
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

test['comments'] = test['comments'].apply(remove_punctuation)



"""
Pre-process

"""
from nltk.corpus import words
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]   


"""
LDA Phase
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
sklearn.decomposition.LatentDirichletAllocation(
	n_topics=10,
	doc_topic_prior=None, 
	topic_word_prior=None, 
	learning_method=None, 
	learning_decay=0.7, 
	learning_offset=10.0, 
	max_iter=10, 
	batch_size=128, 
	evaluate_every=-1, 
	total_samples=1000000.0, 
	perp_tol=0.1, 
	mean_change_tol=0.001, 
	max_doc_update_iter=100, 
	n_jobs=1, 
	verbose=0, 
	random_state=None )
