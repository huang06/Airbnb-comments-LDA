""" 
Dataset Input 
""" 
import pandas as pd 
listings = pd.DataFrame(pd.read_csv('C:/Users/Dormitory/Desktop/listings.csv')) 
reviews = pd.DataFrame(pd.read_csv('C:/Users/Dormitory/Desktop/reviews.csv')) 
reviews = reviews.dropna() 
listings.rename(columns = {'id':'listing_id'}, inplace = True) 
reviews['room_type'] = "" 
reviews['room_type'] = reviews['listing_id'].map(listings.set_index('listing_id')['room_type']) 
 
df = reviews.copy() 
#Select rows with values of 'Private room' in column 'room_type'. 
df = df.loc[df['room_type'] == 'Private room'] 
df = df.dropna() 
 
""" 
Pre-process Phase 
""" 
from nltk.corpus import stopwords   
import time 
# The TypeError: 'float' object is not iterable could happen if the data is missing a value 
df = df.dropna() 
pre_start = time.time() 
 
# Remove Punctuations 
import string 
df['comments'] = [''.join(c for c in s if c not in string.punctuation) for s in df['comments']] 
print("Remove Punctuations : ") 
df['comments'].head(10) 
 
# Transform to lowcase and split 
df['comments'] = df['comments'].str.lower().str.split()   
df['comments'].head(10) print("lowcase and split : ") 
df['comments'].head(10) 
 
# Remove stopwords 
stop = stopwords.words('english') 
df['comments'] = df['comments'].apply(lambda x: [item for item in x if item not in stop]) 
print("Remove stopwords : ") 
df['comments'].head(10) 
 
# Stemming 
from nltk.stem import RegexpStemmer 
st = RegexpStemmer('ing$|s$|e$|able$', min=4) 
for x in df['comments']: 
        for y in x: 
                y = st.stem(y) 
print("Stemming : ") 
df['comments'].head(10) 
 
# Remove Strings which length > 3 
df['comments'] = df['comments'].apply(lambda x: [item for item in x if len(item)>3 ]) 
print("Remove Strings which length > 3    : ") 
df['comments'].head(10) 
 
pre_end = time.time() 
print("It cost %f sec" % (pre_end - pre_start)) 
 
""" 
Group Comments by the column of 'listing_id' 
""" 
df2 = df[['listing_id', 'comments']].copy() 
# To return a Dataframe 
df2.groupby('listing_id').apply(lambda x: x.sum()) 
 
""" 
LDA Phase 
""" 
# Establish dictionary and corpus 
lda_start = time.time() 
from gensim import corpora, models 
dictionary = corpora.Dictionary(df2['comments']) 
corpus = [ dictionary.doc2bow(text) for text in df2['comments'] ] 
 # Transform Bag-of-Words to TF/IDF   
tfidf = models.TfidfModel(corpus) 
corpus_tfidf = tfidf[corpus] 
 
from nltk.probability import FreqDist 
fdist = FreqDist(dictionary) 
top_ten = fdist.most_common(1000) 
lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=20) 
#lda = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=20, workers=3) 
lda_end = time.time() 
print("It cost %f sec" % (lda_end - lda_start)) 
 
# Print Top20 topics 
lda.print_topics(20) 
# Print the dist. of 20th topic 
lda.print_topic(19)
