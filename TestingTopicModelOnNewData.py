import pandas as pd
import os

%matplotlib inline
import csv
import gensim
import numpy as np
import pandas as pd
import re
import os
import pprint
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import nltk
import sys
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk import sent_tokenize, word_tokenize

"""Additional stopwords"""
colnames = ['stopwords']
data = pd.read_csv('StopWords.csv', names=colnames)
#print data
additional_stopwords = data.stopwords.tolist()
additional_stopwords= map(str.strip, additional_stopwords) 
print len(additional_stopwords)

colnamesStem = ['StemStopwords']
data1 = pd.read_csv('AfterStemStopWords.csv', names=colnamesStem)
#print data1
AfterStem_stopwords = data1.StemStopwords.tolist()
AfterStem_stopwords= map(str.strip, AfterStem_stopwords) 
print len(AfterStem_stopwords)

def cln_tokenizer(text):
    text = str(text)
    text = text.strip()
    text = text.lower() 
    
    text = re.sub("[^A-Za-z]", " ", text)
    text =re.sub(' +',' ',text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if (word not in (stopwords.words("english"))) & 
                                         (len(word) >= 3)
              &(word not in additional_stopwords) ]
    stem_tokens = []
    for word in tokens:
        stem_tokens.append(PorterStemmer().stem(word))
    stem_tokens =[word for word in stem_tokens if (word not in (stopwords.words("english"))) & 
                                         (len(word) >= 3)
                                         & (word not in AfterStem_stopwords)]
    return stem_tokens
    
model_location = "C:\Users\Shubhi Tyagi\Downloads"
model_name = "LDA_MODEL-Development.model"
dictionary_name = "LDA_DICT-Development.dict"

%%time
Lda_Model = models.LdaModel.load(os.path.join(model_location, model_name))
dictionary = corpora.Dictionary.load(os.path.join(model_location, dictionary_name))
print dictionary
print type(dictionary)


#improve exists in dictionary
print 'improv' in dictionary.token2id
#corpus = [dictionary.doc2bow(text) for text in texts]

newFeedBack =  "Although we appreciates your efforts to constantly improve but yu need to further work on building and maintaining relationships going outside your comfort zone. The committee believes this as a potential source of better renewal of in the future"
print len(newFeedBack)
cleanedText = cln_tokenizer(newFeedBack)
cleanedText
type(cleanedText)
print len(cleanedText)

#checking for words in feedback which are not part of the existing dictionary
for word in cleanedText:
    if(word in dictionary.token2id):
        print "Yes "
    else:
        print "No "+word
        
print len(cleanedText)

#remove words not part of dictionary from newFeedBack
for word in cleanedText:
    if(word in dictionary.token2id):
        print "Yes "
    else:
        print "Removed "+word
        cleanedText.remove(word)
        
print len(cleanedText)

#Convert doc to bagOfWords
bowText = dictionary.doc2bow(cleanedText)

print(Lda_Model[bowText])
