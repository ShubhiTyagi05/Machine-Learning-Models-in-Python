import pandas as pd
import os
directory = "C:\Users\Shubhi Tyagi\ParsedDocx"
dataList = []
filenames = []
for foldername in os.listdir(directory):
    print foldername
    for filename in os.listdir(directory+"\\"+foldername):
        with open(directory+"\\"+foldername+"\\"+filename, 'r') as myfile:
            data=myfile.read()
            data = str("\"")+data+str("\"")
            dataList.append(data)
            filenames.append(filename+","+foldername)
df = pd.DataFrame(dataList)
filenames = pd.DataFrame(filenames)

print len(df.columns)
print len(df)
print df

df = df.T.squeeze()
print type(df)
print len(df)

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

""" Model and dictionary parameters """
model_location = r"C:\Users\Shubhi Tyagi\Downloads\Projects\Partner Feedback\Scripts"
model_name = r"LDA_MODEL-AllDocsWSW6Threshv2.model"
dictionary_name = r"LDA_DICT-AllDocsWSW6Threshv2.dict"

"""Feature selections parameters"""
# Minimum frequency below which words are not to be considered (an absolute number) #3
low_cutoff = 3 #3
# Maximum fraction of total corpus size above which words are not to be considered (a fraction)
high_cutoff = 0.7 #0.7 

"""LDA model parameters"""
# Number of topics
n_topics = 4#6
# Number of passes (20)
n_passes = 1200#600

"""Additional stopwords"""
colnames = ['stopwords']
data = pd.read_csv('StopWords.csv', names=colnames)
print data
additional_stopwords = data.stopwords.tolist()
additional_stopwords= map(str.strip, additional_stopwords) 
print len(additional_stopwords)

colnamesStem = ['StemStopwords']
data1 = pd.read_csv('AfterStemStopWords.csv', names=colnamesStem)
print data1
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
    
  def dict_bow(texts):
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below = low_cutoff, no_above = high_cutoff)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return (corpus, dictionary)
   
   def LDA_model(corpus, state):
    return gensim.models.ldamodel.LdaModel(corpus,
                                           num_topics = n_topics,
                                           id2word = dictionary,
                                           passes = n_passes,
                                           random_state = state)
                                       
 %%time

state = np.random.RandomState(124)

texts =[]
for document in df:
    texts.append(cln_tokenizer(document))

corpus, dictionary = dict_bow(texts)
Lda_Model = LDA_model(corpus, state)

#Lda_Model.save(os.path.join(model_location, model_name))
#dictionary.save(os.path.join(model_location, dictionary_name))
print len(dictionary)
print dictionary

pprint.pprint(Lda_Model.print_topics(num_topics = n_topics, num_words = 75))

vis_data = gensimvis.prepare(Lda_Model, corpus, dictionary)
pyLDAvis.save_html(vis_data, "LDA_Visual.html")
pyLDAvis.display(vis_data)

definitions = Lda_Model.print_topics(num_topics = n_topics, num_words = 75)
print type(definitions)

firstSplit=str(definitions).split(")")
print len(firstSplit)
print firstSplit[0]
i=0
text = "Topic,Weight,Word Root\n"
for entry in firstSplit:
    entity = entry.split("+")
    entity = entity[4:]
    #print entity[0]
    for entry2 in entity:
        #print "Entry "+entry2
        entry2 = entry2[1:]
        #print "entry2"+entry2
        tokens = entry2.split("*")
        #print "tokens "+str(tokens)
        weight = tokens[0]
        #print tokens[1]
        root = tokens[1][1:-2]
        #print "weight "+weight+"root "+root
        #print "root "+root
        text = text+str(i)+" , "+weight+" , "+root+"\n"
        #print text
        #break
    #break
    print i
    i=i+1
file = open("ThemeDefintion.csv","w") 
file.write(text) 
file.close()

all_scored = []
for corpora in corpus:
    all_scored.append(Lda_Model.get_document_topics(corpora, minimum_probability=0))
    
all_scored = pd.DataFrame(all_scored)
all_scored[['1_topic_prob', '1_topic']] = all_scored[0].apply(pd.Series)
all_scored[['2_topic_prob', '2_topic']] = all_scored[1].apply(pd.Series)
all_scored[['3_topic_prob', '3_topic']] = all_scored[2].apply(pd.Series)
all_scored[['4_topic_prob', '4_topic']] = all_scored[3].apply(pd.Series)
#all_scored[['5_topic_prob', '5_topic']] = all_scored[4].apply(pd.Series)
#all_scored[['6_topic_prob', '6_topic']] = all_scored[5].apply(pd.Series)
#all_scored[['7_topic_prob', '7_topic']] = all_scored[6].apply(pd.Series)
#all_scored[['8_topic_prob', '8_topic']] = all_scored[7].apply(pd.Series)
#all_scored[['9_topic_prob', '9_topic']] = all_scored[8].apply(pd.Series)
#all_scored[['10_topic_prob','10_topic']] = all_scored[9].apply(pd.Series)

all_scored = all_scored[["1_topic", "2_topic", "3_topic", "4_topic"]]
print all_scored

all_scored.to_csv(os.path.join("C:\Users\Shubhi Tyagi", "TopicDistibution.csv"), index=False)

ax=all_scored[:10].plot(kind='bar',legend='TRUE', title='Topic Proportion per Document')
ax.set(xlabel="Documents", ylabel="Topic Proportions")

#Hellingers Distance

import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean


_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64


def hellinger1(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def hellinger3(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2
    
#Creating Similarity Matrix
SimMatrix=[]
for i in range(len(all_scored)):
    n = all_scored[i:i+1]
    #print n
    simList=[]
    for j in range(len(all_scored)):
        r = all_scored[j:j+1]
        #print r
        sim=hellinger2(n,r)
        #print sim
        simList.append(sim)
    #print simList
    print i
    #print simList
    #SimMatrix[i]=simList
    SimMatrix.append(simList)
    #break
pd.DataFrame(SimMatrix)
print SimMatrix

np.savetxt("C:\Users\Shubhi Tyagi\SimilarityMatrix.csv", SimMatrix, delimiter=",")

topic_dist = all_scored.loc[0,:] 
print max(topic_dist)
top_n = 1
output = pd.DataFrame({n: all_scored.T[col].nlargest(top_n).index.tolist() for n, col in enumerate( all_scored.T)}).T
filename = 'C:\Users\Shubhi Tyagi\FourTopic-Output.csv'
output.to_csv(filename, index=False, encoding='utf-8')
