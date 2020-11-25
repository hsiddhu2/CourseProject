import gensim
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.textcorpus import remove_short
import numpy as np


tokenizer = RegexpTokenizer(r'\w+')
f = open("data/BushGore.txt", "r")
documents = []
i = 0
datedocument = {}
topic_number = 10
for x in f:
    line = x
    dateline = line.split(":")
    document = dateline[1].lower()
    documents.append(tokenizer.tokenize(document))
    arr = []
    date = dateline[0]
    if datedocument.get(date)!=None:
        arr = datedocument.get(date)
    arr.append(i)
    datedocument[date] = arr
    i=i+1
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(remove_short(gensim.corpora.textcorpus.remove_stopwords(text),minsize = 2)) for text in documents]
model = gensim.models.LdaModel(corpus, id2word=dictionary,
                               alpha='auto',
                               num_topics=10,
                               passes=5)

dates = datedocument.keys()
doc_ids = []
topics_str = []
i = 0
TS = np.zeros((len(dates),topic_number))
i = 0
print(model.get_document_topics(corpus)[70])
for x in dates:
    doc_ids = datedocument.get(x)
    for k in doc_ids:
        topics_str = model.get_document_topics(corpus)[k]
        for y in topics_str:
            topic_index = y[0]
            topic_prob = y[1]
            TS[i][topic_index] = TS[i][topic_index] + topic_prob
    i = i + 1
                       
print(TS)       
print(model.get_topics())
print(model.print_topics())
print(model.get_document_topics(corpus)[0])