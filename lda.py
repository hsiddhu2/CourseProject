import gensim
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer


tokenizer = RegexpTokenizer(r'\w+')
f = open("data.txt", "r")
documents = []
i = 0
datedocument = {}
for x in f:
    line = x
    dateline = line.split(":")
    documents.append(tokenizer.tokenize(dateline[1].lower()))
    arr = []
    date = dateline[0]
    if datedocument.get(date)!=None:
        arr = datedocument.get(date)
    arr.append(i)
    datedocument[date] = arr
    i=i+1
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]
model = gensim.models.LdaModel(corpus, id2word=dictionary,
                               alpha=0.1,
                               num_topics=10,
                               passes=5)
print(model.get_topics())
print(model.get_document_topics(corpus)[0])