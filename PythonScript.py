import gensim
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.textcorpus import remove_short,remove_stopwords
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string,strip_punctuation,strip_numeric,stem_text,strip_short
from scipy import stats


def preprocessData():
    tokenizer = RegexpTokenizer(r'\w+')
    f = open("Data/BushGore.txt", "r")
    documents = []
    datedocument = {}
    i = 0
    for x in f:
        line = x
        dateline = line.split(":")
        document = dateline[1].lower() 
        arr = []
        date = dateline[0]
        if datedocument.get(date)!=None:
            arr = datedocument.get(date)
        arr.append(i)
        datedocument[date] = arr
        filter = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_numeric,lambda x: strip_short(x,2)]
        preprossed_document = preprocess_string(document,filters=filter)
        duplicate = documents.count(preprossed_document)
        if duplicate == 0:
            #documents.append(tokenizer.tokenize(document))
            documents.append(preprossed_document)
            i=i+1
        
    dictionary = corpora.Dictionary(documents)
    #dict = dictionary.token2id
    #print(dictionary.token2id)
    #print(len(documents))
    #print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in documents]
    return corpus,datedocument,dictionary

def readStockPrice():
    goreNormalizedProbability = {}
    with open( 'Data/StockPrices.txt' ) as f:
        for line1, line2, line3 in itertools.zip_longest( *[f] * 3 ):
            dem = line1.split()
            rep = line3.split()
            denominator = float( dem[-1] ) + float( rep[-1] )
            date = dem[0]
            normProb = float( dem[-1] ) / denominator
            goreNormalizedProbability.update( {date: normProb} )
    return goreNormalizedProbability

def runLda(corpus,ntopics,dictionary,prior):
    model = gensim.models.LdaModel(corpus, id2word=dictionary,
                               alpha='auto',eta = prior,
                               num_topics=ntopics,
                               passes=5)
    return model
    
def calculateTs(model,datedocument,number_topics,corpus):
    dates = datedocument.keys()
    doc_ids = []
    topics_str = []
    i = 0
    TS = np.zeros((len(dates),number_topics))
    i = 0
    for x in dates:
        doc_ids = datedocument.get(x)
        for k in doc_ids:
            topics_str = model.get_document_topics(corpus)[k]
            for y in topics_str:
                topic_index = y[0]
                topic_prob = y[1]
                TS[i][topic_index] = TS[i][topic_index] + topic_prob
        i = i + 1
    return TS

def runGrangerTest(arr):
    testResult = grangercausalitytests(arr, 5,verbose = False)
    return testResult

def calculatePearsonCorelation(x,y):
    return stats.pearsonr(x, y)

def isSignificant(arr):
    result = runGrangerTest(arr)
    max_lag = 5
    lag_pvalue = []
    alpha = 0.05
    alpha1 = 0.95
    for i in range(1,6):
        lag = result.get(i)
        test = lag[0]
        ssr_ftest = test.get("ssr_ftest")
        pvalue = ssr_ftest[1]
        lag_pvalue.append(pvalue)
    min_pvalue = lag_pvalue[1]
    min_lag = 1
    for x in range(len(lag_pvalue)):
        if lag_pvalue[x] < min_pvalue:
            min_pvalue = lag_pvalue[x]
            min_lag = x + 1
    if min_pvalue <= alpha:
        return min_pvalue,min_lag
    else:
        return None,None
    

def findSignificantTopics(TS,goreNormProbability,datedocument,number_topics):
    arr = np.zeros((len(TS[:,0]),2))
    pricearr = []
    significantTopics = []
    for x in datedocument:
        price = goreNormProbability.get(x)
        if price == None:
            price = 0.5
        pricearr.append(price)
    arr[:,0] = pricearr
    total_significant_topic = 0
    for i in range(number_topics):
        arr[:,1] = TS[:,i]
        pvalue,lag = isSignificant(arr)
        if pvalue!=None and lag!=None:
            total_significant_topic = total_significant_topic + 1
            significantTopics.append(i)
        else:
            print("Topic "+str(i)+" is not significant.")
    print("Found "+str(total_significant_topic)+" significant topics")
    return significantTopics

def calculateWS(corpus,datedocument,vocabulary_size):
    dates = datedocument.keys()
    doc_ids = []
    WS = np.zeros((len(dates),vocabulary_size))
    z = 0
    for x in dates:
        doc_ids = datedocument.get(x)
        for i in doc_ids:
            doc = corpus[i]
            for j in doc:
                word_frequency = j
                word_id = word_frequency[0]
                word_count = word_frequency[1]
                WS[z][word_id] = WS[z][word_id] + word_count
        z = z+1
    return WS
        
def calculatePrior(WS,significantTopics,model,datedocument,goreNormProbability,vocabulary_size,number_topics,dictionary):
    arr = np.zeros((len(WS[:,0]),2))
    pricearr = []
    num_words = 200
    probM = 0.4
    top_words = []

    top_significant_words=[]
    prior_list = []
    topic_index = 0
    for x in datedocument:
        price = goreNormProbability.get(x)
        if price == None:
            price = 0.5
        pricearr.append(price)
    arr[:,0] = pricearr
    for i in significantTopics:
        total_prob = 0
        top_words = []
        top_word_tuple = model.get_topic_terms(i,100)
        for x in top_word_tuple:
            prob = x[1]
            total_prob = total_prob + prob
            if total_prob <= probM:
                top_words.append(x[0])
        positive_words = []
        negative_words =[]
        for y in top_words:
            arr[:,1] = WS[:,y]
            pvalue,lag = isSignificant(arr)
            if pvalue!=None and lag!=None:
                pearsonCorelation = calculatePearsonCorelation(pricearr,WS[:,y])
                word_prob = (y,pvalue)
                if(pearsonCorelation[0]<0):
                    negative_words.append(word_prob)
                else:
                    positive_words.append(word_prob)
        total_sig_words = len(negative_words)+len(positive_words)
        print("Total significant words in "+str(i)+" are:"+str(total_sig_words))
        print("Total positive words:"+str(len(positive_words)))
        print("Total negative words:"+str(len(negative_words)))
        topic_Word_prior = np.zeros((vocabulary_size))
        topic_Word_prior1 = np.zeros((vocabulary_size))
        denom = 0
        if len(positive_words)!=0 and len(positive_words)/total_sig_words < 0.1:
            for x in negative_words:
                print(dictionary.id2token[x[0]])
                word_prior = (100 *(1- x[1])) - 95
                topic_Word_prior[x[0]] =  word_prior
                denom = denom + word_prior
            prior_list.append(topic_Word_prior/word_prior)
        elif len(positive_words)!=0 and len(negative_words)/total_sig_words < 0.1 :
            for x in positive_words:
                print(dictionary.id2token[x[0]])
                word_prior = (100 *(1- x[1])) - 95
                topic_Word_prior[x[0]] =  word_prior
                denom = denom + word_prior
            prior_list.append(topic_Word_prior/word_prior)
        elif len(negative_words)==0 and len(positive_words)==0:
            print("No significant words found")
        else:
            for x in negative_words:
                print(dictionary.id2token[x[0]])
                word_prior = (100 *(1- x[1])) - 95
                topic_Word_prior[x[0]] =  word_prior
                denom = denom + word_prior
            prior_list.append(topic_Word_prior/word_prior)
            denom = 0
            for x in positive_words:
                print(dictionary.id2token[x[0]])
                word_prior = (100 *(1- x[1])) - 95
                topic_Word_prior[x[0]] =  word_prior
                denom = denom + word_prior
            prior_list.append(topic_Word_prior/word_prior)
    #prior = np.zeros((len(prior_list),vocabulary_size))
    prior = np.zeros((number_topics,vocabulary_size))
    j = 0
    for x in prior_list:
        for i in range(len(x)):
            prior[j,i] = x[i]
        j = j+1
        
    return prior

def calculatePrior1(WS,significantTopics,model,datedocument,goreNormProbability,vocabulary_size,number_topics,dictionary):
    arr = np.zeros((len(WS[:,0]),2))
    pricearr = []
    num_words = 200
    probM = 0.1
    top_words = []

    top_significant_words=[]
    prior_list = []
    topic_index = 0
    for x in datedocument:
        price = goreNormProbability.get(x)
        if price == None:
            price = 0.5
        pricearr.append(price)
    arr[:,0] = pricearr
    for i in significantTopics:
        print(model.print_topic(i))
        total_prob = 0
        top_words = []
        top_word_tuple = model.get_topic_terms(i,vocabulary_size)
        positive_words = []
        negative_words =[]
        for y in top_word_tuple:
            id = y[0]
            arr[:,1] = WS[:,id]
            pvalue,lag = isSignificant(arr)
            if pvalue!=None and lag!=None:
                prob = y[1]
                total_prob = total_prob + prob
                if total_prob <= probM:
                    pearsonCorelation = calculatePearsonCorelation(pricearr,WS[:,id])
                    word_prob = (id,pvalue)
                    if(pearsonCorelation[0]<0):
                        negative_words.append(word_prob)
                    else:
                        positive_words.append(word_prob)
                else:
                    break
        total_sig_words = len(negative_words)+len(positive_words)
        print("Total significant words in "+str(i)+" are:"+str(total_sig_words))
        print("Total positive words:"+str(len(positive_words)))
        print("Total negative words:"+str(len(negative_words)))
        topic_Word_prior = np.zeros((vocabulary_size))
        topic_Word_prior1 = np.zeros((vocabulary_size))
        denom = 0
        if len(positive_words)!=0 and len(positive_words)/total_sig_words < 0.1:
            for x in negative_words:
                print(dictionary.id2token[x[0]])
                word_prior = (100 *(1-x[1])) - 95
                topic_Word_prior[x[0]] =  word_prior
                denom = denom + word_prior
            prior_list.append(topic_Word_prior/word_prior)
        elif len(positive_words)!=0 and len(negative_words)/total_sig_words < 0.1 :
            for x in positive_words:
                print(dictionary.id2token[x[0]])
                word_prior = (100 *(1-x[1])) - 95
                topic_Word_prior[x[0]] =  word_prior
                denom = denom + word_prior
            prior_list.append(topic_Word_prior/word_prior)
        elif len(negative_words)==0 and len(positive_words)==0:
            print("No significant words found")
        else:
            for x in negative_words:
                print(dictionary.id2token[x[0]])
                word_prior = (100 *(1-x[1])) - 95
                topic_Word_prior[x[0]] =  word_prior
                denom = denom + word_prior
            prior_list.append(topic_Word_prior/word_prior)
            denom = 0
            for x in positive_words:
                print(dictionary.id2token[x[0]])
                word_prior = (100 *(1-x[1])) - 95
                topic_Word_prior[x[0]] =  word_prior
                denom = denom + word_prior
            prior_list.append(topic_Word_prior/word_prior)
    #prior = np.zeros((len(prior_list),vocabulary_size))
    prior = np.zeros((number_topics,vocabulary_size))
    j = 0
    for x in prior_list:
        for i in range(len(x)):
            prior[j,i] = x[i]
        j = j+1
        
    return prior

            
def main():
    corpus,datedocument,dictionary = preprocessData()
    print(len(corpus))
    #print(corpus.get_texts())
    #print(datedocument)
    goreNormProbability = readStockPrice()
    #print( goreNormProbability )
    number_topics = 10
    vocabulary_size = len(dictionary)
    print("Vocabulary Size:"+str(vocabulary_size))
    prior = 0
    for i in range(5):
        print("---Iteration:"+str(i)+"---")
        print("Number of topics:"+str(number_topics))
        model = runLda(corpus,number_topics,dictionary,prior)
        print(model.print_topics(30))
        topic_word_prob = model.get_topics()
        TS = calculateTs(model,datedocument,number_topics,corpus)
        print(TS)
        WS = calculateWS(corpus,datedocument,vocabulary_size)
        print(WS)
        significantTopics = findSignificantTopics(TS,goreNormProbability,datedocument,number_topics)
        print("-----Significant Topics are----")
        print(significantTopics)
        if number_topics<30:
            number_topics = number_topics + 10
        prior = calculatePrior1(WS,significantTopics,model,datedocument,goreNormProbability,vocabulary_size,number_topics,dictionary)
        
        

if __name__ == "__main__":
    main()