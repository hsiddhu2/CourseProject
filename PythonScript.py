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
import math
import matplotlib.pyplot as plt


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
            documents.append(preprossed_document)
            i = i+1
        
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    return corpus , datedocument,dictionary

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

def runLda(corpus,ntopics,dictionary,prior,mu):
    model = gensim.models.LdaModel(corpus, id2word=dictionary,
                               alpha='auto',eta = prior,
                               num_topics=ntopics,
                               passes=5,decay = mu)
    return model

def calculateTs(model,datedocument,number_topics,corpus):
    dates = datedocument.keys()
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
    lag_pvalue = []
    alpha = 0.05
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
    print("Found "+str(total_significant_topic)+" significant topics")
    return significantTopics

def calculateWS(corpus,datedocument,vocabulary_size):
    dates = datedocument.keys()
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
    probM = 0.1
    prior_list = []
    for x in datedocument:
        price = goreNormProbability.get(x)
        if price == None:
            price = 0.5
        pricearr.append(price)
    arr[:,0] = pricearr
    total_purity = 0
    sig_topic= 0
    total_confidence = 0
    total_significant_words = 0
    for i in significantTopics:
        print(model.print_topic(i))
        total_prob = 0
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
                if total_prob <= probM and dictionary[id]!="bush" and dictionary[id]!="gore" and dictionary[id]!="mr":
                    total_confidence = total_confidence + ((1 - prob) * 100)
                    pearsonCorelation = calculatePearsonCorelation(pricearr,WS[:,id])
                    word_prob = (id,pvalue)
                    if(pearsonCorelation[0]<0):
                        negative_words.append(word_prob)
                    else:
                        positive_words.append(word_prob)
                else:
                    break
        total_sig_words = len(negative_words)+len(positive_words)
        total_significant_words = total_significant_words + total_sig_words;
        print("Total significant words in topic "+str(i)+" are:"+str(total_sig_words))
        print("Total positive words:"+str(len(positive_words)))
        print("Total negative words:"+str(len(negative_words)))
        topic_Word_prior = np.zeros((vocabulary_size))
        denom = 0
        if total_sig_words != 0:
            nprob = len(negative_words) / total_sig_words
            pprob = len(positive_words)/ total_sig_words
            if nprob == 0:
                entropy = pprob * math.log(pprob)
            elif pprob == 0:
                entropy = nprob * math.log(nprob)
            else:
                entropy = (nprob * math.log(nprob)) + (pprob * math.log(pprob))
            purity = 100 + (100 * entropy)
            total_purity = total_purity + purity
            sig_topic = sig_topic + 1
            print("Purity of topic "+str(i)+" is:"+str(purity))
            if len(positive_words)!=0 and len(positive_words)/total_sig_words < 0.1:
                for x in negative_words:
                    word_prior = (100 *(1-x[1])) - 95
                    topic_Word_prior[x[0]] =  word_prior
                    denom = denom + word_prior
                prior_list.append(topic_Word_prior/word_prior)
            elif len(positive_words)!=0 and len(negative_words)/total_sig_words < 0.1 :
                for x in positive_words:
                    word_prior = (100 *(1-x[1])) - 95
                    topic_Word_prior[x[0]] =  word_prior
                    denom = denom + word_prior
                prior_list.append(topic_Word_prior/word_prior)
            elif len(negative_words)==0 and len(positive_words)==0:
                print("No significant words found")
            else:
                for x in negative_words:
                    word_prior = (100 *(1-x[1])) - 95
                    topic_Word_prior[x[0]] =  word_prior
                    denom = denom + word_prior
                prior_list.append(topic_Word_prior/word_prior)
                denom = 0
                for x in positive_words:
                    word_prior = (100 *(1-x[1])) - 95
                    topic_Word_prior[x[0]] =  word_prior
                    denom = denom + word_prior
                prior_list.append(topic_Word_prior/word_prior)
    avg_purity = total_purity/sig_topic
    avg_confidence = total_confidence / total_significant_words
    print("Average Purity:"+str(avg_purity))
    prior = np.zeros((number_topics,vocabulary_size))
    j = 0
    for x in prior_list:
        for i in range(len(x)):
            prior[j,i] = x[i]
        j = j+1
        
    return prior,avg_purity,avg_confidence

            
def main():
    corpus,datedocument,dictionary = preprocessData()
    print(len(corpus))
    goreNormProbability = readStockPrice()
    vocabulary_size = len(dictionary)
    print("Vocabulary Size:"+str(vocabulary_size))
    mu = 0.6
    print("----------For varying Mu values----------")
    for j in range(5):
        avg_purity_list = []
        iteration_list = []
        avg_conf_list = []
        number_topics = 10
        prior = 0
        print("-----For Mu:"+str(mu)+"-----")
        for i in range(5):
            itr = i + 1
            iteration_list.append(itr)
            print("---Iteration:"+str(i+1)+"---")
            print("Number of topics:"+str(number_topics))
            model = runLda(corpus,number_topics,dictionary,prior,mu)
            topic_word_prob = model.get_topics()
            TS = calculateTs(model,datedocument,number_topics,corpus)
            #print(TS)
            WS = calculateWS(corpus,datedocument,vocabulary_size)
            #print(WS)
            significantTopics = findSignificantTopics(TS,goreNormProbability,datedocument,number_topics)
            print("-----Significant Topics are----")
            print(significantTopics)
            if number_topics<30:
                number_topics = number_topics + 10
            prior,avg_purity,avg_confidence = calculatePrior(WS,significantTopics,model,datedocument,goreNormProbability,vocabulary_size,number_topics,dictionary)
            avg_purity_list.append(avg_purity)
            avg_conf_list.append(avg_confidence)
        plt.subplot(2, 2, 1)
        plt.plot(iteration_list,avg_purity_list,marker='o',label = "Mu:"+str(mu))
        plt.subplot(2, 2, 2)
        plt.plot(iteration_list, avg_conf_list, marker='o', label="Mu:" + str(mu))
        mu = mu + 0.1
    plt.subplot(2, 2, 1)
    plt.title('Avg. Purity for Different Mu')
    plt.ylabel('Avg. Purity')
    plt.xlabel('Iteration')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title('Avg. Causality Confidence for Different Mu')
    plt.ylabel('Avg. Causality Confidence')
    plt.xlabel('Iteration')
    plt.legend()
    mu = 1
    print("----------For varying topic numbers----------")
    for j in range(5):
        avg_purity_list = []
        iteration_list = []
        avg_conf_list = []
        if j == 4:
            number_topics = 10
        else:
            number_topics = (j + 1) * 10
        prior = 0
        if j == 4:
            print("-----For tn:vartn-----")
        else:
            print("-----For tn:" + str((j + 1) * 10) + "-----")
        for i in range(5):
            itr = i + 1
            iteration_list.append(itr)
            print("---Iteration:" + str(i + 1) + "---")
            print("Number of topics:" + str(number_topics))
            model = runLda(corpus, number_topics, dictionary, prior, mu)
            topic_word_prob = model.get_topics()
            TS = calculateTs(model, datedocument, number_topics, corpus)
            # print(TS)
            WS = calculateWS(corpus, datedocument, vocabulary_size)
            # print(WS)
            significantTopics = findSignificantTopics(TS, goreNormProbability, datedocument, number_topics)
            print("-----Significant Topics are-----")
            print(significantTopics)
            if j == 4:
                if number_topics < 30:
                    number_topics = number_topics + 10
            prior, avg_purity, avg_confidence = calculatePrior(WS, significantTopics, model, datedocument,
                                                               goreNormProbability, vocabulary_size, number_topics,
                                                               dictionary)
            avg_purity_list.append(avg_purity)
            avg_conf_list.append(avg_confidence)
        if j == 4:
            plt.subplot(2, 2, 3)
            plt.plot(iteration_list, avg_purity_list, marker='o', label="tn:var tn")
            plt.subplot(2, 2, 4)
            plt.plot(iteration_list, avg_conf_list, marker='o', label="tn:var tn")
        else:
            plt.subplot(2, 2, 3)
            plt.plot(iteration_list, avg_purity_list, marker='o', label="tn:" + str(number_topics))
            plt.subplot(2, 2, 4)
            plt.plot(iteration_list, avg_conf_list, marker='o', label="tn:" + str(number_topics))
    plt.subplot(2, 2, 3)
    plt.title('Avg. Purity for Different tn')
    plt.ylabel('Avg. Purity')
    plt.xlabel('Iteration')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title('Avg. Causality Confidence for Different tn')
    plt.ylabel('Avg. Causality Confidence ')
    plt.xlabel('Iteration')
    plt.legend()

    print("---------Result----------")
    i = 0
    for x in significantTopics:
        top_words = ""
        words_prob = model.print_topic(x,10).split("+")
        for y in words_prob:
            words_prob_arr = y.split("*")
            word = words_prob_arr[1]
            word = word.replace('"',"")
            if "bush" not in word and "gore" not in word and "mr" not in word:
                top_words = top_words + word+" "
        print(str(i)+". "+top_words)
        i = i + 1
    plt.show()


if __name__ == "__main__":
    main()