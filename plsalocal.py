import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from plsa import Corpus, Pipeline, Visualize
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA
from plsa.algorithms import PLSA

directory = 'data/test'
pipeline = Pipeline(*DEFAULT_PIPELINE)
tag = "textdata"
print(pipeline)
corpus = Corpus.from_xml(directory, pipeline,tag)
print(corpus)
n_topics = 2
plsa = PLSA(corpus, n_topics, True)
print(plsa)
result = plsa.fit()
print(plsa)
print(result.topic)

