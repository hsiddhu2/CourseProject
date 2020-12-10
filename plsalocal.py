import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append('..')

from plsa import Corpus, Pipeline, Visualize, visualize
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA

directory = 'Data/test'
pipeline = Pipeline(*DEFAULT_PIPELINE)
tag = "textdata"
print(pipeline)
corpus = Corpus.from_xml(directory, pipeline, tag)
print(corpus)
print(corpus.vocabulary)
n_topics = 30
plsa = PLSA(corpus, n_topics, True)
print(plsa)
result = plsa.fit()
print(plsa)
print(result.topic)
print(result.topic_given_doc)
print(result.word_given_topic[0][10])

fig = plt.figure(figsize=(9.4, 10))
_ = visualize.wordclouds(fig)
