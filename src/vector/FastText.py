import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import jieba
import re
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from stanfordcorenlp import StanfordCoreNLP
from gensim.models import KeyedVectors

# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')
# es = Elasticsearch(port=7200)
INDEX_NAME = "news_stem"

