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
import json
from nltk.stem.porter import *


# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')
es = Elasticsearch(port=7200)
nlp = StanfordCoreNLP('http://localhost', port=7100)
stemmer = PorterStemmer()
INDEX_NAME = "news_alpha"
WIKI_INDEX = "news_wiki"


def test_entity_ranking():
    # stop words
    stop_words = {}
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        for w in f:
            w = w[:-1]
            stop_words[w] = 1
    print('stop words loaded.')
    # test case: topic_id, entity_id list
    case_mp = {}
    with open(path_mp['DataPath'] + path_mp['entities'], 'r', encoding='utf-8') as f:
        li = []
        mp = {}
        topic_id = ''
        for line in f:
            topic_id_tmp = re.search(r'<num>.*?</num>', line)
            if topic_id_tmp is not None:
                if len(li) > 0:
                    case_mp[topic_id] = li
                    li = []
                topic_id = topic_id_tmp
                topic_id = topic_id.group(0)[5+9:-7]
            entity_id = re.search(r'<id>.*?</id>', line)
            if entity_id is not None:
                entity_id = entity_id.group(0)[5:-6]
                mp['id'] = entity_id
            mention = re.search(r'<mention>.*?</mention>', line)
            if mention is not None:
                mention = mention.group(0)[9:-10]
                mp['mention'] = mention
            link = re.search(r'<link>.*?</link>', line)
            if link is not None:
                link = link.group(0)[6:-7]
                mp['link'] = link
                li.append(mp)
                mp = {}
    print('test case loaded.')
    for topic_id in case_mp.keys():
        for entity in case_mp[topic_id]:
            print(entity)

