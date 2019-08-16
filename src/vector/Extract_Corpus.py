import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import json
import re
from tqdm import tqdm
from elasticsearch import Elasticsearch

# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')
es = Elasticsearch(port=7200)
INDEX_NAME = "news_stem"


def gen_train_corpus():
    dsl = {
        'query': {
            'match': {
                'id': '*'
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=dsl)
    nums = len(res['hits']['hits'])
    print(nums)


gen_train_corpus()



