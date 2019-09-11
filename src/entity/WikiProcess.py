import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import json
import re
from tqdm import tqdm
from elasticsearch import Elasticsearch
from nltk.stem.porter import *

# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')
es = Elasticsearch(port=7200)
stemmer = PorterStemmer()
INDEX_NAME = "news_wiki"


def process_wiki(filepath):
    for p in tqdm(iter_pages(open(filepath, 'rb'))):
        obj = {}
        obj['page_name'] = str(p.page_name).lower()
        skeleton = p.skeleton
        out = ""
        for para in skeleton:
            out += para.get_text().strip()
        obj['body'] = out.lower()
        doc = json.dumps(obj)
        # insert data
        res = es.index(index=INDEX_NAME, body=doc)


# put all the news into elasticsearch
def init_es():
    # create index
    setting = {
        "index": {
            "similarity": {
                "my_bm25": {
                    "type": "BM25",
                    "b": 0.75,
                    "k1": 1.2
                }
            }
        }
    }
    mapping = {
        'properties': {
            'page_name': {
                'type': 'text',
                "similarity": "my_bm25"
            },
            'body': {
                'type': 'text',
                "similarity": "my_bm25",
            }
        }
    }
    create_index_body = {
        "settings": setting,
        "mappings": mapping
    }
    es.indices.delete(index=INDEX_NAME, ignore=[400, 404])
    es.indices.create(index=INDEX_NAME, body=create_index_body, ignore=400)
    # add all the file into elasticsearch
    process_wiki(cfg.WIKIDUMP)


init_es()