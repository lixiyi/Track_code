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


def extract_body(args = None):
    contents = args[0]
    body = ''
    for p in contents:
        if type(p).__name__ == 'dict':
            if 'subtype' in p and p['subtype'] == 'paragraph':
                paragraph = p['content'].strip()
                # Replace <.*?> with ""
                paragraph = re.sub(r'<.*?>', '', paragraph)
                body += ' ' + paragraph
    return body


def filter_kicker(doc):
    # Filter by kicker
    filter_kicker = {"Opinion": 1, "Letters to the Editor": 1, "The Post's View": 1}
    topic_name = ''
    for li in doc['contents']:
        if type(li).__name__ == 'dict':
            if 'type' in li and li['type'] == 'kicker':
                # skip filter kickers
                topic_name = li['content']
                if topic_name in filter_kicker.keys():
                    return False
    return topic_name


def process_washington_post(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            obj = json.loads(line)
            obj['kicker'] = filter_kicker(obj)
            if obj['kicker'] is False:
                continue
            obj['body'] = extract_body([obj['contents']])
            obj['body'] = str(obj['body']).lower()
            del obj['contents']
            obj['title_body'] = str(obj['title']) + ' ' + str(obj['body'])
            obj['title_body'] = obj['title_body'].lower()
            obj['title_author_date'] = str(obj['title']) + ' ' + str(obj['author']) + ' ' + str(obj['published_date'])
            obj['title_author_date'] = obj['title_author_date'].lower()
            obj['title'] = str(obj['title']).lower()
            doc = json.dumps(obj)
            # insert data
            res = es.index(index='news', id=obj['id'], body=doc)


# put all the news into elasticsearch
def init_es():
    # create index
    setting = {
		"news": {
        	"settings": {
        	    "similarity": {
        	        "my_bm25": {
        	            "type": "BM25",
        	            "b": 0.75,
        	            "k1": 1.5
        	        }
        	    }
        	}
		}
    }
    mapping = {
        'properties': {
            'id': {
                'type': 'keyword'
            },
            'article_url': {
                'type': 'keyword'
            },
            'title': {
                'type': 'text',
                "similarity": "my_bm25"
            },
            'author': {
                'type': 'keyword'
            },
            'published_date': {
                'type': 'keyword'
            },
            'body': {
                'type': 'text',
                "similarity": "my_bm25"
            },
            'title_body': {
                'type': 'text',
                "similarity": "my_bm25"
            },
            'kicker': {
                'type': 'keyword'
            },
            'title_author_date': {
                'type': 'keyword'
            },
            # 'contents': {
            #     'type': 'keyword'
            # },
            'type': {
                'type': 'keyword'
            },
            'source': {
                'type': 'keyword'
            }
        }
    }
    es.indices.delete(index='news', ignore=[400, 404])
    es.indices.create(index='news', ignore=400)
    es.indices.put_settings(index='news', body=setting)
    result = es.indices.put_mapping(index='news', body=mapping)
    # add all the file into elasticsearch
    process_washington_post(path_mp['DataPath'] + path_mp['WashingtonPost'])


init_es()
