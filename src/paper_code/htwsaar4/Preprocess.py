#!/usr/bin/env python3

import os
import json
import re
from elasticsearch import Elasticsearch

# get file path conf
path_mp = {}
with open(os.getcwd()+'/../../path.cfg', 'r', encoding='utf-8') as f:
    for line in f:
        li = line[:-1].split('=')
        path_mp[li[0]] = li[1]

es = Elasticsearch()


def process_washington_post(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:
            # if cnt == 1:
            #     break
            obj = json.loads(line)
            # for key in obj.keys():
            #     print(key, ':', obj[key])
            # print()
            contents = obj['contents']
            text = ""
            for li in contents:
                if type(li).__name__ == 'dict' and 'type' in li and li['type'] == 'sanitized_html':
                    content = li['content']
                    # remove html tags, lowercase
                    content = re.sub(r'<.*?>', '', content)
                    text += content.lower()
            obj['text'] = text
            del obj['contents']
            doc = json.dumps(obj)
            # insert data
            res = es.index(index='news', id=cnt, body=doc)
            print(cnt)
            cnt += 1


# put all the news into elasticsearch
def init_es():
    # create index
    mapping = {
        'properties': {
            'text': {
                'type': 'text'
            },
            'id': {
                'type': 'keyword'
            },
            'article_url': {
                'type': 'keyword'
            },
            'title': {
                'type': 'text'
            },
            'author': {
                'type': 'keyword'
            },
            'published_date': {
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
    result = es.indices.put_mapping(index='news', body=mapping)
    # add all the file into elasticsearch
    process_washington_post(path_mp['DataPath'] + path_mp['WashingtonPost'])


init_es()
