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
    for li in doc['contents']:
        if type(li).__name__ == 'dict':
            if 'type' in li and li['type'] == 'kicker':
                # skip filter kickers
                topic_name = li['content']
                if topic_name in filter_kicker.keys():
                    return False
    return topic_name


def filter_doc(doc, date, similar_doc):
    # Filter by date
    doc_title = doc['title']
    doc_author = doc['author']
    doc_date = doc['published_date']
    if doc_date is not None and date is not None and int(doc_date) > int(date):
        return False
    # Filter by date + title + author
    rep_key = ''
    if doc_title is not None:
        rep_key += doc_title
    if doc_author is not None:
        rep_key += '#' + doc_author
    if doc_date is not None:
        rep_key += '#' + str(doc_date)
    if rep_key in similar_doc:
        return False
    similar_doc[rep_key] = 1
    return True


def process_washington_post(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            obj['kicker'] = filter_kicker(doc)
            if obj['kicker'] is False:
                continue
            obj['body'] = extract_body([obj['contents']])
            obj['title_body'] = obj['body'] + obj['title']
            doc = json.dumps(obj)
            # insert data
            res = es.index(index='news', id=obj['id'], body=doc)


# put all the news into elasticsearch
def init_es():
    # create index
    mapping = {
        'properties': {
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
            'body': {
                'type': 'text'
            },
            'title_body': {
                'type': 'text'
            },
            'kicker': {
                'type': 'keyword'
            },
            'contents': {
                'type': 'keyword'
            },
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
