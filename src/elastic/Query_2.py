import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import jieba
import re
import numpy as np
from elasticsearch import Elasticsearch
import src.elastic.xmlhandler as xh

# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')
es = Elasticsearch()


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


def test_backgound_linking():
	# test case: doc_id, topic_id
	case_mp = {}
	with open(path_mp['DataPath'] + path_mp['topics'], 'r', encoding='utf-8') as f:
		li = []
		for line in f:
			topic_id = re.search(r'<num>.*?</num>', line)
			if topic_id is not None:
				topic_id = topic_id.group(0)[5+9:-7]
				li.append(topic_id)
			doc_id = re.search(r'<docid>.*?</docid>', line)
			if doc_id is not None:
				doc_id = doc_id.group(0)[7:-8]
				li.append(doc_id)
			if len(li) == 2:
				case_mp[li[1]] = li[0]
				li = []
	print('test case loaded.')
	with open('bresults.test', 'w', encoding='utf-8') as f1:
		for doc_id in case_mp:
			# search by docid to get the query
			dsl = {
				'query': {
					'match': {
						'id': doc_id
					}
				}
			}
			res = es.search(index='news', body=dsl)
			# print(res)
			doc = res['hits']['hits'][0]['_source']
			dt = doc['published_date']
			# query the doc
			dsl = {
				"query": {
					'bool': {
						'must': {
							'match': {'title_body': doc['title_body']}
						},
						"must_not": {"match": {"id": doc_id}},
						'filter': {
							"range": {"published_date": {"lt": dt}}
						}
					},
				}
			}
			res = es.search(index='news', body=dsl)
			res = res['hits']['hits']
			# output result.test file
			print('result:', len(res))
			cnt = 1
			for ri in res:
				out = []
				out.append(mp['num'].split(':')[1].strip())
				out.append('Q0')
				out.append(ri['_source']['id'])
				out.append(str(cnt))
				out.append(str(ri['_score']))
				out.append('ICTNET')
				ans = "\t".join(out) + "\n"
				f1.write(ans)
				cnt += 1
	return


test_backgound_linking()
