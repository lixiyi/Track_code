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
			'match_all': {}
		}
	}
	page = es.search(index=INDEX_NAME, size=1000, scroll='2m', body=dsl)
	sid = page['_scroll_id']
	scroll_size = page['_shards']['total']
	tot = 0
	# Start scrolling
	while scroll_size > 0:
		page = es.scroll(scroll_id=sid, scroll='2m')
		# Update the scroll ID
		sid = page['_scroll_id']
		# Get the number of results that we returned in the last scroll
		scroll_size = len(page['hits']['hits'])
		tot += scroll_size
	print(tot)


gen_train_corpus()



