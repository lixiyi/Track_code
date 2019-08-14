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
INDEX_NAME = "news_base"

setting = {
	"index": {
		"max_terms_count": 20480,
		"similarity": {
			"my_bm25": {
				"type": "BM25",
				"b": 0.75,
				"k1": 1.2
			}
		}
	}
}
es.indices.close(index=INDEX_NAME)
es.indices.put_settings(index=INDEX_NAME, body=setting)
es.indices.open(index=INDEX_NAME)

