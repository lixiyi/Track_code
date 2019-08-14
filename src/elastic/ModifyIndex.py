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

setting = {
	"index": {
		"similarity": {
			"my_bm25": {
				"type": "BM25",
				"b": 0.75,
				"k1": 1.5
			}
		}
	}
}
es.indices.close(index="news")
es.indices.put_settings(index='news', body=setting)
es.indices.open(index="news")

