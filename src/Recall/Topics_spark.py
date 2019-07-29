import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
from pyspark import SparkContext, SparkConf
import numpy as np


path_mp = cfg.get_path_conf('../path.cfg')


def topics_index_single(line):
	obj = json.loads(line)
	contents = obj['contents']
	doc_id = obj['id']
	topic_name = ''
	for li in contents:
		if type(li).__name__ == 'dict':
			if 'type' in li and li['type'] == 'kicker':
				contents = li['content']
	st = set()
	st.add(doc_id)
	if topic_name == '':
		return ''
	return (topic_name, st)


# create inverted list for topis
# No args
# output: (topic, [doc line numbers])
def topics_index(args = None):
	SparkContext.getOrCreate().stop()
	conf = SparkConf().setMaster("local[*]").setAppName("tfidf_index")
	sc = SparkContext(conf=conf)
	WashingtonPost = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost'])
	WashingtonPost.flatMap(lambda line: topics_index_single(line)) \
		.filter(lambda x:x != '') \
		.reduceByKey(lambda a, b: a | b) \
		.map(lambda t: str(t[0]) + ' ' + ' '.join(t[1])) \
		.saveAsTextFile(cfg.OUTPUT + 'topics_index')


if __name__ == "__main__":
	getattr(__import__('Topics'), sys.argv[1])(sys.argv[2:])



