import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import random
import numpy as np
from pyspark import SparkContext, SparkConf
import bm25
from stanfordcorenlp import StanfordCoreNLP


path_mp = cfg.get_path_conf('../path.cfg')
nlp = StanfordCoreNLP('http://localhost', port=7000)


def get_mapping(args=None):
	SparkContext.getOrCreate().stop()
	conf = SparkConf().setMaster("local[*]").setAppName("bm25") \
		.set("spark.executor.memory", "10g") \
		.set("spark.driver.maxResultSize", "10g") \
		.set("spark.cores.max", 10) \
		.set("spark.executor.cores", 10) \
		.set("spark.default.parallelism", 20)
	sc = SparkContext(conf=conf)
	# words df
	words_df = sc.textFile(cfg.OUTPUT + 'words_index.txt') \
		.filter(lambda line: line != '') \
		.map(lambda line: (line.split(' ')[0], len(line.split(' ')[1:]))) \
		.collectAsMap()
	words_df = sc.broadcast(words_df)
	print('words_df loaded.')
	# avgdl
	avgdl = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost']) \
		.map(lambda line: bm25.calc_doc_length(line)).sum()
	avgdl = avgdl * 1.0 / 595037
	print('avgdl loaded.')
	# WashingtonPost
	WashingtonPost = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost']) \
		.map(lambda line: bm25.return_doc(line)).collectAsMap()
	print('WashingtonPost loaded.')
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
	# answer: topic_id, (doc_id, rel)
	ans_mp = {}
	with open(path_mp['DataPath'] + path_mp['bqrels'], 'r', encoding='utf-8') as f:
		for line in f:
			li = line[:-1].split(' ')
			topic_id = li[0]
			doc_id = li[2]
			if topic_id not in ans_mp:
				ans_mp[topic_id] = []
			ans_mp[topic_id].append([doc_id, li[3]])
	# generate relevance map
	rel_mp = {}
	for cur_id in case_mp.keys():
		obj = WashingtonPost[cur_id]
		topic_id = case_mp[cur_id]
		body = bm25.extract_body([obj['contents']])
		# query (modify)
		tmp = nlp.ner(obj['title'] + ' ' + body)
		query = []
		for w, nn in tmp:
			if nn != 'O':
				query.append(w)
		rel_mp[topic_id] = []
		for doc_id, rel in ans_mp[topic_id]:
			score = bm25.calc_score(WashingtonPost[doc_id], words_df, query, avgdl, True)
			rel_mp[topic_id].append([score, rel])
	with open(cfg.OUTPUT + 'rel_mp.txt', 'w', encoding='utf-8') as f:
		f.write(json.dumps(rel_mp))


# modify bm25 score in col4 to rel
def transform(args=None):
	score2rel = {
		321: { 0: 190.3240164012551, 2: 394.65315265484287, 4: 376.6056238623155, 8: 414.1062907443666},
		336: { 0: 134.81449274235527, 2: 203.8672896832883, 4: 283.5119465326404, 8: 383.5119465326404},
		341: { 0: 376.6595325516464, 2: 372.1735933192249, 4: 455.2365348929022, 8: 701.5219762229876},
		347: { 0: 160.46553131599597, 2: 193.4215542473028, 4: 329.8962634603687, 8: 498.6746148794032},
		350: { 0: 305.65529277197413, 2: 430.4606264882297, 4: 530.4606264882298, 8: 630.4606264882298},
		362: { 0: 158.39868617559353, 2: 234.83125513642514, 4: 224.20748740673062, 8: 271.8105028011548},
		363: { 0: 362.0444390249729, 2: 343.74016559049284, 4: 344.84499903578427, 8: 643.4505182868702},
		367: { 0: 151.53365336916698, 2: 251.53365336916698, 4: 351.533653369167, 8: 451.533653369167},
		375: { 0: 473.0734567518298, 2: 604.6568518924449, 4: 535.5283312659295, 8: 891.5932802146623},
		378: { 0: 576.9156537011727, 2: 863.787905635933, 4: 961.9444998648114, 8: 1050.0385202737666},
		393: { 0: 426.7092803825617, 2: 355.13341390244295, 4: 615.9754891980101, 8: 680.5921964764111},
		397: { 0: 285.9197562100245, 2: 325.710730706131, 4: 664.6230707158121, 8: 757.0023187478041},
		400: { 0: 814.9682312845731, 2: 1865.0913023070693, 4: 2315.299485567116, 8: 2336.5488911647644},
		408: { 0: 149.55634760101304, 2: 224.51638550059945, 4: 276.39579008426256, 8: 272.57664889029127},
		414: { 0: 769.054502891747, 2: 1882.7845919078372, 4: 1907.559069822138, 8: 2247.0763965817505},
		422: { 0: 189.27327912494752, 2: 245.61649514568646, 4: 335.6450056551831, 8: 435.6450056551831},
		426: { 0: 197.04479220015156, 2: 247.59457519519378, 4: 290.07323092596107, 8: 219.51020560934214},
		427: { 0: 341.1316397869576, 2: 717.7372056177259, 4: 817.7372056177259, 8: 1081.9808433149378},
		433: { 0: 266.09749661708486, 2: 615.3829139897915, 4: 715.3829139897915, 8: 815.3829139897915},
		439: { 0: 213.25637206648491, 2: 235.12801271131556, 4: 313.0235894166311, 8: 317.8464419310266},
		442: { 0: 141.59735650007372, 2: 533.8408579469815, 4: 567.7690485972344, 8: 577.6193171597555},
		445: { 0: 175.50722100051837, 2: 196.88444678336586, 4: 277.8030481071621, 8: 518.832468680334},
		626: { 0: 302.69054539847997, 2: 427.3023900116349, 4: 527.3023900116349, 8: 627.3023900116349},
		646: { 0: 211.70898867623342, 2: 304.9733283592106, 4: 361.91938613356854, 8: 294.7123407296571},
		690: { 0: 491.6422482267002, 2: 845.6870627146677, 4: 856.8796892294478, 8: 1391.0739569531422},
		801: { 0: 319.66692579433527, 2: 570.854934953371, 4: 659.2011176779122, 8: 759.2011176779122},
		802: { 0: 529.923457566292, 2: 962.7522704865482, 4: 962.7522704865482, 8: 827.0410191879979},
		803: { 0: 340.4785067736041, 2: 369.56991189784185, 4: 508.3436874616435, 8: 608.3436874616435},
		804: { 0: 229.8035947545992, 2: 351.9197323537418, 4: 512.8217424937379, 8: 612.8217424937379},
		805: { 0: 200.8351046943789, 2: 290.444802182455, 4: 417.99351813698337, 8: 734.4657796842449},
		806: { 0: 96.47203141870418, 2: 136.74656537856535, 4: 252.8520664838264, 8: 352.85206648382643},
		807: { 0: 576.6634001908535, 2: 706.3204652815309, 4: 806.3204652815309, 8: 906.3204652815309},
		808: { 0: 582.4939524420805, 2: 733.0696696742957, 4: 777.8991740478144, 8: 127.43209445611467},
		809: { 0: 395.7468296062834, 2: 616.7314206237629, 4: 921.2589552350772, 8: 1021.2589552350772},
		810: { 0: 556.8266657908347, 2: 709.3994297501443, 4: 910.2935375805766, 8: 1048.764506282321},
		811: { 0: 412.08194617384714, 2: 490.06924779530027, 4: 757.1077528574567, 8: 1227.5979754695682},
		812: { 0: 521.1897274090428, 2: 507.5574271095392, 4: 671.5721080608469, 8: 686.3125888674159},
		813: { 0: 183.53081580397338, 2: 319.69689739022897, 4: 500.24403977822055, 8: 769.4043899610459},
		814: { 0: 420.7830036223449, 2: 375.824379213561, 4: 555.8670290546166, 8: 616.5779052022147},
		815: { 0: 657.6121594623701, 2: 1122.2274686764654, 4: 1162.7998269808704, 8: 1262.7998269808704},
		816: { 0: 228.97547916915318, 2: 310.6991701754521, 4: 420.3281222522314, 8: 431.99568923391627},
		817: { 0: 135.47604598822198, 2: 187.06178935335956, 4: 287.0617893533596, 8: 387.0617893533596},
		818: { 0: 269.15975689925097, 2: 452.0782116608944, 4: 719.070838300356, 8: 631.6428794619742},
		819: { 0: 226.90352185805455, 2: 349.05662815243386, 4: 360.43865370825006, 8: 334.11130458782776},
		820: { 0: 437.7025169149189, 2: 471.4058732096153, 4: 393.3245642313908, 8: 493.3245642313908},
		821: { 0: 263.8025330701786, 2: 252.26537814981316, 4: 268.6618637632625, 8: 228.77486553733073},
		822: { 0: 379.5600375408253, 2: 526.2721848306251, 4: 529.068466523686, 8: 659.251713274961},
		823: { 0: 303.5726637849037, 2: 423.71501146522496, 4: 388.6158309704532, 8: 331.35473284572697},
		824: { 0: 523.1658916060738, 2: 787.6400890461733, 4: 976.6110368018321, 8: 1076.6110368018321},
		825: { 0: 3326.9757915142236, 2: 4363.196148358353, 4: 8225.282175290897, 8: 10792.205398404578}
	}
	with open('/home/trec7/lianxiaoying/trec_eval.9.0/test/bresult.test', 'r', encoding='utf-8') as f:
		with open('/home/trec7/lianxiaoying/trec_eval.9.0/test/bresult.test1', 'w', encoding='utf-8') as out:
			for line in f:
				li = line[:].split('\t')
				topic_id = li[0]
				score = float(li[4])
				rel = 16
				if score <= score2rel[int(topic_id)][0]:
					rel = 0
				elif score <= score2rel[int(topic_id)][2]:
					rel = 2
				elif score <= score2rel[int(topic_id)][4]:
					rel = 4
				elif score <= score2rel[int(topic_id)][8]:
					rel = 8
				li[4] = str(rel)
				out.write('\t'.join(li))


def clip_rel(args=None):
	# 0:2:4:8:16 = 16:8:4:2:1
	up = [1, 3, 7, 15, 31]
	dw = 31
	relevance = [16, 8, 4, 2, 0]
	topic_num = {}
	with open('/home/trec7/lianxiaoying/trec_eval.9.0/test/bresult.test', 'r', encoding='utf-8') as f:
		for line in f:
			li = line[:].split('\t')
			topic_id = li[0]
			if topic_id in topic_num:
				topic_num[topic_id] += 1
			else:
				topic_num[topic_id] = 1
	with open('/home/trec7/lianxiaoying/trec_eval.9.0/test/bresult.test', 'r', encoding='utf-8') as f:
		with open('/home/trec7/lianxiaoying/trec_eval.9.0/test/bresult.test1', 'w', encoding='utf-8') as out:
			cnt = 0
			now = 0
			topic_id = ''
			for line in f:
				li = line[:].split('\t')
				if li[0] != topic_id:
					now = 0
					cnt = 0
				topic_id = li[0]
				rel = relevance[now]
				tot = topic_num[topic_id] * up[now]/dw
				if cnt < tot:
					cnt += 1
				else:
					now = now + 1
				li[4] = str(rel)
				out.write('\t'.join(li))


if __name__ == "__main__":
	getattr(__import__('Score2Rel'), sys.argv[1])(sys.argv[2:])

