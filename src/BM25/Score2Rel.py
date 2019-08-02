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
		321: { 0: 522.2084158574443, 2: 525.3955835858974, 4: 600.477865734854, 8: 660.774943051653},
		336: { 0: 301.0760805229698, 2: 299.5244869010015, 4: 367.96392872932614, 8: 467.96392872932614},
		341: { 0: 824.4186491810123, 2: 1053.2142678726395, 4: 1092.4156555896438, 8: 897.8093129914038},
		347: { 0: 329.2566913010922, 2: 346.90977976574584, 4: 642.3329331623454, 8: 515.7745741232454},
		350: { 0: 724.3985359421357, 2: 709.5791321924233, 4: 809.5791321924233, 8: 909.5791321924233},
		362: { 0: 293.5816652911123, 2: 321.55945855202, 4: 660.8655002289149, 8: 438.2909492963049},
		363: { 0: 686.7121483205217, 2: 528.4500451578325, 4: 711.1308481250661, 8: 1001.4939243539275},
		367: { 0: 523.0487431177764, 2: 623.0487431177764, 4: 723.0487431177764, 8: 823.0487431177764},
		375: { 0: 1027.2667214909038, 2: 1033.1690635171703, 4: 1062.9970240551684, 8: 2110.533453065548},
		378: { 0: 1123.170607009561, 2: 1244.801301650107, 4: 1393.3865946280484, 8: 1305.3327289623742},
		393: { 0: 1064.1321578594186, 2: 992.0812801511436, 4: 1740.7795887627685, 8: 1327.158616638468},
		397: { 0: 832.0971761806521, 2: 886.932465666142, 4: 971.857399193769, 8: 1017.4811452437208},
		400: { 0: 2494.8477528321073, 2: 2630.65317698999, 4: 2315.299485567116, 8: 2524.189827740136},
		408: { 0: 652.1710020909566, 2: 689.9067954552711, 4: 589.8288070807791, 8: 740.675289798764},
		414: { 0: 2110.5275457158873, 2: 2189.559378852841, 4: 2211.429557189335, 8: 2557.321670425074},
		422: { 0: 426.9335025574536, 2: 461.85326768537783, 4: 458.8034512752723, 8: 558.8034512752723},
		426: { 0: 445.16475241223696, 2: 415.9274429942142, 4: 410.91959600482403, 8: 442.1784293384159},
		427: { 0: 862.6730761400387, 2: 997.2960750707691, 4: 1097.2960750707691, 8: 1081.9808433149378},
		433: { 0: 481.02921318345386, 2: 630.558155393956, 4: 730.558155393956, 8: 830.558155393956},
		439: { 0: 412.8219630367926, 2: 411.1267002543972, 4: 396.43462817979935, 8: 317.8464419310266},
		442: { 0: 598.2251771986402, 2: 648.5844406415474, 4: 575.1787077186924, 8: 577.6193171597555},
		445: { 0: 372.56881480081825, 2: 438.4499395963671, 4: 277.8030481071621, 8: 518.832468680334},
		626: { 0: 540.1662095223423, 2: 427.3023900116349, 4: 527.3023900116349, 8: 627.3023900116349},
		646: { 0: 443.4135079448109, 2: 503.28133975722164, 4: 410.62908118877345, 8: 530.6745507827281},
		690: { 0: 1248.360015476939, 2: 1397.9992431221574, 4: 1310.188776998764, 8: 1391.0739569531422},
		801: { 0: 736.1531817586443, 2: 832.558853101365, 4: 732.2872514258944, 8: 832.2872514258944},
		802: { 0: 1237.5337769177042, 2: 1196.7804970240156, 4: 1305.2532102736668, 8: 827.0410191879979},
		803: { 0: 777.6021503888037, 2: 688.1219567118421, 4: 820.6596812009559, 8: 920.6596812009559},
		804: { 0: 490.43548939062924, 2: 607.673713088819, 4: 781.1509960984203, 8: 881.1509960984203},
		805: { 0: 556.8995488008843, 2: 609.6041674895827, 4: 807.3986917784673, 8: 734.4657796842449},
		806: { 0: 248.41125408194966, 2: 183.48650135735488, 4: 252.8520664838264, 8: 352.85206648382643},
		807: { 0: 970.7844999322197, 2: 941.2313714386864, 4: 1041.2313714386864, 8: 1141.2313714386864},
		808: { 0: 1247.3117178457364, 2: 1140.4108687338264, 4: 1379.3602605625492, 8: 127.43209445611467},
		809: { 0: 674.0447425804645, 2: 1074.5276525924185, 4: 1242.8286829130677, 8: 1342.8286829130677},
		810: { 0: 1007.7932465614933, 2: 991.5226685579704, 4: 1186.1970121200497, 8: 1176.5971878466069},
		811: { 0: 1464.998478233044, 2: 1034.175209780681, 4: 1179.6527419430429, 8: 1436.1560662775662},
		812: { 0: 1582.5809472958326, 2: 1002.9488611896229, 4: 1012.2521559199961, 8: 766.3424767933037},
		813: { 0: 600.560125202849, 2: 606.3883206985383, 4: 500.24403977822055, 8: 769.4043899610459},
		814: { 0: 840.0698137208745, 2: 944.4581768281806, 4: 1031.0462071529205, 8: 902.6156175454448},
		815: { 0: 1674.5894424914259, 2: 1856.5265800583202, 4: 1783.861278488308, 8: 1883.861278488308},
		816: { 0: 562.1538656200238, 2: 512.6985265195775, 4: 571.1407091710897, 8: 600.7242833372557},
		817: { 0: 221.00890284357098, 2: 187.06178935335956, 4: 287.0617893533596, 8: 387.0617893533596},
		818: { 0: 989.9229694457582, 2: 761.5337554149359, 4: 825.0119812696181, 8: 804.1600066340594},
		819: { 0: 496.87346905980723, 2: 515.3467325807405, 4: 442.84892964355487, 8: 334.11130458782776},
		820: { 0: 715.6443232851384, 2: 873.4366934888747, 4: 645.3559443176815, 8: 745.3559443176815},
		821: { 0: 423.3844169130209, 2: 491.4162247151553, 4: 496.979400461135, 8: 316.9537594362387},
		822: { 0: 700.4624305333988, 2: 748.1132567348266, 4: 989.7899653874211, 8: 890.5458283891331},
		823: { 0: 658.2503772438075, 2: 679.0429489297251, 4: 527.8928971421544, 8: 419.554625640364},
		824: { 0: 1062.5334476788278, 2: 1224.0222447781066, 4: 1019.7508810961808, 8: 1119.7508810961808},
		825: { 0: 8490.43779058565, 2: 8652.140383722792, 4: 11223.553089968884, 8: 10792.205398404578}
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

