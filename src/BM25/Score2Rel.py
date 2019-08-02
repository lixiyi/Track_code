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


path_mp = cfg.get_path_conf('../path.cfg')


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
		query = cfg.word_cut(obj['title'] + ' ' + body)
		rel_mp[topic_id] = []
		for doc_id, rel in ans_mp[topic_id]:
			score = bm25.calc_score(WashingtonPost[doc_id], words_df, query, avgdl, True)
			rel_mp[topic_id].append([score, rel])
	with open(cfg.OUTPUT + 'rel_mp.txt', 'w', encoding='utf-8') as f:
		f.write(json.dumps(rel_mp))


# modify bm25 score in col4 to rel
def transform(args=None):
	score2rel = {
		321: { 0: 431.17027553303797, 2: 482.04731343848596, 4: 501.9180612551568, 8: 563.7278068061114},
		336: { 0: 229.35770276685636, 2: 269.08420121080513, 4: 328.54492541464873, 8: 428.54492541464873},
		341: { 0: 615.9301470729367, 2: 555.3749340143655, 4: 739.9258808348904, 8: 826.3505692055076},
		347: { 0: 241.97576813035462, 2: 315.88928308724655, 4: 462.87724572339755, 8: 511.7170046497314},
		350: { 0: 471.97053078277105, 2: 651.1810000706053, 4: 751.1810000706053, 8: 851.1810000706053},
		362: { 0: 228.56989584995364, 2: 302.51505001259284, 4: 458.9227886307135, 8: 390.72510744054773},
		363: { 0: 565.012513033701, 2: 493.3886504039046, 4: 563.7478724719075, 8: 841.3558677480598},
		367: { 0: 291.726196406642, 2: 391.726196406642, 4: 491.726196406642, 8: 591.726196406642},
		375: { 0: 708.2435691688022, 2: 831.6225741533395, 4: 967.780929502237, 8: 1222.5381235546993},
		378: { 0: 915.9887579478358, 2: 1094.9044369714927, 4: 1219.38058816488, 8: 1236.7200650113264},
		393: { 0: 879.3127598432403, 2: 621.4487046516128, 4: 1210.443504782425, 8: 1210.5304516620786},
		397: { 0: 494.7851699573025, 2: 667.1152618110596, 4: 750.2759628200207, 8: 978.2505212984775},
		400: { 0: 1823.7095221959646, 2: 2242.1604692562796, 4: 2315.299485567116, 8: 2459.619674473394},
		408: { 0: 357.39870177795245, 2: 490.2651148252414, 4: 467.23481877794046, 8: 507.4030081042806},
		414: { 0: 1820.6385095656224, 2: 2083.759334924558, 4: 2124.6094179415645, 8: 2423.252932619017},
		422: { 0: 333.7590760642015, 2: 384.05298121848983, 4: 428.1172396927317, 8: 528.1172396927317},
		426: { 0: 360.227678255139, 2: 365.02050025110344, 4: 370.8050554425795, 8: 342.52348815052085},
		427: { 0: 630.0596054330709, 2: 934.8844862054681, 4: 1034.884486205468, 8: 1081.9808433149378},
		433: { 0: 339.9487503243228, 2: 626.2223721356232, 4: 726.2223721356232, 8: 826.2223721356232},
		439: { 0: 302.79084272437007, 2: 344.5730960336855, 4: 361.62094467307367, 8: 317.8464419310266},
		442: { 0: 237.29870314622795, 2: 583.8712974888942, 4: 573.0616622554186, 8: 577.6193171597555},
		445: { 0: 293.09970320919444, 2: 288.03498084509647, 4: 277.8030481071621, 8: 518.832468680334},
		626: { 0: 426.2700224722166, 2: 427.3023900116349, 4: 527.3023900116349, 8: 627.3023900116349},
		646: { 0: 369.08796234035896, 2: 443.935123934986, 4: 385.24469081440157, 8: 497.32309993925014},
		690: { 0: 950.3881613983126, 2: 1234.0919381939325, 4: 1239.7301282399703, 8: 1391.0739569531422},
		801: { 0: 561.3634776197728, 2: 708.83816273085, 4: 711.405498926471, 8: 811.405498926471},
		802: { 0: 1022.6087769473519, 2: 1093.3996663926234, 4: 1154.0407340463225, 8: 827.0410191879979},
		803: { 0: 552.7397722106546, 2: 547.0006510423678, 4: 754.9454616320847, 8: 854.9454616320847},
		804: { 0: 400.3317229787958, 2: 489.75976746715605, 4: 693.7900934509302, 8: 793.7900934509302},
		805: { 0: 331.51056797332535, 2: 428.4882561259944, 4: 719.3126551200353, 8: 734.4657796842449},
		806: { 0: 171.08641818857737, 2: 176.2249323119674, 4: 252.8520664838264, 8: 352.85206648382643},
		807: { 0: 835.6789379302138, 2: 896.7522761844515, 4: 996.7522761844515, 8: 1096.7522761844516},
		808: { 0: 952.3986515279939, 2: 1017.5432682497315, 4: 1339.722725229124, 8: 127.43209445611467},
		809: { 0: 562.7419401773576, 2: 1074.5276525924185, 4: 921.2589552350772, 8: 1021.2589552350772},
		810: { 0: 880.1341598296499, 2: 931.5146287131374, 4: 1069.5111358293634, 8: 1172.4589276569263},
		811: { 0: 829.0233084033657, 2: 868.6629072036142, 4: 997.4008286970932, 8: 1375.5716623022136},
		812: { 0: 935.0809497404271, 2: 795.3622434111317, 4: 899.5731905430151, 8: 743.4767945287643},
		813: { 0: 436.0300101883081, 2: 569.4991414513762, 4: 500.24403977822055, 8: 769.4043899610459},
		814: { 0: 643.8398627984812, 2: 715.9724163198431, 4: 841.6587332641343, 8: 820.8905568759506},
		815: { 0: 1372.699178259034, 2: 1666.8129012933152, 4: 1499.5521407453991, 8: 1599.5521407453991},
		816: { 0: 431.63214622422515, 2: 420.75136886803716, 4: 525.4724267991128, 8: 535.5927525685353},
		817: { 0: 195.39306634250528, 2: 187.06178935335956, 4: 287.0617893533596, 8: 387.0617893533596},
		818: { 0: 501.7315896940711, 2: 623.9460846219349, 4: 825.0119812696181, 8: 720.7521080918244},
		819: { 0: 368.1025700497329, 2: 448.76140232024, 4: 419.3031365191821, 8: 334.11130458782776},
		820: { 0: 587.9919356850046, 2: 710.2171771796302, 4: 591.6664105465209, 8: 691.6664105465209},
		821: { 0: 327.02561541991764, 2: 360.6495302990759, 4: 348.5415286742297, 8: 273.6675809236879},
		822: { 0: 546.8412410542687, 2: 683.772559611358, 4: 715.02718935805, 8: 803.5380195376368},
		823: { 0: 490.5756565320326, 2: 580.3453202786852, 4: 507.1761680822731, 8: 402.2717210867309},
		824: { 0: 867.5063512475199, 2: 1128.1648511370583, 4: 1007.4252112977954, 8: 1107.4252112977954},
		825: { 0: 5558.334035990169, 2: 7277.426076840034, 4: 9668.904550964597, 8: 10792.205398404578}
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

