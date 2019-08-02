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
		321: { 0: 87.45677016118978, 2: 536.7791586396904, 4: 652.1196605800965, 8: 802.693059319594},
		336: { 0: 20.244167222475248, 2: 353.4854510237587, 4: 395.1991304231443, 8: 495.1991304231443},
		341: { 0: 1.3646867612558944, 2: 1201.6345654130039, 4: 1340.4638803678272, 8: 916.7873774200604},
		347: { 0: 9.41559028123158, 2: 359.53946107203586, 4: 701.8769949881695, 8: 516.7889664916239},
		350: { 0: 57.38336581504829, 2: 724.1786652228778, 4: 824.1786652228778, 8: 924.1786652228778},
		362: { 0: 41.98556571550598, 2: 331.8252585982037, 4: 877.4069573435568, 8: 450.18240976024424},
		363: { 0: 0.9747762580399246, 2: 544.3275409653529, 4: 762.4065705903139, 8: 1041.5284385053944},
		367: { 0: 26.982989673799555, 2: 126.98298967379955, 4: 226.98298967379955, 8: 326.9829896737996},
		375: { 0: 126.83519307797391, 2: 1091.0524398251196, 4: 1113.380704492819, 8: 2169.527888812617},
		378: { 0: 62.90153708926512, 2: 1363.488778436188, 4: 1416.451755298497, 8: 1325.6472100014282},
		393: { 0: 104.47316682303686, 2: 1367.4194751373695, 4: 1760.50501271286, 8: 1346.565459524873},
		397: { 0: 120.32547634152543, 2: 1215.7693587332533, 4: 1043.95224658658, 8: 1045.3753232044048},
		400: { 0: 1.7545972644718641, 2: 2889.4892256330954, 4: 2315.299485567116, 8: 2540.3323660568217},
		408: { 0: 23.675685204673684, 2: 755.7336549137298, 4: 594.1897029559602, 8: 882.0919875467375},
		414: { 0: 231.67033617909877, 2: 2269.9807285936513, 4: 2233.1345920012777, 8: 2590.8388548765884},
		422: { 0: 13.161929554677311, 2: 480.1916044388583, 4: 466.4750041709075, 8: 566.4750041709075},
		426: { 0: 0.5848657548239549, 2: 443.4277823261102, 4: 412.51932127579704, 8: 474.8293403278333},
		427: { 0: 36.78099973271944, 2: 1005.1741641003775, 4: 1105.1741641003775, 8: 1081.9808433149378},
		433: { 0: 3.3142392773357434, 2: 631.6421012085392, 4: 731.6421012085392, 8: 831.6421012085392},
		439: { 0: 71.26558977664953, 2: 442.65980058640616, 4: 404.35784480955977, 8: 317.8464419310266},
		442: { 0: 8.74764125492447, 2: 662.0329642974698, 4: 575.7079690845108, 8: 577.6193171597555},
		445: { 0: 18.092734997673528, 2: 585.2828958824136, 4: 277.8030481071621, 8: 518.832468680334},
		626: { 0: 71.28726826947003, 2: 427.3023900116349, 4: 527.3023900116349, 8: 627.3023900116349},
		646: { 0: 0.19495525160798494, 2: 538.9759501807354, 4: 418.8977196813316, 8: 539.0124134935977},
		690: { 0: 65.06396417669137, 2: 1464.491236769901, 4: 1325.8087615377485, 8: 1391.0739569531422},
		801: { 0: 42.42082102698362, 2: 851.7483752797649, 4: 737.5076895507502, 8: 837.5076895507502},
		802: { 0: 71.61443625406967, 2: 1250.0070735639274, 4: 1360.6288953463595, 8: 827.0410191879979},
		803: { 0: 38.86470548352989, 2: 786.2266040455897, 4: 950.2011662016864, 8: 1050.2011662016864},
		804: { 0: 35.99871190858497, 2: 867.6296801965404, 4: 802.9912217602928, 8: 902.9912217602928},
		805: { 0: 42.261322503509774, 2: 705.2691716355839, 4: 815.4532582064867, 8: 734.4657796842449},
		806: { 0: 0.9747762580399246, 2: 185.30189361870177, 4: 252.8520664838264, 8: 352.85206648382643},
		807: { 0: 37.996048191623906, 2: 952.3511452522451, 4: 1052.3511452522453, 8: 1152.3511452522453},
		808: { 0: 38.41814594257344, 2: 1251.1352658881015, 4: 1389.2696443959055, 8: 127.43209445611467},
		809: { 0: 149.0827651040307, 2: 1074.5276525924185, 4: 1242.8286829130677, 8: 1342.8286829130677},
		810: { 0: 247.8904404561592, 2: 1006.5246785191787, 4: 1215.3684811927214, 8: 1177.631752894027},
		811: { 0: 22.841959968738315, 2: 1081.6830738934586, 4: 1225.2157202545302, 8: 1451.3021672714044},
		812: { 0: 27.532184277699887, 2: 1109.3485907186214, 4: 1070.831166996853, 8: 772.0588973594386},
		813: { 0: 11.071970117013862, 2: 613.912705331351, 4: 500.24403977822055, 8: 769.4043899610459},
		814: { 0: 160.28024033867675, 2: 957.9821365853567, 4: 1097.1636591502352, 8: 923.0468827128185},
		815: { 0: 22.511598310293763, 2: 2042.0382640028718, 4: 1838.88568544611, 8: 1938.88568544611},
		816: { 0: 1.2679571452660459, 2: 607.0312066283087, 4: 669.0188718313286, 8: 620.5245462989097},
		817: { 0: 0.7798210064319397, 2: 187.06178935335956, 4: 287.0617893533596, 8: 387.0617893533596},
		818: { 0: 75.80823225060573, 2: 804.6183307345211, 4: 825.0119812696181, 8: 825.0119812696181},
		819: { 0: 14.014104544190713, 2: 551.6081335812588, 4: 448.73537792464805, 8: 334.11130458782776},
		820: { 0: 46.824719077497036, 2: 1006.8370132668201, 4: 662.2938401251046, 8: 762.2938401251046},
		821: { 0: 85.15636672954388, 2: 564.415382978843, 4: 572.9986050229693, 8: 327.7753040643764},
		822: { 0: 114.86170342498886, 2: 801.4516719813921, 4: 1243.5730502997346, 8: 912.2977806020073},
		823: { 0: 101.41468719775297, 2: 749.3220624811357, 4: 537.2304291705872, 8: 423.8753517787723},
		824: { 0: 1.7545972644718641, 2: 1375.4525308689128, 4: 1022.8322985457771, 8: 1122.8322985457771},
		825: { 0: 434.5945051559846, 2: 9283.437523746126, 4: 11282.455369817328, 8: 10792.205398404578}
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

