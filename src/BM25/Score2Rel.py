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
	print('bqrel loaded.')
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
		321: { 0: 311.76771888095675, 2: 450.71389147163404, 4: 448.915325586973, 8: 460.5660977328565},
		336: { 0: 185.77456197666376, 2: 255.3335243498566, 4: 301.5495264546049, 8: 401.5495264546049},
		341: { 0: 463.6781750222536, 2: 448.9268186480715, 4: 548.0276714833371, 8: 768.0480871522502},
		347: { 0: 203.7592719057282, 2: 222.45848040993778, 4: 418.3486322237899, 8: 506.6450428078389},
		350: { 0: 371.5872562290564, 2: 578.1833349183328, 4: 678.1833349183328, 8: 778.1833349183328},
		362: { 0: 192.9300263084473, 2: 256.5957498300259, 4: 296.9573773023884, 8: 331.26780512085134},
		363: { 0: 470.3005434239084, 2: 451.6596472945846, 4: 473.53533499566, 8: 714.4877385854606},
		367: { 0: 221.99405017366144, 2: 321.99405017366144, 4: 421.99405017366144, 8: 521.9940501736614},
		375: { 0: 571.1147734337924, 2: 685.5759823150295, 4: 698.793127947651, 8: 927.4727757910161},
		378: { 0: 750.4408299660397, 2: 981.6412575860722, 4: 1102.5148697299878, 8: 1170.8870533251168},
		393: { 0: 618.189996897289, 2: 408.996953115567, 4: 728.8868900485538, 8: 957.838730797241},
		397: { 0: 407.0373516881528, 2: 472.3802165090451, 4: 687.234220592049, 8: 878.5340946076097},
		400: { 0: 1292.8139309887572, 2: 1965.488654334481, 4: 2315.299485567116, 8: 2413.571603232789},
		408: { 0: 240.26431823918168, 2: 313.65396431629785, 4: 384.05146024836193, 8: 358.2338629735448},
		414: { 0: 1513.1103116357522, 2: 1949.2039005220954, 4: 2016.0842438818513, 8: 2255.667010361445},
		422: { 0: 253.71240727755588, 2: 303.9256475470549, 4: 389.75947521455583, 8: 489.75947521455583},
		426: { 0: 276.8518560017762, 2: 292.0832478149309, 4: 327.700566959646, 8: 278.55796432007634},
		427: { 0: 485.0040506733444, 2: 828.9863855097319, 4: 928.9863855097319, 8: 1081.9808433149378},
		433: { 0: 331.20440269226424, 2: 620.8026430627074, 4: 720.8026430627074, 8: 820.8026430627074},
		439: { 0: 252.97403380882974, 2: 268.9630911048314, 4: 333.67495701131315, 8: 317.8464419310266},
		442: { 0: 176.0270506591727, 2: 563.739650370806, 4: 570.4153554263264, 8: 577.6193171597555},
		445: { 0: 226.17532443575797, 2: 249.79160838981784, 4: 277.8030481071621, 8: 518.832468680334},
		626: { 0: 356.68126161994707, 2: 427.3023900116349, 4: 527.3023900116349, 8: 627.3023900116349},
		646: { 0: 292.7993291066441, 2: 384.5107659330562, 4: 373.9312182523376, 8: 396.01406131805123},
		690: { 0: 713.7083450179059, 2: 1073.409845593601, 4: 1087.3233532878464, 8: 1391.0739569531422},
		801: { 0: 407.26824937666925, 2: 639.3622540712823, 4: 685.3033083021915, 8: 785.3033083021915},
		802: { 0: 818.1345786015986, 2: 1007.2202595572472, 4: 1078.1234065184233, 8: 827.0410191879979},
		803: { 0: 439.2149818257386, 2: 445.4062309368231, 4: 602.80038435094, 8: 702.80038435094},
		804: { 0: 288.87900237077076, 2: 447.69598227912945, 4: 603.7692211200714, 8: 703.7692211200714},
		805: { 0: 224.80968687068764, 2: 378.6341139100918, 4: 528.9491907090357, 8: 734.4657796842449},
		806: { 0: 139.05022590450807, 2: 174.5121764960718, 4: 252.8520664838264, 8: 352.85206648382643},
		807: { 0: 740.6123252189027, 2: 841.1534071166578, 4: 941.1534071166578, 8: 1041.153407116658},
		808: { 0: 756.9742865855429, 2: 933.3473007239743, 4: 984.6708143137496, 8: 127.43209445611467},
		809: { 0: 507.1155796895607, 2: 922.6678408178944, 4: 921.2589552350772, 8: 1021.2589552350772},
		810: { 0: 786.021847732717, 2: 830.3286267166004, 4: 923.6537904660054, 8: 1109.749579430107},
		811: { 0: 612.5235224293854, 2: 733.1523265260467, 4: 914.9424287736831, 8: 1299.8411573330227},
		812: { 0: 706.9907190680082, 2: 570.7863095820312, 4: 834.6473625401522, 8: 714.8946916980901},
		813: { 0: 363.55585533853724, 2: 421.73757574599983, 4: 500.24403977822055, 8: 769.4043899610459},
		814: { 0: 518.9944708756199, 2: 539.2426313438034, 4: 713.9494653917512, 8: 718.7342310390826},
		815: { 0: 915.6197444099608, 2: 1407.1587046628356, 4: 1368.3598284357422, 8: 1468.3598284357422},
		816: { 0: 323.2664707782508, 2: 370.30016950769, 4: 472.39506627729844, 8: 448.9474826521988},
		817: { 0: 167.3337057005214, 2: 187.06178935335956, 4: 287.0617893533596, 8: 387.0617893533596},
		818: { 0: 370.7156664444699, 2: 509.41554689020177, 4: 825.0119812696181, 8: 717.8496825529646},
		819: { 0: 287.0871160169111, 2: 424.55717011133856, 4: 389.8708951137161, 8: 334.11130458782776},
		820: { 0: 562.1829489533839, 2: 549.3521746027054, 4: 441.00985098780933, 8: 541.0098509878094},
		821: { 0: 294.8255641097239, 2: 320.6358634348871, 4: 318.9763237148494, 8: 266.0441514692607},
		822: { 0: 474.8862263915545, 2: 577.1705214347863, 4: 632.7562830954967, 8: 694.7782584732664},
		823: { 0: 419.3091314089788, 2: 500.9264731908, 4: 438.3370836069338, 8: 380.6680903946895},
		824: { 0: 705.8861439583579, 2: 946.6849240155043, 4: 992.0181240498138, 8: 1092.0181240498137},
		825: { 0: 4271.1953630012795, 2: 5540.266247042031, 4: 9021.750428540248, 8: 10792.205398404578}
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

