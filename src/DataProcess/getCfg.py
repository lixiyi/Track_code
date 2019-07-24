DOCUMENT_COUNT = 595037 - 23074
OUTPUT = '/home/trec7/lianxiaoying/Track_code/src/outputs/'

# get file path conf
def get_path_conf(filename):
	path_mp = {}
	with open(filename, 'r', encoding='utf-8') as f:
		for line in f:
			li = line[:-1].split('=')
			path_mp[li[0]] = li[1]
	return path_mp

