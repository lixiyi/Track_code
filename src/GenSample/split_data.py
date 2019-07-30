import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import random
import numpy as np


path_mp = cfg.get_path_conf('../path.cfg')

with open(cfg.OUTPUT + 'train.txt', 'w', encoding='utf-8') as out1:
	with open(cfg.OUTPUT + 'dev.txt', 'w', encoding='utf-8') as out2:
		with open(cfg.OUTPUT + 'test.txt', 'w', encoding='utf-8') as out3:
			with open(cfg.OUTPUT + 'tmp.txt', 'r', encoding='utf-8') as f:
				cnt = 0
				for line in tqdm(f):
					if cnt < 100:
						out3.write(line)
					elif cnt < 200:
						out2.write(line)
					else:
						out1.write(line)
					cnt += 1
