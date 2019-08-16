import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import jieba
import re
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
import fasttext


# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')


