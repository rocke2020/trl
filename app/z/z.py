import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from loguru import logger
from datasets import load_dataset

SEED = 0
random.seed(SEED)


dataset = load_dataset(path='trl-lib/ultrafeedback_binarized', name=None)
print(dataset)
out_dir = '/data/dong-qichang/corpus/trl-lib/ultrafeedback_binarized'
Path(out_dir).mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(out_dir)