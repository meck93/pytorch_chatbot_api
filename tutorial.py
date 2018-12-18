from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

from util import *

corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# print_file(os.path.join(corpus, 'movie_lines.txt'))

# create_formatted_file(corpus)

# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = load_prepare_data(corpus, corpus_name, os.path.join(corpus, 'formatted_movie_lines.txt'), save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)
pairs = trim_rare_words(voc, pairs, MIN_COUNT)
