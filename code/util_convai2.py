import codecs
import csv
import re
from pprint import pprint

import torch

from models.voc import Voc


def create_formatted_file(filename, fields=['question', 'answer']):
    lines = {}
    conv_nr = 0

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            sep = int(line[:1])
            line = line[2:]
            line = re.sub(r"\n", "", line)
            values = line.split('\t')

            if sep == 1:
                conv_nr += 1
                line_obj = {'convs': [(values[0], values[1])]}
                lines[conv_nr] = line_obj

            else:
                lines[conv_nr]['convs'].append((values[0], values[1]))

    return lines


def create_sentence_pairs(lines):
    qa_pairs = []
    next_q = None
    q = None
    a = None

    for i, values in lines.items():
        for index in range(len(values['convs']) - 1):
            q = values['convs'][index][0]
            a = values['convs'][index][1]
            next_q = values['convs'][index+1][0]
            qa_pairs.append([q, a])
            qa_pairs.append([a, next_q])
        qa_pairs.append([next_q, values['convs'][-1][1]])

    return qa_pairs


in_file = "../../data/convai2/train_none_original_no_cands.txt"
lines = create_formatted_file(in_file)
qa_pairs = create_sentence_pairs(lines)
pprint(qa_pairs[:14])

delimiter = '\t'

# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

with open('formatted_convai2_lines.txt', 'w', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter=delimiter, lineterminator='\n')
    for pair in qa_pairs:
        writer.writerow(pair)
