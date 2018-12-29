import unicodedata
import re
from model import Voc
import torch
import itertools
import random
import os
import codecs
import csv

'''
Utility functions for corpus preprocessing

'''

MAX_LENGTH = 5  # Maximum sentence length to consider
MIN_COUNT = 3   # Minimum word count threshold for trimming
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


def print_file(file_name, num_lines=10):
    with open(file_name, 'rb') as file:
        lines = file.readlines()
    for line in lines[:num_lines]:
        print(line)


def load_lines(file_name, fields):
    '''
    Splits each line of the file into a dictionary of fields
    fields are lineId, characterID, movieID, character, text
    '''
    lines = {}
    with open(file_name, 'r', encoding='iso-8859-1') as file:
        for line in file:
            values = line.split(' +++$+++ ')
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]
            lines[line_obj['lineID']] = line_obj
    print(lines)
    return lines


def load_conversations(file_name, lines, fields):
    '''
    Groups fields of lines from 'load_lines' into conversations based on movie_conversations.txt
    '''
    conversations = []
    with open(file_name, 'r', encoding='iso-8859-1') as file:
        for line in file:
            values = line.split(' +++$+++ ')
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            line_ids = eval(conv_obj["utteranceIDs"])
            # Reassemble lines
            conv_obj["lines"] = []
            for id in line_ids:
                conv_obj["lines"].append(lines[id])
            conversations.append(conv_obj)
    return conversations


def extract_sentence_pairs(conversations):
    """ 
    Extracts pairs of sentences from conversations
    """
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        # We ignore the last line (no answer for it)
        for i in range(len(conversation["lines"]) - 1):
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs


def create_formatted_file(corpus, file_name='formatted_movie_lines.txt'):
    file = os.path.join(corpus, file_name)

    delimiter = '\t'

    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID",
                          "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID",
                                  "character2ID", "movieID", "utteranceIDs"]

    print("\nProcessing corpus...")
    lines = load_lines(os.path.join(
        corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = load_conversations(os.path.join(corpus, "movie_conversations.txt"),
                                       lines, MOVIE_CONVERSATIONS_FIELDS)

    print("\nWriting newly formatted file...")
    with open(file, 'w', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter, lineterminator='\n')
        for pair in extract_sentence_pairs(conversations):
            writer.writerow(pair)

    # Print a sample of the new file
    print("\nSample lines from file:")
    print_file(file)


def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to
    http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    ''' 
    Lowercase, trim, and remove non-letter characters
    '''
    # lowercase the string
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)

    # additional convertions
    s = re.sub(r"i'm", "i am", s)
    s = re.sub(r"he's", "he is", s)
    s = re.sub(r"she's", "she is", s)
    s = re.sub(r"it's", "it is", s)
    s = re.sub(r"that's", "that is", s)
    s = re.sub(r"what's", "that is", s)
    s = re.sub(r"where's", "where is", s)
    s = re.sub(r"how's", "how is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"won't", "will not", s)
    s = re.sub(r"can't", "cannot", s)
    s = re.sub(r"n't", " not", s)
    s = re.sub(r"n'", "ng", s)
    s = re.sub(r"'bout", "about", s)
    s = re.sub(r"'til", "until", s)

    # replace any non-letter character expect for basic punctuation with a whitespace
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    # replace multiple whitespace with a single whitespace
    s = re.sub(r"\s+", r" ", s).strip()

    return s


def read_vocs(datafile, corpus_name):
    """
    Read query/response pairs and return a voc object
    """
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filter_pair(p):
    """
    Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    """
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    """
    Filter pairs using filterPair condition
    """
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus, corpus_name, datafile, save_dir):
    """
    Using the functions defined above, return a populated voc object and pairs list
    """
    print("Start preparing training data ...")

    voc, pairs = read_vocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))

    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")

    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])

    print("Counted words:", voc.num_words)

    return voc, pairs


def trim_rare_words(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)

    # Filter out pairs with trimmed words
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break

        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))

    return keep_pairs


def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zero_padding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def input_var(l, voc):
    """
    Returns padded input sequence tensor and lengths
    """
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zero_padding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def output_var(l, voc):
    """
    Returns padded target sequence tensor, padding mask, and max target length
    """
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zero_padding(indexes_batch)
    mask = binary_matrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2train_data(voc, pair_batch):
    """
    Returns all items for a given batch of pairs
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []

    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    inp, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)

    return inp, lengths, output, mask, max_target_len


def sample_batches(voc, pairs, n_iteration, batch_size):
    training_batches = []
    curr_pairs = []

    for _ in range(n_iteration):
        batches = []

        for _ in range(batch_size):
            # check if all pairs have been used
            if len(curr_pairs) == 0:
                # copy all pairs and shuffle
                curr_pairs = pairs.copy()
                random.shuffle(curr_pairs)

            # pop first pair from the list and add it to current batch
            batches.append(curr_pairs.pop(0))

        # append batch for current iteration
        training_batches.append(batch2train_data(voc, batches))

    return training_batches
