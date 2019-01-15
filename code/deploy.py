import torch
import os
import random

from .util import *

from .models.voc import Voc
from .models.decoder import LuongAttnDecoderRNN
from .models.encoder import EncoderRNN
# from .models.greedysearch import GreedySearchDecoder
from .models.topksearch import TopKSearchDecoder


class Deploy(object):
    """
    Class for running a pre-trained pytorch seq2seq model. 
    Takes care of running the server and the prediction endpoint. 
    Args:
       filepath (string): filepath of the folder containing the pre-trained seq2seq model
       k (int): value for the top-k searcher
    """

    def __init__(self, filepath="./pre_trained_models/max_len_12_6000_cp/", k=5):
        # set the random seed
        self.SEED = 15
        random.seed(self.SEED)

        # setup
        self.setup = False

        # setting device on GPU if available, else CPU
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        # Set checkpoint to load from
        self.filepath = filepath

        # Model configuration
        self.attn_model = 'dot'
        self.hidden_size = 512
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.2
        self.batch_size = 256
        self.max_length = 12

        # top-k value
        self.k = k

        # model setup
        self.searcher = None
        self.voc = None
        self.encoder = None
        self.decoder = None
        self.embedding = None

    def evaluate(self, encoder, decoder, searcher, voc, sentence):
        # Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexes_from_sentence(voc, sentence)]

        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to(self.device)

        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, self.max_length)

        # only transform the top token sequence
        tokens = tokens[0]

        # indexes -> words
        decoded_words = [voc.index2word[token] for token in tokens]

        return decoded_words

    def evaluate_question(self, encoder, decoder, searcher, voc, question):
        try:
            # normalize sentence
            question = normalize_string(question)

            # evaluate input sentence i.e. produce a response sentence
            output_words = self.evaluate(
                encoder, decoder, searcher, voc, question)

            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (
                x == 'EOS' or x == 'PAD')]
            answer = ' '.join(output_words)

            return answer

        except KeyError:
            print("Error: Encountered unknown word.")
            return "Error: Encountered unknown word."

    def setup_model(self):
        # Load/Assemble voc
        self.voc = Voc('deployed ;-)')

        # load the pre-trained componenents saved as tar file
        en_cp = torch.load("{}/encoder_cp.tar".format(self.filepath),
                           map_location=self.device)
        de_cp = torch.load("{}/decoder_cp.tar".format(self.filepath),
                           map_location=self.device)
        cp = torch.load("{}/voc_embedding_cp.tar".format(self.filepath),
                        map_location=self.device)

        # retrieve the stored vocabulary
        self.voc.__dict__ = cp['voc_dict']

        print('Building encoder and decoder ...')

        # Initialize word embeddings
        self.embedding = torch.nn.Embedding(
            self.voc.num_words, self.hidden_size)
        self.embedding.load_state_dict(cp['embedding'])

        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(self.hidden_size, self.embedding,
                                  self.encoder_n_layers, self.dropout)
        self.decoder = LuongAttnDecoderRNN(
            self.attn_model, self.embedding, self.hidden_size, self.voc.num_words, self.decoder_n_layers, self.dropout)
        self.encoder.load_state_dict(en_cp['en'])
        self.decoder.load_state_dict(de_cp['de'])

        # Use appropriate device
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # Set dropout layers to eval mode
        self.encoder.eval()
        self.decoder.eval()

        print('Models built and ready to go!')

        # Initialize search module
        # self.searcher = GreedySearchDecoder(self.encoder, self.decoder)
        self.searcher = TopKSearchDecoder(self.encoder, self.decoder, self.k)

    def reply(self, question):
        # setup the model once
        if not self.setup:
            self.setup_model()
            self.setup = True

        # evaluate answer to question
        answer = self.evaluate_question(
            self.encoder, self.decoder, self.searcher, self.voc, question)

        # log question and answer to heroku console
        print("Q:", question, "\nA:", answer)

        return answer


if __name__ == "__main__":
    dep = Deploy()
    dep.setup_model()

    while(True):
        # Get input sentence
        input_sentence = input('> ')

        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit':
            break

        answer = dep.evaluate_question(
            dep.encoder, dep.decoder, dep.searcher, dep.voc, input_sentence)

        print("Bot:", answer)
