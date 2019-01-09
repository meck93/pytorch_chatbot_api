import torch
import os
import random

from util import *

from models.voc import Voc
from models.decoder import LuongAttnDecoderRNN
from models.encoder import EncoderRNN
from models.greedysearch import GreedySearchDecoder
from models.topksearch import TopKSearchDecoder


class Deploy(object):
    def __init__(self, filename="../models/pretrained_model_checkpoint.tar"):
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
        self.loadFilename = "../models/pretrained_model_checkpoint.tar"

        # Model configuration
        self.attn_model = 'dot'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 128

        # top-k value
        self.k = 6

        # model setup
        self.searcher = None
        self.voc = None
        self.encoder = None
        self.decoder = None
        self.embedding = None

    def evaluate(self, encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
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
        tokens, scores = searcher(input_batch, lengths, max_length)

        decoded_answers = []

        for toks in tokens:
            # indexes -> words
            decoded_words = [voc.index2word[token] for token in toks]
            decoded_answers.append(decoded_words)

        return decoded_answers

    def evaluate_question(self, encoder, decoder, searcher, voc, question):
        try:
            # Normalize sentence
            question = normalize_string(question)

            # Evaluate sentence
            output_words = self.evaluate(
                encoder, decoder, searcher, voc, question)

            # # Format and print response sentence
            # output_words[:] = [x for x in output_words if not (
            #     x == 'EOS' or x == 'PAD')]
            # answer = ' '.join(output_words)
            # print(answer)
            # return answer
            answer = ""

            for output in output_words:
                output[:] = [x for x in output if not (
                    x == 'EOS' or x == 'PAD')]
                answer = ' '.join(output)
                print(answer)

            return answer

        except KeyError:
            print("Error: Encountered unknown word.")
            return "Error: Encountered unknown word."

    def setup_model(self):
        # Load/Assemble voc
        self.voc = Voc('cornell movie-dialogs corpus')

        # If loading on same machine the model was trained on
        checkpoint = torch.load(self.loadFilename, map_location=self.device)

        # retrieve data
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']
        self.voc.__dict__ = checkpoint['voc_dict']

        print('Building encoder and decoder ...')

        # Initialize word embeddings
        self.embedding = torch.nn.Embedding(
            self.voc.num_words, self.hidden_size)
        self.embedding.load_state_dict(embedding_sd)

        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(self.hidden_size, self.embedding,
                                  self.encoder_n_layers, self.dropout)
        self.decoder = LuongAttnDecoderRNN(
            self.attn_model, self.embedding, self.hidden_size, self.voc.num_words, self.decoder_n_layers, self.dropout)
        self.encoder.load_state_dict(encoder_sd)
        self.decoder.load_state_dict(decoder_sd)

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
        if not self.setup:
            self.setup_model()
            self.setup = True

        # log question to be able to see it in heroku logs
        print(question)

        # evaluate answer to question
        return self.evaluate_question(self.encoder, self.decoder, self.searcher, self.voc, question)


if __name__ == "__main__":
    dep = Deploy()
    dep.setup_model()

    while(True):
        # Get input sentence
        input_sentence = input('> ')

        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit':
            break

        dep.evaluate_question(
            dep.encoder, dep.decoder, dep.searcher, dep.voc, input_sentence)
