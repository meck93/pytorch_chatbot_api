import torch
import os

from util import *
from model import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, Voc

# set the random seed
SEED = 15

# setup
setup = False

# set the device to cpu
device = None

# Set checkpoint to load from
loadFilename = "./model/pretrained_model_checkpoint.tar"

# Model configuration
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 128

# model setup
searcher = None
voc = None
encoder = None
decoder = None


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]

    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)

    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)

    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoded_words


def evaluate_question(encoder, decoder, searcher, voc, question):
    try:
        # Normalize sentence
        question = normalize_string(question)

        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, question)

        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (
            x == 'EOS' or x == 'PAD')]
        return(' '.join(output_words))

    except KeyError:
        return "Error: Encountered unknown word."


def setup_model():
    # set the random seed
    random.seed(SEED)

    # set the device to cpu
    device = torch.device('cpu')

    # Load/Assemble voc
    voc = Voc('cornell movie-dialogs corpus')

    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename, map_location=device)

    # retrieve data
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')

    # Initialize word embeddings
    embedding = torch.nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    print('Models built and ready to go!')

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)


def reply(question):
    print(question)
    if not setup:
        setup_model()
        setup = True
    return evaluate_question(encoder, decoder, searcher, voc, question)
