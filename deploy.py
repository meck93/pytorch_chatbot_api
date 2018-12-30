import torch
import os

from util import *
from model import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, Voc


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


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(True):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break

            # Normalize sentence
            input_sentence = normalize_string(input_sentence)

            # Evaluate sentence
            output_words = evaluate(
                encoder, decoder, searcher, voc, input_sentence)

            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (
                x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# set the random seed
SEED = 15
random.seed(SEED)

device = torch.device('cpu')

# Model configuration
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 512
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 128

corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = "/data/save/cb_model/cornell movie-dialogs corpus/max_len_12_best/4000_checkpoint.tar"

# Load/Assemble voc
voc = Voc('trained')

# Load model if a loadFilename is provided
if loadFilename:
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

if loadFilename:
    embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

print('Models built and ready to go!')

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)
