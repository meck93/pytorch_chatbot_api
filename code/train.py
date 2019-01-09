from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import optim

from .models.decoder import LuongAttnDecoderRNN
from .models.encoder import EncoderRNN
from .models.greedysearch import GreedySearchDecoder
from .models.topksearch import TopKSearchDecoder
from .models.voc import Voc
from .util import *


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    """
    Compute the training step for the current batch.
    Update the model weights.
    """

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # clip gradients: modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def validate(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, batch_size):
    """
    Evaluate performance on validation set
    """
    # switch model to evaluation mode
    encoder.eval()
    decoder.eval()

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    print_losses = []
    n_totals = 0

    with torch.no_grad():
        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor(
            [[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # switch model to training mode
    encoder.train()
    decoder.train()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # shuffle dataset
    random.shuffle(pairs)

    # split the pairs into shuffled training and validation splits
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=0.15, random_state=SEED)

    n_train_pairs = len(train_pairs)
    n_val_pairs = len(val_pairs)
    print("Dataset: # of samples: {}, Train: {}, Val: {}, {} Iters / Epoch, Batch Size: {}".format(
        len(pairs), n_train_pairs, n_val_pairs, n_train_pairs // batch_size, batch_size))

    print("Creating training and validation batches ...")

    # create a training batch for each iteration
    training_batches = sample_batches(
        voc, train_pairs, n_iteration, batch_size)

    # create a validation batch for each %print_every% training batches
    validation_batches = sample_batches(
        voc, val_pairs, (n_iteration // print_every) + 1, batch_size)
    val_count = 0

    # Initializations
    curr_epoch = 0
    start_iteration = 1
    print_loss = 0

    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Let's Train ...")
    for iteration in range(start_iteration, n_iteration + 1):
        # Current training batch
        training_batch = training_batches[iteration - 1]

        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Check when a full epoch has been completed
        if iteration % (n_train_pairs // batch_size) == 0:
            curr_epoch += 1
            # TODO: Figure out if learning rate scheduling is really necessary with Adam optimizer
            # encoder_scheduler.step()
            # decoder_scheduler.step()
            print('Current Learning Rate: {}'.format(
                encoder_scheduler.get_lr()))

        # Print progress
        if iteration % print_every == 0 or iteration in [1, 5, 10]:
            print_loss_avg = print_loss / print_every
            print("Epoch: {}, Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                curr_epoch, iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

            # log the average loss every %print_progress% number of iterations
            train_writer.add_scalar('avg_loss', print_loss_avg, iteration)

        # compute the validation loss
        if iteration % print_every == 0:
            # current validation batch
            val_batch = validation_batches[val_count]
            input_variable, lengths, target_variable, mask, max_target_len = val_batch
            val_count += 1

            # compute validation loss
            dev_loss = validate(input_variable, lengths, target_variable, mask,
                                max_target_len, encoder, decoder, embedding, batch_size)

            print("Epoch: {}, Iteration: {}; Val Loss: {:.4f}".format(
                curr_epoch, iteration, dev_loss))

            # log the average loss every %print_progress% number of iterations
            val_writer.add_scalar('avg_loss', dev_loss, iteration)

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(
                encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'en_sch': encoder_scheduler.state_dict(),
                'de_sch': decoder_scheduler.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = - \
        torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


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


####################
# Main Section
####################
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# set the random seed
SEED = 15
random.seed(SEED)

# Model configuration
model_name = 'cb_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 512
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.2
batch_size = 128

# corpus_name = 'cornell movie-dialogs corpus'
# filename = 'formatted_movie_lines.txt'
corpus_name = "combined"
filename = 'combined_formatted.txt'

# input filepath
input_filepath = '../data/{}/{}'.format(corpus_name, filename)

# Set checkpoint to load from; set to None if starting from scratch
# loadFilename = "../pre_trained_models/pretrained_model_checkpoint.tar"
loadFilename = None

# tensorboardX summary writer for visualizations
train_writer = SummaryWriter("../evaluation/runs/test2/train", flush_secs=10)
val_writer = SummaryWriter('../evaluation/runs/test2/val', flush_secs=10)

# Initial data formatting and writing (only done once)
# create_formatted_file(corpus)

# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = load_prepare_data(corpus_name, input_filepath)

# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

# trim the vocabulary to the lower word count limit
voc, pairs = trim_rare_words(voc, pairs, MIN_COUNT)

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    encoder_scheduler_sd = checkpoint['en_sch']
    decoder_scheduler_sd = checkpoint['de_sch']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
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

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
l2_penalty = 0.001
n_iteration = 4000
print_every = 25
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(
    encoder.parameters(), lr=learning_rate, weight_decay=l2_penalty)
decoder_optimizer = optim.Adam(decoder.parameters(
), lr=learning_rate * decoder_learning_ratio, weight_decay=l2_penalty)

# Initialize learning rate schedulers
encoder_scheduler = optim.lr_scheduler.MultiStepLR(
    encoder_optimizer, milestones=[5, 10], gamma=0.1)
decoder_scheduler = optim.lr_scheduler.MultiStepLR(
    decoder_optimizer, milestones=[5, 10], gamma=0.1)

if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)
    encoder_scheduler.load_state_dict(encoder_scheduler_sd)
    decoder_scheduler.load_state_dict(decoder_scheduler_sd)

# Run training iterations
start_time = datetime.now()
print("Starting Training!", str(start_time)[
      str(start_time).find(" ") + 1:str(start_time).rfind(".")])

trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, embedding,
           encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename)

# write tensorboard data to file
train_writer.close()
val_writer.close()

delta = datetime.now() - start_time
print("Training took: {}".format(str(delta)[:str(delta).rfind(".")]))

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = TopKSearchDecoder(encoder, decoder, 5)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)
