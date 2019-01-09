import torch

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class TopKSearchDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, k):
        super(TopKSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.k = k

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(
            1, 1, device=device, dtype=torch.long) * SOS_token

        # initial sequence, score
        init_seq = torch.zeros([0], device=device, dtype=torch.long)
        init_score = 1.0

        # start value
        sequences = [[init_seq, init_score, decoder_input, decoder_hidden]]

        # Iteratively decode one word token at a time
        for _ in range(max_length):

            # storing intermediate candidates of current iteration
            temp_sequences = []

            for sequence in sequences:
                # split sequence
                seq, score, decoder_input, decoder_hidden = sequence

                # forward pass through decoder
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                # obtain top-k most likely word tokens and their softmax score
                new_sequence = self.beam_search(
                    seq, score, self.k, decoder_output, decoder_hidden)
                temp_sequences += new_sequence

            # order all candidates by score
            ordered = sorted(temp_sequences, key=lambda tup: tup[1])

            # select top-k candidates of this round
            sequences = ordered[:self.k]

        # tokens which are returned and transformed to words
        tokens = [element[0].tolist() for element in sequences]
        all_scores = [element[1] for element in sequences]

        # return collections of word tokens and scores
        return tokens, all_scores

    def beam_search(self, prev_seq, prev_score, k, decoder_output, decoder_hidden):
        from math import log
        new_candidates = list()

        # find top-k candidates for current output
        scores, candidates = decoder_output.topk(k=k, dim=1)

        # reshape to k-separate rank one tensors
        candidates = candidates.view(self.k, 1)
        scores = scores.view(self.k, 1)

        # concat the top-k current outputs to the previous candidate
        for score, candidate in zip(scores, candidates):
            # store the new sequence, score, input, hidden state
            new_seq = torch.cat((prev_seq, candidate), dim=0)
            new_score = prev_score * -log(score)
            new_input = torch.unsqueeze(candidate, 0)

            new_candidates.append(
                [new_seq, new_score, new_input, decoder_hidden])

        # order all new candidates by score
        ordered = sorted(new_candidates, key=lambda tup: tup[1])

        # select k best candidate sequences
        sequences = ordered[: k]

        return sequences
