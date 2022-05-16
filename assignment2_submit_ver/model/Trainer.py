import torch
import torch.nn as nn
from . import models
from . import utils
import random


def train(input_var, lengths, target_var, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip):
          
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_var, lengths)
    decoder_input = torch.LongTensor([[0 for _ in range(batch_size)]])
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
        mask_loss, nTotal = models.maskNLLLoss(decoder_output, target_var[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal
    loss.backward()
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()
    return sum(print_losses) / n_totals
    

def trainIters(model_name, input_vocab, output_vocab, input_var, output_var, encoder, decoder,
               encoder_optimizer, decoder_optimizer, en_embed, zh_embed, encoder_n_layers, 
               decoder_n_layers, n_iteration, batch_size, clip, print_every):
    print("Initializing...")

    training_batches = [utils.batch2TrainData(input_vocab, output_vocab, [random.choice(input_var) for _ in range(batch_size)], [random.choice(output_var) for _ in range(batch_size)]) for _ in range(n_iteration)] 
    start_iteration = 1
    print_loss = 0

    print("Start training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_var, lengths, target_var, mask, max_target_len = training_batch

        loss = train(input_var, lengths, target_var, mask, max_target_len, encoder, decoder, en_embed,
                     encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0


class Searcher(nn.Module):
    def __init__(self, encoder, decoder):
        super(Searcher, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.zeros(1,1, dtype=torch.long)
        all_tokens = torch.zeros([0], dtype=torch.long)
        all_scores = torch.zeros([0])
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, input_voc, output_voc, sentence, max_length):
    indexes = [utils.sentence2index(input_voc, sentence)]
    input_length = torch.tensor([len(index) for index in indexes])
    input_seq = torch.LongTensor(indexes).transpose(0, 1)
    #print(input_length)
    tokens, scores = searcher(input_seq, input_length, input_length)
    decoded_words = utils.index2sentence(output_voc, tokens.tolist())
    return decoded_words

