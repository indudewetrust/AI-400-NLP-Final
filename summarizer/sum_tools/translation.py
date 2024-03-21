# Adapted from the Pytorch tutorial https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Data for translation found at https://www.manythings.org/anki/ from https://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size= hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = F.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention_scores, dim=1)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        # Attention mechanism
        self.attention = Attention(hidden_size)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        embedded = self.embedding(decoder_input)
        embedded = F.relu(embedded)

        # Calculate attention weights
        attention_weights = self.attention(decoder_hidden[-1], encoder_outputs)
        attention_weights = attention_weights.unsqueeze(1)

        # Apply attention to encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)

        # Concatenate context vector with embedded input
        decoder_input_with_context = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(decoder_input_with_context, decoder_hidden)

        output = self.out(output)
        return output, hidden

