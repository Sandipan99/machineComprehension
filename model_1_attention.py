# This is basic sequence to vector model for cloz-style question answering with local-attention (Manning et. al. 1508.04025v5)
from io import open
import unicodedata
import string
import re
import random
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model_1 import build_vocabulary, EncodeTrainExamples, randomEvaluate

art_ques_ans, w2i, i2w = build_vocabulary("cnn/questions/training")

class Encoder_attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder_attention, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        for ei in range(input.size(0)):
            inp = input[ei]
            embedded = self.embedding(inp).view(1,1,-1) # view is same as reshape
            output = embedded
            output, hidden = self.gru(output, hidden)

        output = torch.cat((output[0],hidden[0]),1)
        output = self.out(output[0]).view(1,-1)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def train(encoder, iter = 5000, learning_rate = 0.01):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for i in range(iter):
        encoder_hidden = encoder.initHidden()
        sample = random.choice(art_ques_ans)
        input_tensor, target_tensor = EncodeTrainExamples(sample)
        output = encoder(input_tensor,encoder_hidden)

        loss = criterion(output, target_tensor)

        print('loss:',loss)
        encoder_optimizer.zero_grad()

        loss.backward(retain_graph=True)
        encoder_optimizer.step()

if __name__=="__main__":
    hidden_size = 50
    input_size = len(w2i)
    output_size = len(w2i)
    encoder = Encoder_attention(input_size, hidden_size, output_size)
    train(encoder)
