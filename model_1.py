# This is basic sequence to vector model for cloz-style question answering...
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

def map_original(article,entity_map):
    for i in range(len(article)):
        w = article[i]
        if w in entity_map:
            article[i] = entity_map[w]
    return article

def tokenise(line):
    line = line.translate(translator)
    temp = line.strip().split(' ')
    temp = [a for a in temp if a!='']
    return temp

def insert(art,w_c,w2i,i2w):
    for w in art:
        if w not in w2i:
            w2i[w] = w_c
            i2w[w_c] = w_c
            w_c+=1
    return w2i,i2w,w_c


def build_vocabulary(dir): # create a word2index and index2word maps for documents in train, test and validation set
    art_ques_ans = []
    w_c = 0
    w2i = {}
    i2w = {}
    translator = str.maketrans('', '', string.punctuation)
    for r,d,f in os.walk(dir):
        print (r)
        for file in f:
            flag = 0
            article = []
            entity_map = {}
            question = []
            answer = ''
            with open(r+"/"+file) as fs:
                for line in fs:
                    flag+=1
                    #print (line.strip(),flag)
                    if flag==3:
                        article = tokenise(line)
                    elif flag==5:
                        question = tokenise(line)
                    elif flag==7:
                        answer = line.strip().replace('@','')
                    elif flag>=9:
                        key,val = line.strip().replace('@','').split(':')
                        entity_map[key] = val
                    else:
                        continue
            article = map_original(article,entity_map)
            question = map_original(question,entity_map)
            answer = entity_map[answer]
            w2i,i2w,w_c = insert(article,w_c,w2i,i2w)
            w2i,i2w,w_c = insert(question,w_c,w2i,i2w)
            art_ques_ans.append((article,question,answer))
            break
        break
    w2i['del'] = w_c
    i2w[w_c] = 'del'
    return art_ques_ans, w2i, i2w

art_ques_ans, w2i, i2w = build_vocabulary("cnn/questions/training")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1) # view is same as reshape
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def IndexFromArticle(article):
    return [w2i[w] for w in article]

def ObtainArticleQuestionTensor(article, question):
    a_v = IndexFromArticle(article)
    a_v.append(w2i['del'])
    q_v = IndexFromArticle(question)
    t_v = a_v + q_v
    return torch.tensor(t_v, dtype=torch.long).view(-1, 1)

def EncodeTrainExamples(sample):
    ans = w2i(sample[2])
    target_tensor = torch.LongTensor([pair[1]])
    input_tensor = ObtainArticleQuestionTensor(sample[0],sample[1])
    return input_tensor,target_tensor


def train(encoder, w2i, iter = 5000, learning_rate = 0.01):
    encode_hidden = encoder.initHidden()
    vector_optimizer = optim.SGD(vector.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for i in range(iter):
        sample = random.choice(art_ques_ans)
        input_tensor, target_tensor = EncodeTrainExamples(sample)
        input_length = input_tensor.size(0)
        for ei in range(input_length):
            output, encoder_hidden = vector(input_tensor[ei],encoder_hidden)

        loss = criterion(output, target_tensor)

        vector_optimizer.zero_grad()

        loss.backward()
        vector_optimizer.step()
