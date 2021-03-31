import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import sys
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_name = sys.argv[1]
embedding_size = 300
data = load_data(dataset_name + '_tweets')
V = data['all_words']
new_tweets = data['tweets']
Y = data['Y']

class Word2Vec(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_1 = nn.Linear(len(V),embedding_size)
    self.hidden_2 = nn.Linear(embedding_size,len(V))
    self.logsoftmax = nn.LogSoftmax()


  def forward(self, one_hot):
    out = self.hidden_1(one_hot)
    # print(out.shape)
    # out = self.hidden_2(out)
    # out = self.logsoftmax(out)
    return out

def word_to_one_hot(word):
    one_hot = torch.zeros(len(V))
    for i,w in enumerate(V):
        if word == w:
            one_hot[i] = 1
            break
    assert(torch.sum(one_hot)==1)
    return one_hot


model = Word2Vec()
model.load_state_dict(torch.load(dataset_name + '_embeddings_model'))
model.eval()
sen_len = int(sum([len(sentence) for sentence in new_tweets])/len(new_tweets))

embeddings = torch.zeros((len(new_tweets),sen_len,embedding_size))
# Y = torch.zeros((len(new_tweets)))
print('Started getting embeddings:')
for i,sentence in enumerate(new_tweets):
    j = 0
    cnt = 0
    for word in sentence:
        if j == sen_len:
            break
        if word in V:
            # print(model(word_to_one_hot(word)).shape)
            embeddings[i][j] = model(word_to_one_hot(word))
            cnt += 1
        else:
            print('-----ERR------')
        j += 1
    # if len(sentence) == 95:
    # print(sen_len,len(sentence), cnt)
    while j<sen_len:
        embeddings[i][j] = torch.rand(embedding_size)
        j += 1
print('Finished getting embeddings')
Y = torch.from_numpy(Y)
print(torch.sum(Y))
import random
frac = 0.8
num_data = len(new_tweets)
num_train = int(frac * num_data)
num_test = int((num_data - num_train)/2)
num_val = num_data - (num_test + num_train)
indices = np.arange(num_data)
random.shuffle(indices)
indices_train = indices[:num_train]
indices_val = indices[num_train:num_train+num_val]
indices_test = indices[num_train+num_val:num_train+num_val+num_test]

data_split = {}
data_split['X'] = embeddings[indices_train]
data_split['Y'] = Y[indices_train]


val = {}
val['X'] = embeddings[indices_val]
val['Y'] = Y[indices_val]

data_split['val'] = val

test = {}
test['X'] = embeddings[indices_test]
test['Y'] = Y[indices_test]
data_split['test'] = test
#
# with open('hindi_data' + '.pkl', 'wb') as f:
#     pickle.dump(data_split, f, pickle.HIGHEST_PROTOCOL)

save_data(data_split,dataset_name + '_data')

