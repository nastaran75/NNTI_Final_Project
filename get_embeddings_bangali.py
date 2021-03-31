
import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import random
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# !/usr/bin/env python
# coding: utf-8

# # Task 1: Word Embeddings (10 points)
#
# This notebook will guide you through all steps necessary to train a word2vec model (Detailed description in the PDF).

# ## Imports
#
# This code block is reserved for your imports.
#
# You are free to use the following packages:
#
# (List of packages)

# In[1]:


# Imports
import numpy as np
import csv
from utils import *

# # 1.1 Get the data (0.5 points)
#
# The Hindi portion HASOC corpus from [github.io](https://hasocfire.github.io/hasoc/2019/dataset.html) is already available in the repo, at data/hindi_hatespeech.tsv . Load it into a data structure of your choice. Then, split off a small part of the corpus as a development set (~100 data points).
#
# If you are using Colab the first two lines will let you upload folders or files from your local file system.

# In[2]:


# TODO: implement!

# from google.colab import files
# uploaded = files.upload()


data = open('bengali_hatespeech.csv')
data = csv.reader(data, delimiter='\t')
tweets = [line for line in data]  # [:1000]

# ## 1.2 Data preparation (0.5 + 0.5 points)
#
# * Prepare the data by removing everything that does not contain information.
# User names (starting with '@') and punctuation symbols clearly do not convey information, but we also want to get rid of so-called [stopwords](https://en.wikipedia.org/wiki/Stop_word), i. e. words that have little to no semantic content (and, but, yes, the...). Hindi stopwords can be found [here](https://github.com/stopwords-iso/stopwords-hi/blob/master/stopwords-hi.txt) Then, standardize the spelling by lowercasing all words.
# Do this for the development section of the corpus for now.
#
# * What about hashtags (starting with '#') and emojis? Should they be removed too? Justify your answer in the report, and explain how you accounted for this in your implementation.

# In[3]:


# TODO: implement!
stopwords = [line[:-2] for line in open('bangali_stopwords')]


# english_stopwords = [line[:-2] for line in open('english_stopwords')]
# print(stopwords)
def valid(word):
    if word.startswith('@') or word.startswith('(@') or word.startswith('.@'):
        return False
    if word in [',', ';', ':', '.', '!', '?', '\'', '\""', '-', '_', '/', '(', ')', '[', ']', '...', '*']:
        return False
    if word in stopwords:
        return False
    #     if word in english_stopwords:
    #         return False
    if word.startswith('http'):
        return False
    return True


new_tweets = []
labels = []
indices = np.arange(len(tweets))
random.shuffle(indices)
num_0 = 0
num_1 = 0
# tweets = tweets[indices]
for i in indices:
    if num_0>2500 and num_1>2500:
        break
    # print(tweets[i][0].split(',')[1])
    tweet = tweets[i][0].split(',')
    if tweet[1] not in ['0','1']:
        continue
    if i==0:
        continue
    new_sentence = []
    # print(tweet)
    if tweet[1] == '0':
        num_0 += 1
        if num_0>2500:
            continue
    if tweet[1] == '1':
        num_1 += 1
        if num_1>2500:
            continue

    labels.append(int(tweet[1]))
    for word in tweet[0].split():
#         print(word)
        if valid(word):
            cnt = 0
            while(word[cnt]==' '):
                cnt += 1
            word = word[cnt:]
            new_sentence.append(word.lower())
#                 print(word)
    new_tweets.append(new_sentence)
# ## 1.3 Build the vocabulary (0.5 + 0.5 points)
#
# The input to the first layer of word2vec is an one-hot encoding of the current word. The output od the model is then compared to a numeric class label of the words within the size of the skip-gram window. Now
#
# * Compile a list of all words in the development section of your corpus and save it in a variable ```V```.

# In[4]:


# TODO: implement!
word_dict = {}
all_words = []
for tweet in new_tweets:
    for word in tweet:
        if not word in word_dict.keys():
            all_words.append(word)
            word_dict[word] = 1
        else:
            word_dict[word] += 1
V = all_words
# print(V)


# * Then, write a function ```word_to_one_hot``` that returns a one-hot encoding of an arbitrary word in the vocabulary. The size of the one-hot encoding should be ```len(v)```.

# In[5]:


# TODO: implement!
def word_to_one_hot(word):
    one_hot = np.zeros(len(V))
    for i, w in enumerate(V):
        if word == w:
            one_hot[i] = 1
            break
    assert (np.sum(one_hot) == 1)
    return one_hot


# ## 1.4 Subsampling (0.5 points)
#
# The probability to keep a word in a context is given by:
#
# $P_{keep}(w_i) = \Big(\sqrt{\frac{z(w_i)}{0.001}}+1\Big) \cdot \frac{0.001}{z(w_i)}$
#
# Where $z(w_i)$ is the relative frequency of the word $w_i$ in the corpus. Now,
# * Calculate word frequencies
# * Define a function ```sampling_prob``` that takes a word (string) as input and returns the probabiliy to **keep** the word in a context.

# In[6]:


# TODO: implement!
# word_dict calculated above
import math

Z = word_dict


# print(Z)
def sampling_prob(word):
    return (math.sqrt(Z[word] / 0.001) + 1) * (0.001 / Z[word])


# # 1.5 Skip-Grams (1 point)
#
# Now that you have the vocabulary and one-hot encodings at hand, you can start to do the actual work. The skip gram model requires training data of the shape ```(current_word, context)```, with ```context``` being the words before and/or after ```current_word``` within ```window_size```.
#
# * Have closer look on the original paper. If you feel to understand how skip-gram works, implement a function ```get_target_context``` that takes a sentence as input and [yield](https://docs.python.org/3.9/reference/simple_stmts.html#the-yield-statement)s a ```(current_word, context)```.
#
# * Use your ```sampling_prob``` function to drop words from contexts as you sample them.

# In[7]:


# TODO: implement!
def get_target_context(sentence):
    context_array = []
    for i, word in enumerate(sentence):
        for w in range(-window_size, window_size + 1):
            context_idx = i + w
            if context_idx < 0 or context_idx >= len(sentence) or context_idx == i:
                continue
            keep = np.random.choice(2, 1, p=[1.0 - sampling_prob(word), sampling_prob(word)])
            if keep:
                context_array.append((word, sentence[context_idx]))
    return context_array


# # 1.6 Hyperparameters (0.5 points)
#
# According to the word2vec paper, what would be a good choice for the following hyperparameters?
#
# * Embedding dimension
# * Window size
#
# Initialize them in a dictionary or as independent variables in the code block below.

# In[8]:


# Set hyperparameters
window_size = 5
embedding_size = 300

# More hyperparameters
learning_rate = 0.007
epochs = 50

# # 1.7 Pytorch Module (0.5 + 0.5 + 0.5 points)
#
# Pytorch provides a wrapper for your fancy and super-complex models: [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). The code block below contains a skeleton for such a wrapper. Now,
#
# * Initialize the two weight matrices of word2vec as fields of the class.
#
# * Override the ```forward``` method of this class. It should take a one-hot encoding as input, perform the matrix multiplications, and finally apply a log softmax on the output layer.
#
# * Initialize the model and save its weights in a variable. The Pytorch documentation will tell you how to do that.

# In[9]:


# Create model
import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_1 = nn.Linear(len(V), embedding_size)
        self.hidden_2 = nn.Linear(embedding_size, len(V))
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, one_hot):
        out = self.hidden_1(one_hot)
        out = self.hidden_2(out)
        out = self.logsoftmax(out)
        return out


# # 1.8 Loss function and optimizer (0.5 points)
#
# Initialize variables with [optimizer](https://pytorch.org/docs/stable/optim.html#module-torch.optim) and loss function. You can take what is used in the word2vec paper, but you can use alternative optimizers/loss functions if you explain your choice in the report.

# In[10]:


# Define optimizer and loss
import torch

my_model = Word2Vec().to(device)
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


# # 1.9 Training the model (3 points)
#
# As everything is prepared, implement a training loop that performs several passes of the data set through the model. You are free to do this as you please, but your code should:
#
# * Load the weights saved in 1.6 at the start of every execution of the code block
# * Print the accumulated loss at least after every epoch (the accumulate loss should be reset after every epoch)
# * Define a criterion for the training procedure to terminate if a certain loss value is reached. You can find the threshold by observing the loss for the development set.
#
# You can play around with the number of epochs and the learning rate.

# In[ ]:


# Define train procedure

# load initial weights
def train():
    print("Training started")
    X = []
    Y = []
    for sentence in new_tweets:
        for tuples in get_target_context(sentence):
            X.append(word_to_one_hot(tuples[0]))
            Y.append(int(np.where(word_to_one_hot(tuples[1]) == 1)[0][0]))


    num_data = len(X)
    num_train = int(0.8 * num_data)
    X_train = torch.tensor(X[:num_train])
    Y_train = torch.tensor(Y[:num_train])


    batch_size = 64
    num_batches = int(len(X_train) / batch_size)
    for epoch in range(epochs):
        print(epoch)
        epoch_loss = 0
        for i in range(num_batches):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size].float().to(device)
            Y_batch = Y_train[i * batch_size:(i + 1) * batch_size].to(device)
            # print(X_batch)
            # print(Y_batch)
            optimizer.zero_grad()
            loss = criterion(my_model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        print('train loss : ', float(epoch_loss / num_batches))



train()

print("Training finished")

# # 1.10 Train on the full dataset (0.5 points)
#
# Now, go back to 1.1 and remove the restriction on the number of sentences in your corpus. Then, reexecute code blocks 1.2, 1.3 and 1.6 (or those relevant if you created additional ones).
#
# * Then, retrain your model on the complete dataset.
#
# * Now, the input weights of the model contain the desired word embeddings! Save them together with the corresponding vocabulary items (Pytorch provides a nice [functionality](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for this).

# In[ ]:


for param_tensor in my_model.state_dict():
    print(param_tensor, "\t", my_model.state_dict()[param_tensor].size())
torch.save(my_model.state_dict(), 'bangali_embeddings_model')

data = {}
data['tweets'] = new_tweets
data['Y'] = np.array(labels)
data['all_words'] = np.array(V)
save_data(data, 'bangali_tweets')




