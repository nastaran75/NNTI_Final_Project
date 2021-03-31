import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import sys
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.logsoftmax = nn.LogSoftmax()

    def forward(self, embedded):
        # text = [batch size, sent len]

        # embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # print(embedded.shape)

        # embedded = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        # print(conved_0.shape)
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        # print(conved_1.shape)
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        # print(conved_2.shape)

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        fc =  self.fc(cat)
        return self.logsoftmax(fc)


def train(data_file):
    data = load_data(data_file)
    X = data['X'].to(device)
    print(X.shape)
    Y = data['Y'].to(device)
    print(torch.sum(Y))
    val_X = data['val']['X'].to(device)
    val_Y = data['val']['Y'].to(device)
    print(torch.sum(val_Y))

    batch_size = 32
    num_batches = int(X.shape[0]/batch_size)

    N_FILTERS = 100  # hyperparameterr
    FILTER_SIZES = [3, 4, 5]
    DROPOUT = 0.5
    INPUT_DIM = data['X'].shape[1]
    EMBEDDING_DIM = 300
    OUTPUT_DIM = 2
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)
    #finetune
    if dataset_name == 'bangali':
        model.load_state_dict(torch.load('hindi_classifier_model'))
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.005)
    loss_func = torch.nn.NLLLoss()

    num_epochs = 5
    min_val_loss = 100
    for epoch in range(num_epochs):
        print(epoch)
        epoch_loss = 0
        for i in range(num_batches):
            X_batch = X[i * batch_size:(i+1) * batch_size]
            Y_batch = Y[i * batch_size:(i + 1) * batch_size].long()
            optimizer.zero_grad()
            loss = loss_func(model(X_batch),Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        print('train_loss : ',epoch_loss/num_batches)
        with torch.no_grad():
            val_loss = loss_func(model(val_X),val_Y.long())
            if val_loss < min_val_loss:
                torch.save(model.state_dict(), dataset_name + '_classifier_model')
                min_val_loss = val_loss

        print('validation_loss : ' , float(val_loss))

def test(data_file):
    data = load_data(data_file)
    X = data['test']['X'].to(device)
    Y = data['test']['Y'].to(device)
    N_FILTERS = 100  # hyperparameterr
    FILTER_SIZES = [3, 4, 5]
    DROPOUT = 0
    INPUT_DIM = data['X'].shape[1]
    EMBEDDING_DIM = 300
    OUTPUT_DIM = 2
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)
    model.load_state_dict(torch.load(dataset_name + '_classifier_model'))
    with torch.no_grad():
        pred = torch.argmax(model(X),dim=1)
    print(np.mean(np.array([p==y for p,y in zip(pred.cpu().data.numpy(),Y.cpu().data.numpy())])))

dataset_name = sys.argv[1]
data_file = dataset_name + '_data'
train(data_file)
test(data_file)




