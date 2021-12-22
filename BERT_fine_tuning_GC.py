'''

Authors: Xiangyu Zhao
Date: Tuesday, 21st December, 2021
Title: Use gender classification taks to fine tune BERT model

'''

import pandas as pd
from os.path import abspath
import numpy as np
from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer
import torch
import torch.nn.functional as F
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf


text_path = abspath("./dataset/features/acoustic_features/audio_features.csv")
text = pd.read_csv(text_path, usecols=['ID', 'Gender', 'Transcript'])
label = text['Gender']
label[label == 'f'] = 0
label[label == 'm'] = 1
data = text.drop(['Gender'], axis=1)

# train test split
data_train, data_test, label_train, label_test = train_test_split(data, label)

# load pretrained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-german-cased')

# set training mode
model.train()

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
encoded_train = tokenizer(data_train['Transcript'].values.tolist(), padding=True, truncation=True, return_tensors='pt')
encoded_test = tokenizer(data_test['Transcript'].values.tolist(), padding=True, truncation=True, return_tensors='pt')

train_input_id = encoded_train['input_ids']
train_attention_mask = encoded_train['attention_mask']
train_id = train_input_id[:]
train_am = train_attention_mask[:]

test_input_id = encoded_test['input_ids']
test_attention_mask = encoded_test['attention_mask']
test_id = test_input_id[:]
test_am = test_attention_mask[:]

train_labels = torch.tensor(label_train.values.tolist())
train_labels = train_labels.type(torch.LongTensor)

test_labels = torch.tensor(label_test.values.tolist())
test_labels = test_labels.type(torch.LongTensor)

optimizer = AdamW(model.parameters(), lr=1e-5)


# train
n_epochs = 1
batch_size = 4
for epoch in range(n_epochs):
    permutation = torch.randperm(train_id.size()[0])
    for i in range(0, train_id.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_x, batch_y, batch_am = train_id[indices], train_labels[indices], train_am[indices]
        outputs = model(batch_x, attention_mask=batch_am, labels=batch_y)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

torch.save(model, 'finetuned_BERT_GC')

# model = torch.load('finetuned_BERT_hs')
# test
model.eval()
batch_size = 8
prediction = np.empty((0, 2))
ids = torch.tensor(range(test_id.size()[0]))
for i in range(0, test_id.size()[0], batch_size):
    indices = ids[i:i+batch_size]
    batch_x1, batch_am1 = test_id[indices], test_am[indices]
    pred = model(batch_x1, batch_am1)
    pt_predictions = F.softmax(pred[0], dim=-1)
    prediction = np.append(prediction, pt_predictions.detach().numpy(), axis=0)

predict = np.argmax(prediction, axis=1)
accuracy = accuracy_score(label_test.values.tolist(), predict)
print('finish')