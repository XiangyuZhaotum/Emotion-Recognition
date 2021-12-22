'''

Authors: Xiangyu Zhao
Date: Tuesday, 14th December, 2021
Title: Use emotion recognition task to fine tune BERT

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

'''
https://towardsdatascience.com/first-time-using-and-fine-tuning-the-bert-framework-for-classification-799def68a5e4
'''

text_path = abspath("./dataset/features/acoustic_features/audio_features.csv")
text = pd.read_csv(text_path, usecols=['ID', 'Gender', 'Transcript'])
text_female = text[text['Gender'] == 'f']
label_path = abspath("./dataset/labels/normalize_german_bert_features_binary_labels/linguistic.csv")
label = pd.read_csv(label_path, usecols=['ID', 'Gender', 'good_bad'])
label_female = label[label['Gender'] == 'f']
data = pd.merge(text_female, label_female, on=['ID'])
data = data.drop(['ID', 'Gender_x', 'Gender_y'], axis=1)

# train test split
data_train, data_test, label_train, label_test = train_test_split(data, label_female)
data_train_upsample = resample(data_train[data_train['good_bad'] == 1], replace=True,
                               n_samples=data_train[data_train['good_bad'] == 0].shape[0])
data_train_balanced = np.vstack((data_train[data_train['good_bad'] == 0], data_train_upsample))
data_train_balanced = pd.DataFrame(data_train_balanced, columns=['Transcript', 'good_bad'])

data_test_upsample = resample(data_test[data_test['good_bad'] == 1], replace=True,
                              n_samples=data_test[data_test['good_bad'] == 0].shape[0])
data_test_balanced = np.vstack((data_test[data_test['good_bad'] == 0], data_test_upsample))
data_test_balanced = pd.DataFrame(data_test_balanced, columns=['Transcript', 'good_bad'])

'''
data_upsample = resample(data[data['happy_sad'] == 1], replace=True, n_samples=data[data['happy_sad'] == 0].shape[0])
data_balanced = np.vstack((data[data['happy_sad'] == 0], data_upsample))
data_balanced = pd.DataFrame(data_balanced, columns=['Transcript', 'happy_sad'])
data_train = pd.concat([data_balanced[:212], data_balanced[394:]])
data_test = data_balanced[212:394]
'''
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-german-cased')
model.train()

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
encoded_train = tokenizer(data_train_balanced['Transcript'].values.tolist(), padding=True, truncation=True, return_tensors='pt')
encoded_test = tokenizer(data_test_balanced['Transcript'].values.tolist(), padding=True, truncation=True, return_tensors='pt')

train_input_id = encoded_train['input_ids']
train_attention_mask = encoded_train['attention_mask']
train_id = train_input_id[:]
train_am = train_attention_mask[:]
train = data_train_balanced

test_input_id = encoded_test['input_ids']
test_attention_mask = encoded_test['attention_mask']
test_id = test_input_id[:]
test_am = test_attention_mask[:]
test = data_test_balanced


train_labels = torch.tensor(train['good_bad'].values.tolist())
train_labels = train_labels.type(torch.LongTensor)

test_labels = torch.tensor(test['good_bad'].values.tolist())
test_labels = test_labels.type(torch.LongTensor)

optimizer = AdamW(model.parameters(), lr=1e-5)

'''
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

torch.save(model, 'finetuned_BERT_ga')

'''

model = torch.load('finetuned_BERT_ga')
test_id = train_id
test_am = train_am
test = train

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
accuracy = accuracy_score(test['good_bad'].values.tolist(), predict)
print('finish')
