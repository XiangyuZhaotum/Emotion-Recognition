import pandas as pd
from os.path import abspath
import numpy as np
from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer
import torch
import torch.nn.functional as F
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

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
data_upsample = resample(data[data['good_bad'] == 1], replace=True, n_samples=data[data['good_bad'] == 0].shape[0])
data_balanced = np.vstack((data[data['good_bad'] == 0], data_upsample))
data_balanced = pd.DataFrame(data_balanced, columns=['Transcript', 'good_bad'])

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-german-cased')
model.train()

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
encoded = tokenizer(data_balanced['Transcript'].values.tolist(), padding=True, truncation=True, return_tensors='pt')

input_id = encoded['input_ids']
attention_mask = encoded['attention_mask']
train_id = input_id[212:392]
train_am = attention_mask[212:392]
test_id = input_id[280:320]
test_am = attention_mask[280:320]
train = data_balanced.iloc[212:392]
test = data_balanced.iloc[280:320]
labels = torch.tensor(train['good_bad'].values.tolist())
labels = labels.type(torch.LongTensor)

optimizer = AdamW(model.parameters(), lr=1e-5)


# train
n_epochs = 1
batch_size = 4
for epoch in range(n_epochs):
    permutation = torch.randperm(train_id.size()[0])
    for i in range(0, train_id.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_x, batch_y, batch_am = train_id[indices], labels[indices], train_am[indices]
        outputs = model(batch_x, attention_mask=batch_am, labels=batch_y)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

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
