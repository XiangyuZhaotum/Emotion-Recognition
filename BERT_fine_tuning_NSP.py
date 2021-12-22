'''

Authors: Xiangyu Zhao
Date: Saturday, 18th December, 2021
Title: Use NSP task to fine tune BERT

'''

from transformers import BertTokenizer ,BertForNextSentencePrediction
import torch
import pandas as pd
from os.path import abspath
import random
from transformers import AdamW
from tqdm import tqdm

'''
https://github.com/jamescalam/transformers/blob/main/course/training/06_nsp_training.ipynb
'''

tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-german-cased')

text_path = abspath("./dataset/features/acoustic_features/audio_features.csv")
text = pd.read_csv(text_path, usecols=['ID', 'Gender', 'Transcript'])
# text_female = text[text['Gender'] == 'f']

# sentences = text_female['Transcript'][0].split('.')

# create 50/50 NSP training data
sentence_a = []
sentence_b = []
label = []

for transcript in text['Transcript']:
    sentences = [sentence for sentence in transcript.split('.') if sentence != '']
    num_sentences = len(sentences)
    if num_sentences > 1:
        for i in range(10):
            start = random.randint(0, num_sentences-2)
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(sentences[start+1])
                label.append(0)
            else:
                index = random.randint(0, num_sentences-1)
                # this is NotNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(sentences[index])
                label.append(1)

inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs['labels'] = torch.LongTensor([label]).T


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = ConversationDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# activate training mode
model.train()
optim = AdamW(model.parameters(), lr=5e-6)

epochs = 1
for epoch in range(epochs):
    loop = tqdm(loader, leave=True)
    for batch in loop:

        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

torch.save(model, 'finetuned_BERT_NSP')
print('finish')
