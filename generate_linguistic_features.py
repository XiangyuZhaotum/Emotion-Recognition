'''

Authors: Xiangyu Zhao
Date: Tuesday, 31th November, 2021
Title: Generate linguistic features from transcripts using German BERT

'''

from sentence_transformers import SentenceTransformer, models
import numpy as np
import pandas as pd
from os.path import abspath
import os

def extract_sentence_bert_features(data, embedding_model):
    sentences = data
    '''
    # Get German BERT model
    embedding_model = models.Transformer('bert-base-german-cased')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=False,
                                   pooling_mode_cls_token=False, pooling_mode_max_tokens=True)
    model = SentenceTransformer(modules=[embedding_model, pooling_model])
    print("Max Sequence Length: ", model.max_seq_length)

    # Change the length to 512
    model.max_seq_length = 512
    print("Max Sequence Length: ", model.max_seq_length)
    '''
    #model = SentenceTransformer('bert-base-german-cased')
    # Get sentence embeddings
    sentence_embeddings = embedding_model.encode(sentences)

    return sentence_embeddings

def get_all_text_features(text_path, output_path, embedding_model):

    print("Extracting linguistic features")
    data = pd.read_csv(text_path, usecols=['ID', 'Name', 'Gender', 'Transcript'])

    for i, row in data.iterrows():
        print("Couple ID: " + str(row['ID']))
        print("Annotation file name:" + row['Name'])
        print("Gender:" + row['Gender'])
        sentence_embeddings = extract_sentence_bert_features(row['Transcript'], embedding_model)
        output_file = output_path + '/' + str(row['ID']) + '_' + row['Gender'] + '_bert.csv'
        columns = ['Col ' + str(i) for i in range(len(sentence_embeddings))]
        text_pd = pd.DataFrame(sentence_embeddings.reshape(1, -1), columns=columns)
        text_pd.to_csv(output_file, index=False)

    print(str(i) + "transcripts")


# Main linguistic feature extraction
text_path = abspath("./dataset/features/acoustic_features/audio_features.csv")
output_path = abspath("./dataset/features/german_bert_features")
model = SentenceTransformer('bert-base-german-cased')
get_all_text_features(text_path, output_path, model)

print('extract linguistic features')