'''

Authors: Xiangyu Zhao
Date: Wednesday, 1st December, 2021
Title: Attach linguistic features csv files into the existing acoustic feature csv

'''

from os.path import abspath
import pandas as pd

acoustic_features_path = abspath("./dataset/features/acoustic_features/audio_features.csv")
linguistic_features_path = abspath("./dataset/features/german_bert_features")
combined_features_path = abspath("./dataset/features/combined_german_bert_features")

acoustic_features = pd.read_csv(acoustic_features_path)
acoustic_columns = acoustic_features.columns
new_columns = ['Col' + str(i) for i in range(768)]
linguistic_feature_total = pd.DataFrame()

for index, row in acoustic_features.iterrows():
    ID = row['ID']
    Gender = row['Gender']
    linguistic_file_path = linguistic_features_path + '/' + str(ID) + '_' + Gender + '_bert.csv'
    linguistic_feature = pd.read_csv(linguistic_file_path)
    linguistic_feature_total = pd.concat([linguistic_feature_total, linguistic_feature], axis=0, ignore_index=True)

combined_features = pd.DataFrame()
combined_features[acoustic_columns] = acoustic_features
combined_features[new_columns] = linguistic_feature_total
combined_file = combined_features_path + '/combined_features.csv'
combined_features.to_csv(combined_file, index=False)

print("finish")
