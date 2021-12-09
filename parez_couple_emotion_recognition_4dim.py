'''

Authors: Xiangyu Zhao
Date: Tuesday, 7th Dec, 2021
Title: Preprocess features and labels for machine learning training
Data: PAREZ: Contains lab conversation of Swiss couples (n=368)

'''

# Import libraries
from os.path import abspath
import pandas as pd
import numpy as np
'''
Explanation of headings
Female, before conflict interaction
afemos11 – female, pre assessment,  Good mood / bad mood
afemos12 – female, pre assessment,  Stress / relaxed      
afemos13 – female, pre assessment,  Peaceful / angry    
afemos14 – female, pre assessment,  Happy / sad
 
Male, before conflict interaction
amemos11 – male, pre assessment,  Good mood / bad mood
amemos12 – male, pre assessment,  Stress / relaxed       
amemos13 – male, pre assessment,  Peaceful / angry      
amemos14 – male, pre assessment,  Happy / sad
 
Female, after conflict interaction
afemos21 – female, post assessment,  Good mood / bad mood
afemos22 – female, post assessment,  Stress / relaxed    
afemos23 – female, post assessment,  Peaceful / angry  
afemos24 – female, post assessment,  Happy / sad
 
Male, after conflict interaction
amemos21 – male, post assessment,  Good mood / bad mood
amemos22 – male, post assessment,  Stress / relaxed     
amemos23 – male, post assessment,  Peaceful / angry    
amemos24 – male, post assessment,  Happy / sad

Range: 1 - 6, higher values are more negative

'''

MALE = 0
FEMALE = 1
BOTH = 3
COUPLE_ID_INDEX = 0
GENDER_INDEX = 1
ACOUSTIC = 1
LINGUISTIC = 2

NONE = -1

# Load and prepocess labels
def preprocess_labels(labels_path, output_path):
    data = pd.read_csv(labels_path)
    print(data)
    print(data.describe())
    print(data.info())

    # Get data for post conflict ratings
    data_f = data.loc[:, ['CoupleID', 'afemos21', 'afemos22', 'afemos23', 'afemos24']]
    data_m = data.loc[:, ['CoupleID', 'amemos21', 'amemos22', 'amemos23', 'amemos24']]
    data_f['Gender'] = 'f'
    data_m['Gender'] = 'm'
    
    # Drop rows with any NA values
    data_f.dropna(inplace=True)
    data_m.dropna(inplace=True)

    '''
    # Average them 
    data_f.drop('afemos22', axis=1, inplace=True)
    data_m.drop('amemos22', axis=1, inplace=True)

    data_f.drop('afemos23', axis=1, inplace=True)
    data_m.drop('amemos23', axis=1, inplace=True)

    data_f['avg_f'] = data_f.loc[:, 'afemos21': 'afemos24'].mean(axis=1)
    data_m['avg_m'] = data_m.loc[:, 'amemos21': 'amemos24'].mean(axis=1)

    # Binarize label
    # 0 (negative) - >=3
    # 1 (positive) - <=3
    binarize = lambda x: 0 if x >= 3.5 else 1
    data_f['valence_f'] = data_f['avg_f'].map(binarize)
    data_m['valence_m'] = data_m['avg_m'].map(binarize)

    print()
    print('Female  valence count: ', data_f['valence_f'].value_counts())
    print('Male  valence count: ', data_m['valence_m'].value_counts())
    '''

    # binarize labels value
    binarize = lambda x: 0 if x < 3.5 else 1
    data_f['good_bad'] = data_f['afemos21'].map(binarize)
    data_f['stress_relaxed'] = data_f['afemos22'].map(binarize)
    data_f['peaceful_angry'] = data_f['afemos23'].map(binarize)
    data_f['happy_sad'] = data_f['afemos24'].map(binarize)
    data_m['good_bad'] = data_m['amemos21'].map(binarize)
    data_m['stress_relaxed'] = data_m['amemos22'].map(binarize)
    data_m['peaceful_angry'] = data_m['amemos23'].map(binarize)
    data_m['happy_sad'] = data_m['amemos24'].map(binarize)
    # Combine male and female labels so each is on a separate row
    new_columns = ['ID', 'Gender', 'good_bad', 'stress_relaxed', 'peaceful_angry', 'happy_sad']
    new_data_f = data_f.loc[:, ['CoupleID', 'Gender', 'good_bad', 'stress_relaxed', 'peaceful_angry', 'happy_sad']]
    new_data_m = data_m.loc[:, ['CoupleID', 'Gender', 'good_bad', 'stress_relaxed', 'peaceful_angry', 'happy_sad']]
    new_data_f.columns = new_columns
    new_data_m.columns = new_columns
    data_combined = pd.concat([new_data_f, new_data_m], ignore_index=True)
    print(data_combined.describe())

    output_file = output_path + '/binary_labels.csv'
    data_combined.to_csv(output_file, index=False)

# Combine features and labels
def combine_features_labels(labels_path, features_path, output_path):

    labels_data = pd.read_csv(labels_path)
    features_data = pd.read_csv(features_path)
    # normalization each feature
    df = features_data.loc[:, 'F0semitoneFrom27.5Hz_sma3nz_amean-0':'Col767']
    normalized_df = (df - df.mean()) / df.std()
    features_data.loc[:, 'F0semitoneFrom27.5Hz_sma3nz_amean-0':'Col767'] = normalized_df

    data = pd.merge(labels_data, features_data, on=['ID', 'Gender'])
    data_acous_ling = data.drop(['Transcript'], axis=1)
    data_acous = data_acous_ling.drop(data_acous_ling.loc[:, 'Col0':'Col767'], axis=1)
    data_ling = data_acous_ling.drop(data_acous_ling.loc[:, 'F0semitoneFrom27.5Hz_sma3nz_amean-0':'equivalentSoundLevel_dBp-1'], axis=1)

    output_file_acous = output_path + '/normalize_german_bert_features_binary_labels/acoustic.csv'
    output_file_ling = output_path + '/normalize_german_bert_features_binary_labels/linguistic.csv'
    output_file_acous_ling = output_path + '/normalize_german_bert_features_binary_labels/acous_ling.csv'

    data_ling.to_csv(output_file_ling, index=False)
    data_acous.to_csv(output_file_acous, index=False)
    data_acous_ling.to_csv(output_file_acous_ling, index=False)

# Read file containing feature vectors and return features and labels
# Input: path to features and gender
# Returns: X, y
def load_features(path, gender=BOTH, modality='acous_ling'):
    print()
    print('----------------------------')
    print('Loading features...')
    print('----------------------------')

    data = pd.read_csv(path)  # default to using both genders

    if gender == MALE:
        data = data[data['Gender'] == 'm']

    elif gender == FEMALE:
        data = data[data['Gender'] == 'f']

    X = data.drop(['good_bad', 'stress_relaxed', 'peaceful_angry', 'happy_sad', 'Gender', 'Name'], axis=1).to_numpy()

    if modality == 'linguistic':
        X = X.loc[:, 'ID', 'F0semitoneFrom27.5Hz_sma3nz_amean-0':'equivalentSoundLevel_dBp-1'].to_numpy()

    elif modality == 'acoustic':
        X = data.loc[:, 'Col0: Col767'].to_numpy()



    y_good_bad = data.loc[:, 'good_bad'].to_numpy()

    print('Train: ' + str(len(X)) + '  samples containing ' + str(X.shape[1]-1) + ' features')
    return X, y_good_bad

# Read file containing feature vectors and return features of both partners combined but labels of the specified gender
# Input: path to features and gender, dropped modality is the modality of the attaching partner to drop
# eg: acous and ling features of partner a and acous features of partner b
# Returns: X, y
def load_features_couple(path, gender=BOTH, dropped_modality=NONE):
    print()
    print('----------------------------')
    print('Loading features...')
    print('----------------------------')

    data = pd.read_csv(path)  # default to using both genders
    data_m = data[data['Gender'] == 'm']
    data_f = data[data['Gender'] == 'f']

    if gender == MALE:
        # Drop columns of female
        data_f = data_f.drop(['Gender', 'good_bad', 'stress_relax', 'peaceful_angry', 'happy_sad', 'Name'], axis=1)

        if dropped_modality == LINGUISTIC:
            data_f = data_f.drop(data_f.loc[:, 'Col0':'Col767'], axis=1)
        elif dropped_modality == ACOUSTIC: 
            data_f = data_f.drop(data_f.loc[:, 'F0semitoneFrom27.5Hz_sma3nz_amean-0': 'equivalentSoundLevel_dBp-1'], axis=1)

        # Combine female with male
        data = pd.merge(data_m, data_f, on=['ID'])

    elif gender == FEMALE:
        # Drop columns of male
        data_m = data_m.drop(['Gender', 'Valence', 'Name'], axis=1)

        if dropped_modality == LINGUISTIC:
            data_m = data_m.drop(data_m.loc[:, 'Col0':'Col767'], axis=1)
        elif dropped_modality == ACOUSTIC: 
            data_m = data_m.drop(data_m.loc[:, 'F0semitoneFrom27.5Hz_sma3nz_amean-0':'equivalentSoundLevel_dBp-1'], axis=1)

        # Combine with female with male 
        data = pd.merge(data_f, data_m, on=['ID'])

    y_val = data.loc[:, 'Valence'].to_numpy()
    X = data.to_numpy()

    print('Datasize: ' + str(len(X)) + '  samples containing ' + str(X.shape[1]-4) + ' features')
    return X, y_val 

# Main
labels_path = abspath("./dataset/labels/pre_post_conflict_emotions_labels.csv")
output_path = abspath("./dataset/labels")
binary_labels_path = abspath("./dataset/labels/binary_labels.csv")
features_path = abspath("./dataset/features/combined_german_bert_features/combined_features.csv")

#preprocess_labels(labels_path, output_path)
#combine_features_labels(binary_labels_path, features_path, output_path)
#load_features(output_path + '/features_labels/acoustic.csv', MALE)
#load_features_couple(output_path + '/features_labels/acous_ling.csv', FEMALE, ACOUSTIC)

print('finish')



