'''

Authors: Xiangyu Zhao
Date: Sunday, 5th December, 2021
Title: Explore PAREZ dataset and make distribution histograms

'''

from os.path import abspath
import pandas as pd
import matplotlib.pyplot as plt
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

# Load and preprocess labels
def explore_labels(labels_path, output_path):
    data = pd.read_csv(labels_path)
    print(data)
    print(data.describe())
    print(data.info())
    # Get data for pre and post conflict labels
    data = data.loc[:, ['afemos11', 'afemos12', 'afemos13', 'afemos14', 'amemos11', 'amemos12', 'amemos13', 'amemos14',
                        'afemos21', 'afemos22', 'afemos23', 'afemos24', 'amemos21', 'amemos22', 'amemos23', 'amemos24']]
    # Drop rows with any NA values
    data.dropna(inplace=True)
    # make distribution histogram for selected columns
    labels, count = np.unique(data['amemos24'], return_counts=True)
    labels = [1, 2, 3, 4, 5, 6]
    #count = [60, 171, 108, 9, 1, 0]
    plt.bar(labels, count, align='center')
    plt.gca().set_xticks(labels)
    plt.show()
    print('finish')


labels_path = abspath("./dataset/labels/pre_post_conflict_emotions_labels.csv")
output_path = abspath("./dataset/labels")
explore_labels(labels_path, output_path)
