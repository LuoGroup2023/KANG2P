#!/usr/bin/env python
import pickle
import json
import csv
import numpy as np
import os
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from feature_engine.selection import MRMR
#set randome seed
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) #fix hash seed
	np.random.seed(seed)
	

seed = 1521024
seed_torch(seed)

method=sys.argv[1]
feature_file=sys.argv[2]
ratio=float(sys.argv[3]) #for downsampling train sampels
prefix=sys.argv[4]

data = pd.read_csv(feature_file, sep='\t', header=0)  # Adjust sep and header based on your file
print(data)
gene_id = data.columns
print(gene_id[0:3])
dataset_X = data.values  # Convert to NumPy array

dataset_X=np.array(dataset_X) # extract selected features

_,dataset_Y=pickle.load(open('../A3GALT2.pkl','rb'))

dataset_X.shape
dataset_Y.shape



# train dataset
train_idx = [int(line.strip()) for line in open("../train_val.balanced.idx", 'r')]
# train_idx = [int(line.strip()) for line in open("../train_val.unique.idx", 'r')]
print(len(train_idx))

# test dataset
te_idx = [int(line.strip()) for line in open("../test.idx", 'r')]

#subsampling

#random.seed(123)
random.shuffle(train_idx)
random.shuffle(te_idx)

train_idx = random.sample(train_idx,int(len(train_idx)*ratio))
random.shuffle(train_idx)

##
x_train=[]
y_train=[]
x_test=[]
y_test=[]

x_train=dataset_X[train_idx]
x_test=dataset_X[te_idx]

y_train=dataset_Y[train_idx]
y_test =dataset_Y[te_idx]

print('x_train:{}'.format(x_train.shape))
print('x_test:{}'.format(x_test.shape))
y_train.shape
y_test.shape

y_train = np.argmax(y_train, axis=1)
y_test_num = np.argmax(y_test, axis=1)

if method=='lr':
    #### LR
    print("\nrunning LR... MRMR \n")
    sel = MRMR(method="FCQ", regression=False)
    sel.fit(x_train, y_train)

    selected_mask = sel.get_support()
    selected_indices = np.where(selected_mask)[0]
    selected_features = gene_id[selected_indices]
    print(len(selected_features))
    output_file = f"{prefix}_mrmr_selected_features.txt"
    with open(output_file, 'w') as f:
        for idx, feat_name in zip(selected_indices, selected_features):
            f.write(f"{idx}\t{feat_name}\n")

    print(f"✅ Selected features saved to {output_file}")

    print(len(sel.relevance_))
    x_train_selected = sel.transform(x_train)
    print(x_train_selected.shape)
    x_test_selected = sel.transform(x_test)
    # Step 5: Train logistic regression model on selected features
    logisticRegr = LogisticRegression(random_state=1991, solver='saga')
    logisticRegr.fit(x_train_selected, y_train)

    # Predict and evaluate
    y_pred = logisticRegr.predict(x_test_selected)

    tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()
    acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
    ps = round(tp*1./(tp+fp),3)
    rc = round(tp*1./(tp+fn),3)
    f1=round(2*(ps*rc)/(ps+rc),3)

    print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
    print("Pression: ", ps)
    print("Recall:", rc)
    print("F1: ",2*(ps*rc)/(ps+rc))

    print("TP={}, TN={}, FP={}, FN={}".format(tp,tn,fp,fn))
    with open(prefix+'.out.csv','a') as fw:
        fw.write(','.join([prefix]+list(map(str,[ps,rc,f1,acc])))+'\n')


