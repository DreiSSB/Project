import csv
import glob
import json
import random
import logging
import os
from enum import Enum
from typing import List, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tqdm
import numpy as np

import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

def evaluate_standard(preds, labels):

    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds)
    r = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    return acc, p, r, f1

def pairwise_accuracy(guids, preds, labels):

    acc = 0.0  # The accuracy to return.
    
    ########################################################
    counter=0
    for i in range(int(len(guids)/2)):
        p = np.array([preds[i*2], preds[i*2+1]])
        l = np.array([labels[i*2], labels[i*2+1]])
        if (p==l).all():
            counter+=1
    
    acc = counter/(len(guids)/2)
    # End of TODO
    ########################################################
     
    return acc


if __name__ == "__main__":

    # Unit-testing the pairwise accuracy function.
    guids = [0, 0, 1, 1, 2, 2, 3, 3]
    preds = np.asarray([0, 0, 1, 0, 0, 1, 1, 1])
    labels = np.asarray([1, 0,1, 0, 0, 1, 1, 1])
    acc = pairwise_accuracy(guids, preds, labels)
    print(evaluate_standard(preds, labels))
    if acc == 0.75:
        print("Your `pairwise_accuracy` function is correct!")
    else:
        raise NotImplementedError("Your `pairwise_accuracy` function is INCORRECT!")
