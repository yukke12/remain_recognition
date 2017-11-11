# -*- coding: utf-8 -*- 
import os
import sys
import numpy as np
import pandas as pd


def main():
    train_txt = 'train.txt'
    test_txt = 'test.txt'
    train_list = []
    test_list = []
    with open(train_txt, 'r') as train_data:
        for line in train_data:
            line = line.rstrip()
            vals = line.rsplit(' ')
            train_list.append(vals[0])

    with open(test_txt, 'r') as test_data:
        for line in test_data:
            line = line.rstrip()
            vals = line.rsplit(' ')
            test_list.append(vals[0])

    train_set = set(train_list)
    test_set = set(test_list)
    matched_list = list(train_set & test_set)
    print(matched_list)

if __name__ == '__main__':
    main()