#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division

import argparse
import os
import codecs
import random
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from collections import Counter



def main(opt):
    with codecs.open(opt.testdatasrc, 'r', encoding="utf-8") as f1:
        lines = f1.readlines()
        chooseIdsrc = lines[opt.id].strip()
    with codecs.open(opt.testdatatgt, 'r', encoding="utf-8") as f2:
        lines = f2.readlines()
        chooseIdtgt = lines[opt.id].strip()
    f1.close()
    f2.close()
    f = codecs.open(opt.outputsrc, 'w+', encoding="utf-8")
    f.write(''.join(chooseIdsrc))
    f.close()
    f = codecs.open(opt.outputtgt, 'w+', encoding="utf-8")
    f.write(''.join(chooseIdtgt))
    f.close()
    # print(chooseIdRaw)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='oneCaseData.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-id', type=int,
                        required=True, default=0,
                       help='which pair of the testdataset you want to train with RL,exp-id')
    parser.add_argument('-testdatasrc', type=str,
                        required=True,
                       help='load all testdata src path')
    parser.add_argument('-testdatatgt', type=str,
                        required=True,
                       help='load all testdata tgt path')
    parser.add_argument('-outputsrc', type=str,
                        required=True,
                       help='one case output src path.')
    parser.add_argument('-outputtgt', type=str,
                        required=True,
                       help='one case output tgt path.')
    opt = parser.parse_args()
    main(opt)


