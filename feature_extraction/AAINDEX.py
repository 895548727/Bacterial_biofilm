#!/usr/bin/env python
# _*_coding:utf-8_*_

# import argparse
# import sys, os, re, platform

# pPath = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(pPath)
# father_path = os.path.abspath(
#     os.path.dirname(pPath) + os.path.sep + ".") + r'\pubscripts' if platform.system() == 'Windows' else os.path.abspath(
#     os.path.dirname(pPath) + os.path.sep + ".") + r'/pubscripts'
# sys.path.append(father_path)
# import check_sequences
# import read_fasta_sequences
# import save_file
import re

def AAINDEX(fastas, props=None, **kw):
    # if check_sequences.check_fasta_with_equal_length(fastas) == False:
    #     print('Error: for "AAINDEX" encoding, the input fasta sequences should be with equal length. \n\n')
    #     return 0

    AA = 'ARNDCQEGHILKMFPSTWYV'

    fileAAindex = "./feature_extraction/AAindex.txt"
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    #  use the user inputed properties
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex
    
    encodings = []
    header = ['#','label']
    for pos in range(1, len(fastas[0][1]) + 1):
        for idName in AAindexName:
            header.append('SeqPos.' + str(pos) + '.' + idName)
    encodings.append(header)

    for i in range(len(fastas)):
        if i < len(fastas)/2:
            name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 1
        else:
            name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 0
        code = [name, label]
        for aa in sequence:
            if aa == '-':
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encodings.append(code)
    # savetsv.savetsv(encodings, "aaindex.tsv")
    return encodings

