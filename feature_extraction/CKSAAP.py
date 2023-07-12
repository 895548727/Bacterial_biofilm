#!/usr/bin/env python
# _*_coding:utf-8_*_

# import sys, os
# import argparse
# import read_fasta_sequences
# import check_sequences
import re
# gap = 2,3,4,5,6
def CKSAAP(fastas, gap=2, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    # if check_sequences.get_min_sequence_length(fastas) < gap + 2:
    #     print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
    #     return 0

    # AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = ['#','label']
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for i in range(len(fastas)):
        if i < len(fastas) / 2:
            name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 1
        else:
            name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 0
        code = [name, label]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 1
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                    if myDict.get(sequence[index1]+sequence[index2],-1) == -1:
                        myDict[sequence[index1]+sequence[index2]] = 1
                    else:
                        myDict[sequence[index1] + sequence[index2]] += 1
                    # myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings
