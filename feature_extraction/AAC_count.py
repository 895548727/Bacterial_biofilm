#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
from collections import Counter

# def read_protein_sequences(file):
#     # if os.path.exists(file) == False:
#     #     print('Error: file %s does not exist.' % file)
#     #     sys.exit(1)
#     with open(file) as f:
#         records = f.read()
#     # if re.search('>', records) == None:
#     #     print('Error: the input file %s seems not in FASTA format!' % file)
#     #     sys.exit(1)
#     records = records.split('>')[1:]
#     fasta_sequences = []
#     for fasta in records:
#         array = fasta.split('\n')
#         header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
#         header_array = header.split('|')
#         name = header_array[0]
#         label = header_array[1] if len(header_array) >= 1 else '0'
#         label_train = header_array[2] if len(header_array) >= 2 else 'training'
#         fasta_sequences.append([name, sequence, label, label_train])
#     return fasta_sequences

def AAC(fastas):
    # AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = ['#', 'label']
    for i in AA:
        header.append(i)
    encodings.append(header)
    # fastas = read_protein_sequences(fastas)

    for i in range(len(fastas)):
        if i < len(fastas) / 2:
            name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 1
        else:
            name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 0
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]
        code = [name, label]
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings

# with open('protein.tsv') as f:
#     records = f.readlines()
# print(len(records[0].split()))
# f.close()
# def read_tsv(file='protein.tsv'):
#     encodings = []
#     labels = []
#     with open(file) as f:
#         records = f.readlines()
#
#     ##
#     feature = 1
#     header = ['#']
#     for i in range(1, len(records[0].split())):
#         header.append('f.%d' % feature)
#         feature = feature + 1
#     encodings.append(header)
#
#     ##
#     sample = 1
#     for line in records:
#         array = line.strip().split('\t') if line.strip() != '' else None
#         encodings.append(['s.%d' % sample] + array[1:])
#         labels.append(int(array[0]))
#         sample = sample + 1
#     return np.array(encodings), labels