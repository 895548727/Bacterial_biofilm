#!/usr/bin/env python
#_*_coding:utf-8_*_

import re, sys, os
from collections import Counter
# 5,6,7,8,9,10
def EAAC(fastas, window=5, **kw):
	AA = 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#','label']
	for aa in AA:
		header.append(aa)
	encodings.append(header)

	for i in range(len(fastas)):
		if i < len(fastas) / 2:
			name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 1
		else:
			name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 0
		code = [name,label]
		dict_bb = {i:0 for i in AA}
		for j in range(len(sequence)):
			if j < len(sequence) and j + window <= len(sequence):
				count = Counter(re.sub('-', '', sequence[j:j+window]))
				for key in count:
					dict_bb[key] += count[key]
		for aa in AA:
			code.append(dict_bb[aa])
		encodings.append(code)
	return encodings


