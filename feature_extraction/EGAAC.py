#!/usr/bin/env python
#_*_coding:utf-8_*_

import re, sys, os
# 5,6,7,8,9,10
from collections import Counter
def EGAAC(fastas, window=5, **kw):
	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	groupKey = group.keys()

	encodings = []
	header = ['#','label']

	for key in groupKey:
		header.append(key)
	encodings.append(header)

	for i in range(len(fastas)):
		if i < len(fastas) / 2:
			name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 1
		else:
			name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 0
		code = [name,label]
		myDict = {}
		for j in range(len(sequence)):
			if j + window <= len(sequence):
				count = Counter(sequence[j:j + window])
				for key in groupKey:
					for aa in group[key]:
						myDict[key] = myDict.get(key, 0) + count[aa]
		for key in groupKey:
				code.append(myDict[key] / len(sequence))
		encodings.append(code)
	return encodings