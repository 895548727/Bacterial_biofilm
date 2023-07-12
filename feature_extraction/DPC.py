#!/usr/bin/env python
#_*_coding:utf-8_*_

import re

def DPC(fastas, **kw):
	AA = 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	header = ['#','label'] + diPeptides
	encodings.append(header)

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in range(len(fastas)):
		if i < len(fastas) / 2:
			name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 1
		else:
			name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 0
		code = [name,label]
		tmpCode = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]
		code = code + tmpCode
		encodings.append(code)
	return encodings