#!/usr/bin/env python
#_*_coding:utf-8_*_

import re

def TPC(fastas, **kw):
	AA = 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
	header = ['#','label'] + triPeptides
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
		tmpCode = [0] * 8000
		for j in range(len(sequence) - 3 + 1):
			tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j+1]]*20 + AADict[sequence[j+2]]] = tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j+1]]*20 + AADict[sequence[j+2]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]
		code = code + tmpCode
		encodings.append(code)
	return encodings
