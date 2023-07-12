#!/usr/bin/env python
#_*_coding:utf-8_*_

import sys, platform, os, re
import numpy as np

def Geary(fastas, props=['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102',
						 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
				nlag = 30, **kw):
	AA = 'ARNDCQEGHILKMFPSTWYV'
	fileAAidx = './data/AAidx.txt'
	with open(fileAAidx) as f:
		records = f.readlines()[1:]
	myDict = {}
	for i in records:
		array = i.rstrip().split('\t')
		myDict[array[0]] = array[1:]

	AAidx = []
	AAidxName = []
	for i in props:
		if i in myDict:
			AAidx.append(myDict[i])
			AAidxName.append(i)
	AAidx1 = np.array([float(j) for i in AAidx for j in i])
	AAidx = AAidx1.reshape((len(AAidx), 20))

	propMean = np.mean(AAidx, axis=1)
	propStd = np.std(AAidx, axis=1)

	for i in range(len(AAidx)):
		for j in range(len(AAidx[i])):
			AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

	index = {}
	for i in range(len(AA)):
		index[AA[i]] = i

	encodings = []
	header = ['#','label']
	for p in props:
		for n in range(1, nlag+1):
			header.append(p + '.lag' + str(n))
	encodings.append(header)

	for i in range(len(fastas)):
		if i < len(fastas) / 2:
			name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 1
		else:
			name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 0
		code = [name,label]
		N = len(sequence)
		for prop in range(len(props)):
			xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
			for n in range(1, nlag + 1):
				if len(sequence) > nlag:
					# if key is '-', then the value is 0
					rn = (N-1)/(2*(N-n)) * ((sum([(AAidx[prop][index.get(sequence[j], 0)] - AAidx[prop][index.get(sequence[j + n], 0)])**2 for j in range(len(sequence)-n)])) / (sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))])))
				else:
					rn = 'NA'
				code.append(rn)
		encodings.append(code)
	return encodings