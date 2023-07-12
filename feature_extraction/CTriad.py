#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
# k 0,1,2,3,4,5
def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[
                    sequence[i + 2 * g + 2]]
                if myDict.get(fea,-1) == -1:
                    myDict[fea] = 1
                else:
                    myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res

def CTriad(fastas, gap = 0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    header = ['#','label']
    for f in features:
        header.append(f)
    encodings.append(header)

    for i in range(len(fastas)):
        if i < len(fastas) / 2:
            name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 1
        else:
            name, sequence, label = fastas[i][0], re.sub('-', '', fastas[i][1]), 0
        code = [name, label]
        code = code + CalculateKSCTriad(sequence, 0, features, AADict)
        encodings.append(code)
    return encodings