import re
def readNonFasta(file):
	with open(file) as f:
		records = f.read()
	myFasta = []
	records = records.split('\n')
	for fasta in records:
		sequence = fasta
		myFasta.append(sequence)
	return myFasta
# 1) 读入数据
def readFasta(file):
	with open(file) as f:
		records = f.read()
	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name,sequence])
	return myFasta
def writeFasta(data,out):
	with open(out,"w") as f:
		f.writelines(data)

def window_count(input,window=200):
	dataset = []
	for i in input:
		if len(i) <= window:
			dataset.append(i)
		else:
			for j in range(len(i)):
				if j + window < len(i):
					dataset.append(i[j:j+window])
				else:
					break
	return dataset
# if __name__ == '__main__':
# degs_data = readNonFasta("./data/deg_result.txt")
# print(negative_data)
# data = window_count(posi_data)

# writeFasta(data,"./data/bbsdb/negative.fasta")
# # a = window_count(positive_data,10000)
# # # print(a)
# degs_data = [">" + str(i) + "\n" + degs_data[i] + "\n" for i in range(len(degs_data))]
# print(len(a))
# b = window_count(negative_data,200)
# window_negative_data = [">" + str(i) + "\n"+ data[i] + "\n" for i in range(len(data))]

# writeFasta(degs_data,"./data/degs_data.fasta")