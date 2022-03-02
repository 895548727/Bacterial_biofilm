# from Bio import SeqIO
# def read_original_data(path):
#     data=[]
#     for seq_record in SeqIO.parse(r'./data/user/ecoli/'+path+'.fasta',"fasta"):
#         data.append(seq_record.description)
#     print(len(data))
#     return data
#
# def write_dataset(data,path):
#     trainfile=open(r"data/user/ecoli/"+path,"w")
#     for i in range(int(len(data))):
#         trainfile.write(str(data[i]))
#         trainfile.write('\n')
#     trainfile.close()
# data = read_original_data('ecoli')
# write_dataset(data,'ecoliid')

# def getfinaldata():
#     data = []
#     with open(r'data/user/vibrio_cholerae','r') as f:
#         line=f.readline()
#         while line:
#             data.append(line)
#             line=f.readline()
#     return data
# b= []
# with open(r'data/user/vibrio_cholerae','r') as f:
#     line = f.readline()
#     while line:
#         b.append(len(line))
#         line=f.readline()
# print(b)
# f.close()
# print(b.sort())