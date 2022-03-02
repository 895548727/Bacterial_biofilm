################################
########## 计算CNN指标 ##########
###############################

from datautil import sign_ML
TP,FP,TN,FN=int(input("TP:")),int(input("FP:")),int(input("TN:")),int(input("FN:"))
sign_ML(TP,FP,TN,FN)



################################
####### 处理fasta格式文件 ########
###############################

# def transfer_fasta():
#     import os
#     os.chdir("C:/Users/MI/Desktop")
#     result = []
#     with open("lisite.txt","r") as t:
#         b = t.readlines()
#         for line in b:
#             if ">VIMSS" in line:
#                 result.append(line)
#     t.close()
#     res = open("gene_name.txt","w")
#     res.writelines(result)
#     res.close()
#     gene_id = []
#     with open("gene_name.txt","r") as m:
#         c = m.readlines()
#         for line in c:
#             middile_value = line.split("(")[0].split(" ",2)
#             if len(middile_value)==3:
#                 gene_id.append(middile_value[0]+"$"+middile_value[1]+"$"+middile_value[2] + "\n")
#             else:
#                 gene_id.append(middile_value[0] + "$" + middile_value[1] + "$" + "\n")
#     m.close()
#     f1 = open("gene_name.txt","w")
#     # print(gene_id)
#     f1.writelines(gene_id)
#     f1.close()
# transfer_fasta()