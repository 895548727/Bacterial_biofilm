#!/usr/bin/env python
#_*_ coding:utf-8 _*_
from feature_extraction import *
# from normalization import ZScore
import re
from other_script import *
import numpy as np
import pandas as pd
import math
import joblib
# from scipy import interp
from itertools import cycle
from data_process import readFasta
#####  1.特征提取
# 1) 读入数据
positive_data, negative_data = readFasta("./window/count/window_positive_data.fasta"), readFasta("./window/count/window_negative_data.fasta")

all_data = positive_data + negative_data
# print(all_data)
# 二、特征提取
AAC_fea = AAC.AAC(all_data)

AAC_count_fea = AAC_count.AAC(all_data)

AAindex_fea = AAINDEX.AAINDEX(all_data)
CKSAAP_2_fea = CKSAAP.CKSAAP(all_data)
CKSAAP_3_fea = CKSAAP.CKSAAP(all_data,3)
CKSAAP_4_fea = CKSAAP.CKSAAP(all_data,4)
CKSAAP_5_fea = CKSAAP.CKSAAP(all_data,5)
CKSAAP_6_fea = CKSAAP.CKSAAP(all_data,6)
CTDC_fea = CTDC.CTDC(all_data)
CTDD_fea = CTDD.CTDD(all_data)

CTDT_fea = CTDT.CTDT(all_data)

CTriad_0_fea = CTriad.CTriad(all_data)
CTriad_1_fea = CTriad.CTriad(all_data,1)
CTriad_2_fea = CTriad.CTriad(all_data,2)
CTriad_3_fea = CTriad.CTriad(all_data,3)
CTriad_4_fea = CTriad.CTriad(all_data,4)
CTriad_5_fea = CTriad.CTriad(all_data,5)

PAAC_fea = PAAC.PAAC(all_data)

DDE_fea = DDE.DDE(all_data)
DPC_fea = DPC.DPC(all_data)
EAAC_5_fea = EAAC.EAAC(all_data)
EAAC_6_fea = EAAC.EAAC(all_data,6)
EAAC_7_fea = EAAC.EAAC(all_data,7)
EAAC_8_fea = EAAC.EAAC(all_data,8)
EAAC_9_fea = EAAC.EAAC(all_data,9)
EAAC_10_fea = EAAC.EAAC(all_data,10)
p_EAAC_5_fea = EAAC_pencent.EAAC(all_data)
p_EAAC_6_fea = EAAC_pencent.EAAC(all_data,6)
p_EAAC_7_fea = EAAC_pencent.EAAC(all_data,7)
p_EAAC_8_fea = EAAC_pencent.EAAC(all_data,8)
p_EAAC_9_fea = EAAC_pencent.EAAC(all_data,9)
p_EAAC_10_fea = EAAC_pencent.EAAC(all_data,10)
GAAC_fea = GAAC.GAAC(all_data)
EGAAC_5_fea = EGAAC.EGAAC(all_data)
EGAAC_6_fea = EGAAC.EGAAC(all_data,6)
EGAAC_7_fea = EGAAC.EGAAC(all_data,7)
EGAAC_8_fea = EGAAC.EGAAC(all_data,8)
EGAAC_9_fea = EGAAC.EGAAC(all_data,9)
EGAAC_10_fea = EGAAC.EGAAC(all_data,10)
GDPC_fea = GDPC.GDPC(all_data)
Geary_fea = Geary.Geary(all_data)
GTPC_fea = GTPC.GTPC(all_data)
TPC_fea = TPC.TPC(all_data)
aac_dpc = [AAC_fea[i] + DPC_fea[i][2:] for i in range(len(AAC_fea))]
aac_tpc = [AAC_fea[i] + TPC_fea[i][2:] for i in range(len(AAC_fea))]
ctd = [CTDC_fea[i] + CTDD_fea[i][2:] + CTDT_fea[i][2:] for i in range(len(CTDC_fea))]
#
#
AAindex_fea = [i[:5000] for i in AAindex_fea]

# # # 转化为pandas格式
# #
AAindex_pd = pd.DataFrame(AAindex_fea, columns=AAindex_fea[0])
AAC_count_pd = pd.DataFrame(AAC_count_fea, columns=AAC_count_fea[0])
AAC_pd = pd.DataFrame(AAC_fea, columns=AAC_fea[0])
CTDC_pd = pd.DataFrame(CTDC_fea, columns=CTDC_fea[0])
CTDT_pd = pd.DataFrame(CTDT_fea, columns=CTDT_fea[0])
CKSAAP_2_pd = pd.DataFrame(CKSAAP_2_fea, columns=CKSAAP_2_fea[0])
CKSAAP_3_pd = pd.DataFrame(CKSAAP_3_fea, columns=CKSAAP_3_fea[0])
CKSAAP_4_pd = pd.DataFrame(CKSAAP_4_fea, columns=CKSAAP_4_fea[0])
CKSAAP_5_pd = pd.DataFrame(CKSAAP_5_fea, columns=CKSAAP_5_fea[0])
CKSAAP_6_pd = pd.DataFrame(CKSAAP_6_fea, columns=CKSAAP_6_fea[0])
GAAC_pd = pd.DataFrame(GAAC_fea, columns=GAAC_fea[0])
CTriad_0_pd = pd.DataFrame(CTriad_0_fea, columns=CTriad_0_fea[0])
CTriad_1_pd = pd.DataFrame(CTriad_1_fea, columns=CTriad_1_fea[0])
CTriad_2_pd = pd.DataFrame(CTriad_2_fea, columns=CTriad_2_fea[0])
CTriad_3_pd = pd.DataFrame(CTriad_3_fea, columns=CTriad_3_fea[0])
CTriad_4_pd = pd.DataFrame(CTriad_4_fea, columns=CTriad_4_fea[0])
CTriad_5_pd = pd.DataFrame(CTriad_5_fea, columns=CTriad_5_fea[0])
PAAC_pd = pd.DataFrame(PAAC_fea, columns=PAAC_fea[0])
DDE_pd = pd.DataFrame(DDE_fea, columns=DDE_fea[0])
DPC_pd = pd.DataFrame(DPC_fea, columns=DPC_fea[0])
EAAC_5_pd = pd.DataFrame(EAAC_5_fea, columns=EAAC_5_fea[0])
EAAC_6_pd = pd.DataFrame(EAAC_6_fea, columns=EAAC_6_fea[0])
EAAC_7_pd = pd.DataFrame(EAAC_7_fea, columns=EAAC_7_fea[0])
EAAC_8_pd = pd.DataFrame(EAAC_8_fea, columns=EAAC_8_fea[0])
EAAC_9_pd = pd.DataFrame(EAAC_9_fea, columns=EAAC_9_fea[0])
EAAC_10_pd = pd.DataFrame(EAAC_10_fea, columns=EAAC_10_fea[0])
p_EAAC_5_pd = pd.DataFrame(p_EAAC_5_fea, columns=p_EAAC_5_fea[0])
p_EAAC_6_pd = pd.DataFrame(p_EAAC_6_fea, columns=p_EAAC_6_fea[0])
p_EAAC_7_pd = pd.DataFrame(p_EAAC_7_fea, columns=p_EAAC_7_fea[0])
p_EAAC_8_pd = pd.DataFrame(p_EAAC_8_fea, columns=p_EAAC_8_fea[0])
p_EAAC_9_pd = pd.DataFrame(p_EAAC_9_fea, columns=p_EAAC_9_fea[0])
p_EAAC_10_pd = pd.DataFrame(p_EAAC_10_fea, columns=p_EAAC_10_fea[0])
EGAAC_5_pd = pd.DataFrame(EGAAC_5_fea, columns=EGAAC_5_fea[0])
EGAAC_6_pd = pd.DataFrame(EGAAC_6_fea, columns=EGAAC_6_fea[0])
EGAAC_7_pd = pd.DataFrame(EGAAC_7_fea, columns=EGAAC_7_fea[0])
EGAAC_8_pd = pd.DataFrame(EGAAC_8_fea, columns=EGAAC_8_fea[0])
EGAAC_9_pd = pd.DataFrame(EGAAC_9_fea, columns=EGAAC_9_fea[0])
EGAAC_10_pd = pd.DataFrame(EGAAC_10_fea, columns=EGAAC_10_fea[0])
GDPC_pd = pd.DataFrame(GDPC_fea, columns=GDPC_fea[0])
Geary_pd = pd.DataFrame(Geary_fea, columns=Geary_fea[0])
TPC_pd = pd.DataFrame(TPC_fea, columns=TPC_fea[0])
aac_dpc = pd.DataFrame(aac_dpc, columns=aac_dpc[0])
aac_tpc = pd.DataFrame(aac_tpc, columns=aac_tpc[0])
ctd = pd.DataFrame(ctd, columns=ctd[0])
CTDD_pd = pd.DataFrame(CTDD_fea, columns=CTDD_fea[0])
GTPC_pd = pd.DataFrame(GTPC_fea, columns=GTPC_fea[0])



# # # new_feature用于承接特征提取方法的一维数据

new_feature = AAC_pd.iloc[1:,:2]
print(new_feature)
# cv = StratifiedKFold(n_splits=10)
# classifier = svm.SVC(kernel="poly", probability=True, random_state=random_state)
# accuracy_score = []
# tprs = []
# aucs = []

# mean_fpr = np.linspace(0, 1, 100)
# fig, ax = plt.subplots(figsize=(10, 10))
# for fold, (train, test) in enumerate(skf.split(X, y)):
# 	classifier.fit(X[train], y[train])
# 	viz = RocCurveDisplay.from_estimator(classifier, X[test], y[test], name=f"ROC fold {fold}", alpha=0.3, lw=1,
# 										 ax=ax, )
# 	interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#
# 	accuracy_score.append(accuracy_score(y[test], clf.predict(X[test]))
#
# 	interp_tpr[0] = 0.0
# 	tprs.append(interp_tpr)
# 	aucs.append(viz.roc_auc)
# 	ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
#
# 	mean_tpr = np.mean(tprs, axis=0)
# 	mean_tpr[-1] = 1.0
# 	mean_auc = auc(mean_fpr, mean_tpr)
# 	std_auc = np.std(aucs)
# 	ax.plot(
# 		mean_fpr,
# 		mean_tpr,
# 		color="b",
# 		label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
# 		lw=2,
# 		alpha=0.8,
# 	)
#
# 	std_tpr = np.std(tprs, axis=0)
# 	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# 	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# 	ax.fill_between(
# 		mean_fpr,
# 		tprs_lower,
# 		tprs_upper,
# 		color="grey",
# 		alpha=0.2,
# 		label=r"$\pm$ 1 std. dev.",
# 	)
#
# 	ax.set(
# 		xlim=[-0.05, 1.05],
# 		ylim=[-0.05, 1.05],
# 		xlabel="False Positive Rate",
# 		ylabel="True Positive Rate",
# 		title=f"Mean ROC curve with variability\n(Positive label '{target_names[1]}')",
# 	)
# 	ax.axis("square")
# 	ax.legend(loc="lower right")
# 	plt.show()

# # # 训练学习模型
AAC_pd  = machine_learning.svm_self(AAC_pd,"AAC")
AAindex_pd  = machine_learning.svm_self(AAindex_pd,"AAindex")
CKSAAP_2_pd  = machine_learning.svm_self(CKSAAP_2_pd,"CKSAAP_2")
CKSAAP_3_pd, CKSAAP_4_pd, CKSAAP_5_pd,CKSAAP_6_pd = machine_learning.svm_self(CKSAAP_3_pd,"CKSAAP_3"), machine_learning.svm_self(CKSAAP_4_pd,'CKSAAP_4'), machine_learning.svm_self(CKSAAP_5_pd,'CKSAAP_5'), machine_learning.svm_self(CKSAAP_6_pd,'CKSAAP_6')
CTriad_0_pd, CTriad_1_pd, CTriad_2_pd = machine_learning.svm_self(CTriad_0_pd,"CTriad_0"), machine_learning.svm_self(CTriad_1_pd,'CTriad_1'), machine_learning.svm_self(CTriad_2_pd,'CTriad_2')
CTriad_3_pd, CTriad_4_pd, CTriad_5_pd = machine_learning.svm_self(CTriad_3_pd,"CTriad_3"), machine_learning.svm_self(CTriad_4_pd,'CTriad_4'), machine_learning.svm_self(CTriad_5_pd,'CTriad_5')
DDE_pd, DPC_pd = machine_learning.svm_self(DDE_pd,"DDE"), machine_learning.svm_self(DPC_pd,'DPC')
CTDC_pd, CTDD_pd =  machine_learning.svm_self(CTDC_pd,'CTDC'), machine_learning.svm_self(CTDD_pd,'CTDD')

CTDT_pd, AAC_count_pd, PAAC_pd = machine_learning.svm_self(CTDT_pd,'CTDT'), machine_learning.svm_self(AAC_count_pd,'AAC_count'), machine_learning.svm_self(PAAC_pd,'PAAC')
# CTriad_0_pd = machine_learning.ml(CTriad_0_pd,"CTriad")
aac_dpc, aac_tpc, ctd = machine_learning.svm_self(aac_dpc,'aac_dpc'), machine_learning.svm_self(aac_tpc,'aac_tpc'), machine_learning.svm_self(ctd,'ctd')
#
EAAC_7_pd,EAAC_5_pd, EAAC_6_pd = machine_learning.svm_self(EAAC_7_pd,"EAAC_7"), machine_learning.svm_self(EAAC_5_pd,'EAAC_5'), machine_learning.svm_self(EAAC_6_pd,'EAAC_6')
EAAC_8_pd, EAAC_9_pd, EAAC_10_pd = machine_learning.svm_self(EAAC_8_pd,"EAAC_8"), machine_learning.svm_self(EAAC_9_pd,'EAAC_9'), machine_learning.svm_self(EAAC_10_pd,'EAAC_10')

p_EAAC_7_pd,p_EAAC_5_pd, p_EAAC_6_pd = machine_learning.svm_self(p_EAAC_7_pd,"p_EAAC_7"), machine_learning.svm_self(p_EAAC_5_pd,'p_EAAC_5'), machine_learning.svm_self(p_EAAC_6_pd,'p_EAAC_6')
p_EAAC_8_pd, p_EAAC_9_pd, p_EAAC_10_pd = machine_learning.svm_self(p_EAAC_8_pd,"EAAC"), machine_learning.svm_self(p_EAAC_9_pd,'EAAC'), machine_learning.svm_self(p_EAAC_10_pd,'p_EAAC_10')
EGAAC_7_pd,EGAAC_5_pd, EGAAC_6_pd = machine_learning.svm_self(EGAAC_7_pd,"EGAAC_7"), machine_learning.svm_self(EGAAC_5_pd,'EGAAC_5'), machine_learning.svm_self(EGAAC_6_pd,'EGAAC')
EGAAC_8_pd, EGAAC_9_pd, EGAAC_10_pd = machine_learning.svm_self(EGAAC_8_pd,"EGAAC_8"), machine_learning.svm_self(EGAAC_9_pd,'EGAAC_9'), machine_learning.svm_self(EGAAC_10_pd,'EGAAC_10')

GDPC_pd, Geary_pd, GTPC_pd, TPC_pd = machine_learning.svm_self(GDPC_pd,"GDPC"), machine_learning.svm_self(Geary_pd,'Geary'), machine_learning.svm_self(GTPC_pd,'GTPC'),machine_learning.svm_self(TPC_pd,'TPC')
GAAC_pd = machine_learning.svm_self(GAAC_pd,"GDPC")

# AAC_pd  = machine_learning.svm_self(AAC_pd,"AAC")
# CKSAAP_4_pd = machine_learning.svm_self(CKSAAP_4_pd,'CKSAAP_4')
# TPC_pd = machine_learning.svm_self(TPC_pd,'TPC')
# plot_roc.plot_roc_cv(CKSAAP_3_pd[3], CKSAAP_3_pd[4], "CKSAAP_3_pd", "./")

# index_feature = ['AAC_pd','AAindex_pd','CKSAAP_2_pd','CKSAAP_3_pd','CKSAAP_4_pd','CKSAAP_5_pd','CKSAAP_6_pd','CTriad_0_pd','CTriad_1_pd','CTriad_2_pd','CTriad_3_pd','CTriad_4_pd','CTriad_5_pd',
#  				 'DDE_pd','DPC_pd','CTDC_pd','CTDD_pd','CTDT_pd','AAC_count_pd','PAAC_pd','EAAC_7_pd','EAAC_5_pd','EAAC_6_pd','EAAC_8_pd','EAAC_9_pd','EAAC_10_pd','GDPC_pd','Geary_pd','GTPC_pd','TPC_pd','GAAC_pd','p_EAAC_7_pd',
# 				 'p_EAAC_5_pd','p_EAAC_6_pd','p_EAAC_8_pd','p_EAAC_9_pd','p_EAAC_10_pd','aac_dpc','aac_tpc','ctd','EGAAC_5_pd','EGAAC_6_pd','EGAAC_7_pd','EGAAC_8_pd','EGAAC_9_pd','EGAAC_10_pd']
#
# for i in index_feature:
# 	plot_roc.plot_roc_cv(eval(i)[3], eval(i)[4], i, "./")


# index_feature = ['AAC_pd']
# def evaluated(index):
# 	# new_feature[index] = eval(index)[0]
# 	print(index + "评估指标：")
# 	print(calculate_prediction.calculate_metrics(eval(index)[3].tolist(),eval(index)[4]))
#


# # # 对数据集进行分类，返回0,1

# new_feature['AAC_pd'] = AAC_pd[0]
# new_feature['AAindex_pd'] = AAindex_pd[0]
# new_feature['CKSAAP_2_pd'] = CKSAAP_2_pd[0]
# new_feature['CKSAAP_3_pd'] = CKSAAP_3_pd[0]
# new_feature['CKSAAP_4_pd'] = CKSAAP_4_pd[0]
# new_feature['CKSAAP_5_pd'] = CKSAAP_5_pd[0]
# new_feature['CKSAAP_6_pd'] = CKSAAP_6_pd[0]
# new_feature['CTriad_0_pd'] = CTriad_0_pd[0]
# new_feature['CTriad_1_pd'] = CTriad_1_pd[0]
# new_feature['CTriad_2_pd'] = CTriad_2_pd[0]
# new_feature['CTriad_3_pd'] = CTriad_3_pd[0]
# new_feature['CTriad_4_pd'] = CTriad_4_pd[0]
# new_feature['CTriad_5_pd'] = CTriad_5_pd[0]
# new_feature['DDE_pd'] = DDE_pd[0]
# new_feature['DPC_pd'] = DPC_pd[0]
# new_feature['CTDC_pd'] = CTDC_pd[0]
# new_feature['CTDD_pd'] = CTDD_pd[0]
# new_feature['CTDT_pd'] = CTDT_pd[0]
# new_feature['AAC_count_pd'] = AAC_count_pd[0]
# new_feature['PAAC_pd'] = PAAC_pd[0]
# new_feature['aac_dpc'] = aac_dpc[0]
# new_feature['aac_tpc'] = aac_tpc[0]
# new_feature['ctd'] = ctd[0]
# new_feature['EAAC_7_pd'] = EAAC_7_pd[0]
# new_feature['EAAC_5_pd'] = EAAC_5_pd[0]
# new_feature['EAAC_6_pd'] = EAAC_6_pd[0]
# new_feature['EAAC_8_pd'] = EAAC_8_pd[0]
# new_feature['EAAC_9_pd'] = EAAC_9_pd[0]
# new_feature['EAAC_10_pd'] = EAAC_10_pd[0]
# new_feature['p_EAAC_7_pd'] = p_EAAC_7_pd[0]
# new_feature['p_EAAC_5_pd'] = p_EAAC_5_pd[0]
# new_feature['p_EAAC_6_pd'] = p_EAAC_6_pd[0]
# new_feature['p_EAAC_8_pd'] = p_EAAC_8_pd[0]
# new_feature['p_EAAC_9_pd'] = p_EAAC_9_pd[0]
# new_feature['p_EAAC_10_pd'] = p_EAAC_10_pd[0]
# new_feature['EGAAC_7_pd'] = EGAAC_7_pd[0]
# new_feature['EGAAC_5_pd'] = EGAAC_5_pd[0]
# new_feature['EGAAC_6_pd'] = EGAAC_6_pd[0]
# new_feature['EGAAC_8_pd'] = EGAAC_8_pd[0]
# new_feature['EGAAC_9_pd'] = EGAAC_9_pd[0]
# new_feature['EGAAC_10_pd'] = EGAAC_10_pd[0]
# new_feature['GDPC_pd'] = GDPC_pd[0]
# new_feature['Geary_pd'] = Geary_pd[0]
# new_feature['GTPC_pd'] = GTPC_pd[0]
# new_feature['TPC_pd'] = TPC_pd[0]
# new_feature['GAAC_pd'] = GAAC_pd[0]
# print("AAC评估指标：")
# print(calculate_prediction.calculate_metrics(AAC_pd[3].tolist(), AAC_pd[4]))
# # plot_roc_cv(AAC_pd[0], AAC_pd[5], "pCT", "./")
# new_feature['new_feature'] = AAindex_pd[0]
# print("AAindex评估指标：")
# print(calculate_prediction.calculate_metrics(AAindex_pd[3].tolist(), AAindex_pd[4]))
# # plot_roc_cv(AAC_pd[3], AAC_pd[4], "AAindex", "./")
# new_feature['CTriad_5_pd'] = CTriad_5_pd[0]
# new_feature['GTPC_pd'] = GTPC_pd[0]
# new_feature['CTDD_pd'] = CTDD_pd[0]
# new_feature['CKSAAP_pd'] = CKSAAP_pd[0]
# print("CKSAAP评估指标：")
# print(calculate_prediction.calculate_metrics(CKSAAP_pd[3].tolist(), CKSAAP_pd[4]))
# # plot_roc_cv(CKSAAP_pd[0], CKSAAP_pd[5], "nCT", "./")
# new_feature['CTDC_pd'] = CTDC_pd[0]
# print("CTDC_pd评估指标：")
# print(calculate_metrics(CTDC_pd[3].tolist(), CTDC_pd[4]))
#
# new_feature['CTDD_pd'] = CTDD_pd[0]
# print("CTDD_pd评估指标：")
# print(calculate_prediction.calculate_metrics(CTDD_pd[3].tolist(), CTDD_pd[4]))
#
# new_feature['CTDT_pd'] = CTDT_pd[0]
# print("CTDT_pd评估指标：")
# print(calculate_prediction.calculate_metrics(CTDT_pd[3].tolist(), CTDT_pd[4]))
#
# new_feature['CTriad_pd'] = CTriad_pd[0]
# print("CTriad_pd评估指标：")
# print(calculate_prediction.calculate_metrics(CTriad_pd[3].tolist(), CTriad_pd[4]))
#
# new_feature['PAAC_pd'] = PAAC_pd[0]
# print("PAAC_pd评估指标：")
# print(calculate_prediction.calculate_metrics(PAAC_pd[3].tolist(), PAAC_pd[4]))
# # new_feature['PSSM_pd'] = PSSM_pd
# new_feature['CKSAAP_4_pd'] = CKSAAP_4_pd[0]
# new_feature['AAC_pd'] = AAC_pd[0]
new_feature['TPC_pd'] = TPC_pd[0]
new_feature['aac_dpc'] = TPC_pd[0]
# copy_feature = new_feature
index_feature = ['AAC_pd','AAindex_pd','CKSAAP_2_pd','CKSAAP_3_pd','CKSAAP_5_pd','CKSAAP_6_pd','CTriad_0_pd','CTriad_1_pd','CTriad_2_pd','CTriad_3_pd','CTriad_4_pd','CTriad_5_pd','CKSAAP_4_pd',
 				 'DDE_pd','DPC_pd','CTDC_pd','CTDD_pd','CTDT_pd','AAC_count_pd','PAAC_pd','EAAC_7_pd','EAAC_5_pd','EAAC_6_pd','EAAC_8_pd','EAAC_9_pd','EAAC_10_pd','GDPC_pd','Geary_pd','GTPC_pd','GAAC_pd','p_EAAC_7_pd',
				 'p_EAAC_5_pd','p_EAAC_6_pd','p_EAAC_8_pd','p_EAAC_9_pd','p_EAAC_10_pd','aac_tpc','ctd','EGAAC_5_pd','EGAAC_6_pd','EGAAC_7_pd','EGAAC_8_pd','EGAAC_9_pd','EGAAC_10_pd']

for i in index_feature:
    new_feature[i] = eval(i)[0]
    three_pd = machine_learning.svm_self(new_feature, 'three')
    plot_roc.plot_roc_cv(three_pd[3], three_pd[4], "Combined feature", "./")
    # new_feature = copy_feature
    del new_feature[i]
    print(new_feature)
# #
# three_pd = machine_learning.svm_self(new_feature,'three')
# print(importance)

# plot_roc.plot_roc_cv(three_pd[3], three_pd[4], "Combined feature", "./")
# print(importance)
# print(calculate_metrics(other[1].tolist(), other[2]))
# plot_roc_cv(AAC_pd[3], AAC_pd[4], "test", "./")
# plot_roc_cv(clf, other[3], "NiCT", "./")
# AAindex_array = ml(AAindex_pd)
# AAC_pd = ml(AAC_pd)
# AAC_pd = ml(CKSAAP_pd)
# CTDC_pd = ml(CTDC_pd)
# CTDD_pd = ml(CTDD_pd)
# CTDT_pd = ml(CTDT_pd)
# CTriad_pd = ml(CTriad_pd)
# PAAC_pd = ml(PAAC_pd)
# PSSM_pd = ml(PSSM_pd)
#
# new_feature = AAC_pd.iloc[:,:2]
# print(new_feature)
# importance = clf.feature_importances_;
# importance = importance.tolist()
# print(len(importance))
# c = importance[:]
# del c[c.index(max(c))]
# del c[c.index(max(c))]
# del c[c.index(max(c))]
# del c[c.index(max(c))]
# del c[c.index(max(c))]
# importance_dict = {}
# for i in range(len(importance)):
# 	if i >= max(c):
# 		importance_dict[i] = importance[i]
# print(importance_dict)
#####  4.模型构建
