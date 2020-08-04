import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 



###### reading from csv file that contains 3 columns, feature selection, risk factor and probability ######
df = pd.read_csv('/Users/vantran/Desktop/PLOTS/heatmapfeatures.csv')
result = df.pivot('Feature Selection','Risk Factor', 'Probability')
#fig,ax = 

# plt.subplots(figsize=(4,4))
###### reordering the rows and columns #######
# Features = ['KNN-F1','KNN-Ac','SVM-F1','SVM-Ac','RF-F1','RF-Ac','NB-F1','NB-Ac','LR-F1','LR-Ac','XGB-F1','XGB-Ac','IG-20','ReF-20','IG-10','ReF-10','CFS','MRMR','FCBF' ]
# Columns = [1,12,13,18, 24,7,2,3,4,5, 6,8,9,10,11,14,15,16,17,19,20,21,22,23,25,26,27]
# result = result.reindex(index = Features,columns = Columns)
# plt.title(title,fontsize = 30)
# df['Feature Selection']=pd.Categorical(df['Feature Selection'], categories=Features,ordered=True)
# plt.xlabel("Risk Factor", fontsize = 20)
# plt.ylabel("Feature Selection", fontsize = 20)
# # ttl = ax.title
# # ttl.set_position([0.5,1.05])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set


# # res = sns.heatmap(pd.crosstab(df['Feature Selection'],df['Feature']),cmap ='RdYlGn',annot = False,linewidths=0.30,ax=ax)

# res = sns.heatmap(result,cmap ='RdYlGn',annot = False,linewidths=0.30,ax=ax )#annot_kws={"size": 10},fmt ='.2%'
# res.set_xticklabels(res.get_xmajorticklabels(),fontsize =15)
# res.set_yticklabels(res.get_ymajorticklabels(),fontsize =15,rotation =45)
# colorbar  = res.collections[0].colorbar
# colorbar.ax.locator_params(nbins =3)
# plt.savefig('feature.png')
sns.clustermap(result, method='weighted',linewidths=0.30,figsize=(10,7),cbar_pos=(0, .2, .03, .4),cmap="vlag")
# cm.set_axis_labels('Risk Factor','Feature Selection')

 
# hm = cm.ax_heatmap.get_position()
# plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=15)
# plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), fontsize=15)
# plt.setp(cm.ax_heatmap.xaxis., fontsize=20)
# cm.ax_heatmap.set_position([hm.x0, hm.y0, hm.width, hm.height])
# col = cm.ax_col_dendrogram.get_position()
# cm.ax_col_dendrogram.set_position([col.x0, col.y0, col.width, col.height])





plt.savefig('cluster_weightedmethod.png')

plt.show()

#RdYlGn
