import data_preprocess as dp
import numpy as np
import pandas as pd
import scoring as score
import normal_run as nr
import ranking_subset_run as rsr
import sfs_run as sfs_r
import boost_bag_run as bbr
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
import uni_multiStats as stats

data1 = pd.read_csv('preprocessed_anemia.csv',skipinitialspace=True, header = 0)
data1 = data1.drop(columns = ['Unnamed: 0', 'CASE ID'], axis=1)

target = 'ChildAnemia'
# selected = ['All_Population_Count_2015','Annual_Precipitation_2015','Day_Land_Surface_Temp_2015','Diurnal_Temperature_Range_2015','Enhanced_Vegetation_Index_2015','Global_Human_Footprint','Gross_Cell_Production','Growing_Season_Length','Irrigation','ITN_Coverage_2015','Land_Surface_Temperature_2015','Malaria_Prevalence_2015','Mean_Temperature_2015','Nightlights_Composite','Proximity_to_Water']
# nominal = data.drop(columns = selected,axis =1 )
# nominal = nominal.drop(columns =['Cluster','CASE ID'], axis =1)

# data1 = dp.modify_data(data,selected,nominal,target)
# data1.to_csv('preprocessed_anemia.csv')
print(data1.columns)



# stats.gen_stats(data1, target ane 
# n_features = data1.shape[1]-1

# # # #################### RUNNING WITHOUT BOOSTING AND BAGGING for all ranking feature selections and CFS###############
n_seed = 5
splits =10
runs = stats.runSKFold(n_seed,splits,data=data1,target=target)
score.score(rsr.normal_run( n_seed, splits, ['infogain_10'], ['naive_bayes'], runs, n_features),n_seed,splits)
# score.score(sfs_r.subset_run(n_seed, splits,['elasticnet'],['accuracy'],runs,n_features),n_seed,splits)
# sfs_r.subset_features(n_seed,splits, ['elasticnet'],['accuracy'],runs, n_features)
# score.score(bbr.boostbag_run(n_seed,splits,['infogain_20'],['elasticnet'],runs,'boost',n_features), n_seed,splits)
# # subset_features(n_seed, splits, estimators,metrics, runs, n_features):





# data = pd.read_csv('child_uncleaned.csv',skipinitialspace=True, header = 0)
# data2 =pd.read_csv('GIS.csv', skipinitialspace = True, header=0)
# selected = ['DHSCLUST','All_Population_Count_2015','Annual_Precipitation_2015','Day_Land_Surface_Temp_2015','Diurnal_Temperature_Range_2015','Enhanced_Vegetation_Index_2015','Global_Human_Footprint','Gross_Cell_Production','Growing_Season_Length','Irrigation','ITN_Coverage_2015','Land_Surface_Temperature_2015','Malaria_Prevalence_2015','Mean_Temperature_2015','Nightlights_Composite','Proximity_to_Water']
# GIS = data2[selected]

# df = pd.DataFrame(columns=selected)



# samples  = data.shape[0]
# clusters = GIS['DHSCLUST']
# num_GIS_variables = len(selected)
# zero_row = pd.DataFrame(0, index =[0],columns = selected )

# for i in range(samples):
#     clusnum = data.iloc[i,:]['Cluster']
    
#     if clusnum in clusters:
#         row = GIS.loc[GIS['DHSCLUST']==clusnum]
        
#         df = df.append(row, ignore_index=True)
#     else:
#         df = df.append(zero_row, ignore_index=True)
        
        
# GIS_added = pd.concat([data,df], axis =1)
# GIS_added = GIS_added.drop(['DHSCLUST'], axis=1)
# GIS_added.to_csv('gis_added.csv')

        
    
   
    
    
    

