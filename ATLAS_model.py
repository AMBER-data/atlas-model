# -*- coding: utf-8 -*-
# =============================================================================
"""
This script allows to replicate the results presented in the manuscript 
'More than one million barriers fragment Europe’s rivers' by Belletti et al. 
It takes as input *.csv file with barrier drivers information for different 
aggregation of sub-basins and predict barrier density and barrier numbers 
at pan-european-scale.

Script authors: Simone Bizzi and Barbara Belletti
Contact: simone.bizzi@unipd.it

License: Creative Commons by Attribution.
This script can be used for commercial and non-commercial purposes provided 
the author is acknowledged.
If used in student work, it should be added as a text file in the appendice 
of the work.

"""
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import KFold, cross_validate
import os
from sklearn import preprocessing
######################################################

# Working Directory
fpath='C:\\...\\'

# Input File names 
Path_Ecr_cal = ['C_ZHYD_Ecrins12_Basin_Ecrins_calDB_202005.csv', 'C_ZHYD_Ecrins12_Basin_600_calDB_202005.csv', 'C_ZHYD_Ecrins12_Basin_1200_calDB_202005.csv','C_ZHYD_Ecrins12_Basin_2500_calDB_202005.csv','C_ZHYD_Ecrins12_Basin_3000_calDB_202005.csv']
Path_Ecr_all = ['C_ZHYD_Ecrins12_Basin_Ecrins_202005.csv', 'C_ZHYD_Ecrins12_Basin_600_202005.csv', 'C_ZHYD_Ecrins12_Basin_1200_202005.csv','C_ZHYD_Ecrins12_Basin_2500_202005.csv','C_ZHYD_Ecrins12_Basin_3000_202005.csv']

#Output Table definition
DF_output=pd.DataFrame(index=[0,1,2,3,4], columns=['Kmeans','RMSE','MAE','Barrier_tot','Model_run'])
DF_output['Model_run']=['ECrins','600','1200','2500','3000']

# Cycle of modelling for different basin aggregations
for i in range(0,len(Path_Ecr_cal)):
    
    # load model dataset
    full_name_in = os.path.join(fpath, Path_Ecr_cal[i])
    ATLAS_Ecr_cal = pd.read_csv(full_name_in, header=0)
    
    full_name_in = os.path.join(fpath, Path_Ecr_all[i])
    ATLAS_Ecr_all = pd.read_csv(full_name_in, header=0)
    
    # Select the variable to be used as inputs and output
    pred_val='Nkm2_obs'# Density per kmq
    ATLAS_Ecr_cal_sel=ATLAS_Ecr_cal[['elev','slop','popd','clc1','clc2','clc3','clc4','clc5','LenD','denr','roadD','RivDen','area_1',pred_val]]
    ATLAS_Ecr_all_sel=ATLAS_Ecr_all[['elev','slop','popd','clc1','clc2','clc3','clc4','clc5','LenD','denr','roadD','RivDen','area_1','ZHYD',pred_val]] 

    #Remove basins with Area=0
    ATLAS_Ecr_cal_sel=ATLAS_Ecr_cal_sel[ATLAS_Ecr_cal_sel['area_1']>0] 

    # Model 
    DF_input=ATLAS_Ecr_cal_sel
    X = np.array(DF_input.iloc[:,0:DF_input.shape[1]-3])# This selects out model training features
    X = preprocessing.scale(X)
    y = np.array(DF_input.iloc[:,DF_input.shape[1]-1]) # Sets the response variable for training
    
    #Instantiate the estimator 
    Estimator = RFR(n_estimators = 50, min_samples_split =10,random_state=41)

    #Cross validation with K-folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=41)
    results = cross_validate(Estimator, X, y, cv=kfold, scoring= ('explained_variance','neg_mean_squared_error','neg_mean_absolute_error')) #'neg_mean_absolute_error'; 'explained_variance'
    DF_output.iloc[i]['Kmeans']=np.mean(results['test_explained_variance'])
    DF_output.iloc[i]['RMSE']=np.mean(results['test_neg_mean_squared_error'])
    DF_output.iloc[i]['MAE']=np.mean(results['test_neg_mean_absolute_error'])
    
    ### Apply the model to the entire Europe (i.e., the entire ECrins database) 
    Estimator.fit(X,y) 
    DF_all_input=ATLAS_Ecr_all_sel
    EU_input = np.array(DF_all_input.iloc[:,0:DF_all_input.shape[1]-4])
    EU_input = preprocessing.scale(EU_input)
    EU_barrier = Estimator.predict(EU_input)*DF_all_input['area_1'] # Number of barriers 
    EU_barden = Estimator.predict(EU_input) # Number of barriers every squared kilometer
    
    DF_output.iloc[i]['Barrier_tot']=np.sum(EU_barrier)
            
# Print summary table of modelling results    
print(DF_output)


# To export the results
#EU_barden_export=pd.DataFrame({"Nkm2_pred": EU_barden, "ZHYD": DF_all_input['ZHYD']})
#EU_barden_export.to_csv('NomeFile.csv',index=False, float_format='%.3f')


   