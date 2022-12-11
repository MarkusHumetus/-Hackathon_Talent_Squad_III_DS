#import libraries

import numpy as np
import pandas as pd
from pycaret.classification import * 

#loading datasets
acad_Yield_train = pd.read_csv("https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Talent+Squad+League/3rd_batch/data/train.csv", index_col='Unnamed: 0')
acad_Yield_test = pd.read_csv("https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Talent+Squad+League/3rd_batch/data/test.csv", index_col='Unnamed: 0')

#Splitting initial dataframe according to binary caracteristic (gender, lunch, preparation exam) in several df wrapped in a dictionary (dict_df)
df = acad_Yield_train
dict_df = {}
label_dfs, label_ratio =[], []
for gender in ['male','female']:
    dict_df[gender]={}
    for lunch in ['standard', 'free/reduced']:
        dict_df[gender][lunch]={}        
        for test_preparation_course in ['none', 'completed']:
            dict_df[gender][lunch][test_preparation_course] = df[(df.gender==gender)& (df.lunch==lunch) & (df['test preparation course']==test_preparation_course)][['parental level of education', 'math score', 'reading score', 'writing score' ]]
         
#2 dataframes were to small and were merged:
df_concatenated =pd.concat([dict_df['male']['free/reduced']['completed'], dict_df['female']['free/reduced']['completed']])

#Charge Models from Pycaret (see Jupyter Notebook for how methods were developed)

setup(dict_df['male']['standard']['none'][['parental level of education', 'writing score' ]],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True    
                    )
best_model_male_std_none_1 = load_model ('best_model_male_std_none_1')

setup(dict_df['female']['standard']['none'][['parental level of education','writing score', 'math score']],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
best_model_female_std_none_2 = load_model ('best_model_female_std_none_2')

setup(dict_df['male']['standard']['completed'][['parental level of education','writing score','math score']],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
best_model_male_std_compltd_2 = load_model ('best_model_male_std_compltd_2')

setup(dict_df['female']['standard']['completed'][['parental level of education','writing score','math score']],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )   
best_model_female_std_compltd_2 = load_model ('best_model_female_std_compltd_2')

setup(dict_df['male']['free/reduced']['none'][['parental level of education', 'writing score' ]],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
best_model_male_fr_none_1 = load_model ('best_model_male_fr_none_1')

setup(dict_df['female']['free/reduced']['none'][['parental level of education', 'writing score' ]],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
best_model_female_fr_none_1 = load_model ('best_model_female_fr_none_1')

setup(df_concatenated[['parental level of education','writing score','math score']],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )  
best_model_male_fr_compltd_2 = load_model ('best_model_male_fr_compltd_2')


#Predictions

prediction  = []
for i in  range(len (acad_Yield_test)):

    if acad_Yield_test.iloc[i,:]['lunch'] == 'standard':
        if acad_Yield_test.iloc[i,:]['test preparation course'] == 'none':
            if acad_Yield_test.iloc[i,:]['gender'] == 'male':
                setup(dict_df['male']['standard']['none'][['parental level of education', 'writing score' ]],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
                prediction.append ( predict_model(best_model_male_std_none_1, acad_Yield_test.iloc[i:i+1][['writing score']]).Label)
            else:
                setup(dict_df['female']['standard']['none'][['parental level of education','writing score', 'math score']],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
                prediction.append ( predict_model(best_model_female_std_none_2, acad_Yield_test.iloc[i:i+1][['writing score','math score']]).Label)
        else:
            if acad_Yield_test.iloc[i,:]['gender'] == 'male':
                setup(dict_df['male']['standard']['completed'][['parental level of education','writing score','math score']],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
                prediction.append ( predict_model(best_model_male_std_compltd_2, acad_Yield_test.iloc[i:i+1][['writing score','math score']]).Label)
            else:
                setup(dict_df['female']['standard']['completed'][['parental level of education','writing score','math score']],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )                
                prediction.append ( predict_model(best_model_female_std_compltd_2, acad_Yield_test.iloc[i:i+1][['writing score','math score']]).Label)
    else:
        if acad_Yield_test.iloc[i,:]['test preparation course'] == 'none':
            if acad_Yield_test.iloc[i,:]['gender'] == 'male':
                setup(dict_df['male']['free/reduced']['none'][['parental level of education', 'writing score' ]],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
                prediction.append ( predict_model(best_model_male_fr_none_1, acad_Yield_test.iloc[i:i+1][['writing score']]).Label)
            else:
                setup(dict_df['female']['free/reduced']['none'][['parental level of education', 'writing score' ]],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )
                prediction.append ( predict_model(best_model_female_fr_none_1, acad_Yield_test.iloc[i:i+1][['writing score']]).Label)
        else:
            setup(df_concatenated[['parental level of education','writing score','math score']],
                    target='parental level of education',
                    session_id=42, 
                    normalize = True,
                    fold_strategy = 'stratifiedkfold', 
                    verbose =False ,
                    silent = True        
                )            
            prediction.append ( predict_model(best_model_male_fr_compltd_2, acad_Yield_test.iloc[i:i+1][['writing score','math score']]).Label)
            

print ('success')
