import pandas as pd 
from aikit.transformers import NumericalEncoder
import random
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np


# Reading company data 
data = pd.read_csv("dataf.csv")

# Cleaning the data 
df = data.drop(["domain","year founded","size range","locality","country","linkedin url","current employee estimate","total employee estimate"], axis = 1)


search_values = ["financial services","banking","investment banking","information technology","gaming","publishing","insurance","oil & energy","pharmaceutical","automotive","consumer goods"]

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df1 = df[df.industry.str.contains('|'.join(search_values))]

# Classification of companies into groups
transactionclients = ["automotive","oil & energy","consumer goods","automotive","pharmaceuticals","information technology and services"]
financingclients = ["publishing","gaming"]
assetclients = ["insurance"]
bankingclients = ["financial services","banking","investment banking"]


#key process classification 
def keyprocess(classification, keyprocess):
    
    for industry in classification:
        criteria = 'industry == "%s"' % industry
        df1.loc[df.eval(criteria), 'key_process'] = keyprocess
    
# randomisation of financial key processes   
def keyprocessB(classification, keyprocess):
    for i, val in enumerate(df1['industry']):
        keyp = random.randrange(len(keyprocess) - 1) 
        for bc in bankingclients:
            if(val == bc ):
                keyp = random.randrange(len(keyprocess) - 1) 
                df1.at[i, 'key_process'] = keyprocess[keyp]
                

#key solution generator 
def keysolution(kp, productsolution):
    criteria = 'key_process == "%s"' % kp
    print(criteria)
    df1.loc[df1.eval(criteria), 'product_solution'] = productsolution
    
#df1.to_csv("/Users/samueladdotey/Local work/Personal Projects 2020/Applications/SG/SG Projects /data/cleandata.csv")
    
    

def load_sgdata():
    """ load the SG dataset """

    sg_df = pd.read_csv("cleandata.csv")
    #sg_df = pd.read_csv("/Users/samueladdotey/Local work/Personal Projects 2020/Applications/SG/SG Projects /data/cleandata.csv")
    sg_df = sg_df.loc[:, ~sg_df.columns.str.contains('^Unnamed')]
    

    # Shuffle DF and compute train/test split
    #df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    idx = int(len(sg_df) * (1 - 0.3))
    df_train = sg_df.loc[:idx]
    df_test = sg_df.loc[idx:]

    # Filter Y columns test and train data
    #Y train & test datasets
    y_train = df_train["product_solution"].to_frame()
    y_test = df_test["product_solution"].to_frame()
    
    #X drop uneeded columns from X test & train datasets 
    del df_train["name"]
    del df_train["product_solution"]
    del df_test["product_solution"]
    del df_test["name"]
    
    print(type(df_train))

    # setting parameters for encoding x-values 
    Xencoder = NumericalEncoder(columns_to_use=['industry', 'key_process'],
                 desired_output_type='DataFrame', drop_unused_columns=False,
                 drop_used_columns=True, encoding_type='num',
                 max_cum_proba=0.95, max_modalities_number=100,
                 max_na_percentage=0.05, min_modalities_number=20,
                 min_nb_observations=10, regex_match=False)   
    
    # encoding x values (industry, key process)
    Xencoded_train = Xencoder.fit_transform(df_train)
    Xencoded_test = Xencoder.fit_transform(df_test)

    # setting parameters for encoding y-values
    Yencoder = NumericalEncoder(columns_to_use=['product_solution'],
                 desired_output_type='DataFrame', drop_unused_columns=False,
                 drop_used_columns=True, encoding_type='num',
                 max_cum_proba=0.95, max_modalities_number=100,
                 max_na_percentage=0.05, min_modalities_number=20,
                 min_nb_observations=10, regex_match=False)  

    # encoding y values (product solutions)
    Yencoded_train = Yencoder.fit_transform(y_train)
    Yencoded_test = Yencoder.fit_transform(y_test)
    
   
    infos = {}
    return Xencoded_train, Yencoded_train, Xencoded_test, Yencoded_test
    

    
"""
def set_configs(launcher):
    modify that function to change launcher configuration
    launcher.job_config.score_base_line = 0.75
    launcher.job_config.allow_approx_cv = True
    return launcher
    
def loader():
    dfX, y, *_ = load_sgdata()
    return dfX, y

    
    """
    
    
if __name__ == "__main__":    

    # calling functions for key processes
    keyprocessB(bankingclients, ["Investments","Flow & Hedging","Trading","Securities Services","Asset Management"])
    keyprocess(transactionclients,"Transactions")
    keyprocess(financingclients, "Financing")
    keyprocess(assetclients, "Asset Management")
    df1 = df1.dropna() # drop na fields
        
    # calling functions for key solutions
    keysolution("Transactions","SG Global Cash")
    keysolution("Financing","SG Coop")
    keysolution("Asset Management","SG Lyxor AP")
    keysolution("Investments","SG Analytics")
    keysolution("Flow & Hedging","SG MyHedge")
    keysolution("Trading","SG Clearing Solutions")
    keysolution("Securities Services","SG D-View")


    print("***** loading data *****")

    # saving the training and tests set returned from my data load function
    x_train, y_train, x_test, y_test = load_sgdata()
    
    print("***** about to train model *****")
    
    # run logistic regerssion
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train.values.ravel())
    
    print("***** about to predict *****")
    
    # predict on the test set and print accuracy 
    y_pred = logreg.predict(x_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

    
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(logreg, open(filename, 'wb'))
 
    

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    
    myvals = np.array([7,2]).reshape(1, -1)
    result = loaded_model.predict(myvals)
    
    # dictionary to interpret solutions for output
    solutions = {0: "SG G-Cash", 1:"SG MyHedge", 2:"SG Lyxor AP", 3:"SG Analyst", 4:"SG Coop", 5:"SG Clearing Solutions", 6:"SG D-View"}
    
    
    print(solutions[(result[0])])
    
    
    
    
    
    