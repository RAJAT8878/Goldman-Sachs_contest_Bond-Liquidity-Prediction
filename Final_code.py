
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import *
import xgboost as xgb
import itertools
# import matplotlib.pyplot as plt
# import seaborn


# In[2]:

trades=pd.read_csv('dataset.csv')
bonds=pd.read_csv('ML_Bond_metadata_corrected_dates.csv')


# In[3]:

trades1=pd.read_csv('dataset.csv')


# In[4]:

bonds.ratingAgency1EffectiveDate=pd.to_datetime(bonds.ratingAgency1EffectiveDate)
bonds.ratingAgency2EffectiveDate=pd.to_datetime(bonds.ratingAgency2EffectiveDate)

df_temp=bonds.issuer.value_counts().reset_index().rename(columns={'index':'issuer','issuer':'issuer_freq'})
bonds=pd.merge(bonds,df_temp,on='issuer',how='left')
bonds.drop(labels='issuer',inplace=1,axis=1)
del df_temp


# In[5]:

isin_freq=trades['isin'].value_counts().reset_index().rename(columns={'index':'isin','isin':'isin_counts'})
days_freq=trades.groupby(['isin']).date.nunique().reset_index().rename(columns={'date':'trade_freq_days'})


# In[6]:

trades['date']=pd.to_datetime(trades.date,format='%d%b%Y')

df_avg=trades.groupby(['isin','date','side']).volume.sum().reset_index()
df1=df_avg[df_avg.side=='B'].rename(columns={'volume':'buyvolume'})
df2=df_avg[df_avg.side=='S'].rename(columns={'volume':'sellvolume'})

df1.drop(labels='side',axis=1,inplace=1)
df2.drop(labels='side',axis=1,inplace=1)

trades=pd.merge(df1,df2,on=['isin','date'],how='outer')
trades.loc[trades.buyvolume.isnull(),'buyvolume']=0
trades.loc[trades.sellvolume.isnull(),'sellvolume']=0


# In[7]:

def volume_three_days(date,num,):
    three_days=date-pd.Timedelta(days=2)
    df=trades[(trades.date<=date) & (trades.date>=three_days)]
    df=df.groupby('isin').agg({'buyvolume':'sum','sellvolume':'sum'}).reset_index()

    
    df_2=pd.DataFrame()
    df_2['isin']=list(set(unique_bonds)-set(df['isin'].unique()))
    df_2['buyvolume']=0
    df_2['sellvolume']=0
    df=df.append(df_2)
    
    df['start_date']=three_days
    df['end_date']=date
    df['num']=num
    return df

def volume_last_three_days(num):
    df=train[train.num==num+1][['isin','buyvolume','sellvolume']]
    df['num']=num
    df.rename(columns={'buyvolume':'buy_last_three_days','sellvolume':'sell_last_three_days'},inplace=1)
    return df
    
def vol_till_now(date):
    three_days=date-pd.Timedelta(days=2)
    df=trades[(trades.date<=date) & (trades.date>=three_days)]
    df=df.groupby('isin').agg({'buyvolume':'sum','sellvolume':'sum'}).reset_index()

    
    df_2=pd.DataFrame()
    df_2['isin']=list(set(unique_bonds)-set(df['isin'].unique()))
    df_2['buyvolume']=0
    df_2['sellvolume']=0
    df=df.append(df_2)
    
    df['start_date']=three_days
    df['end_date']=date
    df['num']=num
    return df
    


# In[8]:

last_date=pd.to_datetime('2016-06-09')
train=pd.DataFrame()
unique_bonds=trades['isin'].unique()
for idx,_ in enumerate(range(1,86,3)):
    train=train.append(volume_three_days(last_date,idx))
    last_date-=pd.Timedelta(days=3)


# In[9]:

last_three=pd.DataFrame()
for idx in range (0,28):
    last_three=last_three.append(volume_last_three_days(idx))


# In[10]:

train=pd.merge(train,last_three,on=['isin','num'],how='left')
train=pd.merge(train,bonds,on='isin',how='left')


# In[11]:

train['spike_1']=0
train.loc[(train.ratingAgency1EffectiveDate>=train.start_date),'spike_1']=1

train['spike_2']=0
train.loc[(train.ratingAgency2EffectiveDate>=train.start_date),'spike_1']=1

# train=train[(train.spike_1==0) & (train.spike_2==0)]


# In[12]:

train=pd.merge(train,isin_freq,on='isin',how='left')
train=pd.merge(train,days_freq,on='isin',how='left')
train['month']=train.start_date.dt.month


# In[13]:

# mean_volume=train.groupby('isin').agg({'buyvolume':'sum','sellvolume':'sum'
#                                       }).reset_index().rename(columns={'buyvolume':'avg_buyvolume',
#                                                                        'sellvolume':'avg_sellvolume'})

# mean_volume['difference']=mean_volume.avg_buyvolume-mean_volume.avg_sellvolume

# train=pd.merge(train,mean_volume[['isin','difference']],on='isin',how='left')
# test=pd.merge(test,mean_volume[['isin','difference']],on='isin',how='left')


# In[14]:

test=train[train.num==0]
test.drop(labels=['buy_last_three_days','sell_last_three_days'],axis=1,inplace=1)
test.rename(columns={'buyvolume':'buy_last_three_days','sellvolume':'sell_last_three_days'},inplace=1)


# In[15]:

features=list(set(test.columns)-set(['isin','start_date','end_date','num','ratingAgency1EffectiveDate','ratingAgency2EffectiveDate',
                                    'issueDate','maturity','industrySubgroup',]))


# In[16]:

for col in features:
    if train[col].dtype=='O':
        train[col].replace(np.nan,'AAAA',inplace=1)
        test[col].replace(np.nan,'AAAA',inplace=1)
        lr=preprocessing.LabelEncoder()
        lr.fit(list(train[col])+list(test[col]))
        train[col]=lr.transform(train[col])
        test[col]=lr.transform(test[col])
    else:
        train[col].replace(np.nan,-1,inplace=1)
        test[col].replace(np.nan,-1,inplace=1)
        


# In[92]:

params={"objective":'reg:linear',     
    "learning_rate":0.1,
#     "min_child_weight": 5,
    "subsample":0.6,
    "colsample_bytree": 0.7,
        'eval_metric':'rmse',
    "max_depth":6,
#     'eta':.7,
    'silent':1,
    'nthread':3,
        
       }


# In[122]:

def model(train,test,features,params,isrf,islr,target):    
    cv=[]
    truth=[]
    cv_scores=[]
    lst=[]
    skf=cross_validation.KFold(len(train),n_folds=4,random_state=0)
    for idx1,idx2 in skf:
        
        x_train,x_cv=train[features].iloc[idx1],train[features].iloc[idx2]
        y_train,y_cv=train[target].iloc[idx1],train[target].iloc[idx2]
        truth.extend(y_cv)
#     x_train,x_cv=train[train.num!=0][features],train[train.num==0][features]
#     y_train,y_cv=train[train.num!=0].sellvolume,train[train.num==0].sellvolume
       
        if isrf:
            lr=ensemble.RandomForestRegressor(n_estimators=40,random_state=2016)
            lr.fit(x_train,y_train)
            cv.extend(lr.predict(x_cv))

            lst.append(lr.predict(test[features]))
            print(metrics.mean_squared_error(y_cv,lr.predict(x_cv))**.5)
            cv_scores.append(lr.feature_importances_)



        elif islr:

            lr=linear_model.LinearRegression()
            lr.fit(x_train,y_train)
            cv.extend(lr.predict(x_cv))

            lst.append(lr.predict(test[features]))
            print(metrics.mean_squared_error(y_cv,lr.predict(x_cv))**.5)
            cv_scores.append(np.absolute(lr.coef_))


        else:

            dtrain=xgb.DMatrix(x_train.values,y_train)
            dvalid=xgb.DMatrix(x_cv.values,y_cv)
            dtest=xgb.DMatrix(test[features].values)
            watchlist = [ (dtrain, 'train'),(dvalid, 'cv')]

            gbm=xgb.train(params,dtrain,6000,evals=watchlist,early_stopping_rounds=30,verbose_eval=.00001);
            cv.extend(gbm.predict(dvalid))
            cv_scores.append(metrics.mean_absolute_error(y_cv,gbm.predict(dvalid)))
            lst.append(gbm.predict(dtest))
    print ('Overall',metrics.mean_squared_error(truth,np.array(cv))**.5)
    #     print ('Test error',metrics.mean_absolute_error(test.DV,np.average(lst,axis=0)))
    return cv,lst,np.average(cv_scores,axis=0),truth


# In[20]:

cv_preds=pd.DataFrame()
lst1=[]
lst2=[]
skf=cross_validation.KFold(len(train),n_folds=5,random_state=0)
for _,idx2 in skf:
    lst1.extend(train.buyvolume.iloc[idx2])
    lst2.extend(train.sellvolume.iloc[idx2])

cv_preds['buyvolume']=lst1
cv_preds['sellvolume']=lst2


# In[32]:

output=model(train,test,features,params,0,1,'buyvolume')
cv_preds['buy_lr']=output[0]
test_preds['buy_lr']=np.average(output[1],axis=0)

output=model(train,test,features,params,0,1,'sellvolume')
cv_preds['sell_lr']=output[0]
test_preds['sell_lr']=np.average(output[1],axis=0)

output=model(train,test,features,params,0,0,'buyvolume')
cv_preds['buy_xgb']=output[0]
test_preds['buy_xgb']=np.average(output[1],axis=0)

output=model(train,test,features,params,0,0,'sellvolume')
cv_preds['sell_xgb']=output[0]
test_preds['sell_xgb']=np.average(output[1],axis=0)



# In[120]:

result1=pd.DataFrame()
result1['isin']=test['isin']



# In[119]:

output=model(cv_preds,test_preds,['buy_lr','buy_xgb','sell_lr','sell_xgb'],params,0,0,'buyvolume')
result1['buyvolume']=np.average(output[1],axis=0)

output=model(cv_preds,test_preds,['buy_lr','buy_xgb','sell_lr','sell_xgb'],params,0,0,'sellvolume')
result1['sellvolume']=np.average(output[1],axis=0)


# In[121]:

result1.to_csv('outout.csv',index=0)

