# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""
import zipfile
import pandas as pd
import numpy as np
import loss
from scipy import sparse
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# load data
data_root = "pre"
dfTrain = pd.read_csv("%s/train.csv"%data_root)
dfTest = pd.read_csv("%s/test.csv"%data_root)
dfAd = pd.read_csv("%s/ad.csv"%data_root)
dfUser=pd.read_csv("%s/user.csv"%data_root)
dfPosition=pd.read_csv("%s/position.csv"%data_root)

# process data
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTrain=pd.merge(dfTrain,dfUser,on="userID")
dfTrain=pd.merge(dfTrain,dfPosition,on="positionID")

dfTest = pd.merge(dfTest, dfAd, on="creativeID")
dfTest=pd.merge(dfTest,dfUser,on="userID")
dfTest=pd.merge(dfTest,dfPosition,on="positionID")

# print dfTrain.shape
# print dfTest.shape
y_train = dfTrain["label"].values


enc = OneHotEncoder()
#feats = ["sitesetID","positionType","gender","education","marriageStatus","haveBaby","hometown","residence","connectionType", "telecomsOperator"]

feats = ["sitesetID","positionType","gender","hometown","residence","connectionType", "telecomsOperator"]
for i,feat in enumerate(feats):
    enc.fit(dfTrain[feat].values.reshape(-1, 1))
    x_train = enc.transform(dfTrain[feat].values.reshape(-1, 1))
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

# feature engineering/encoding
enc = OneHotEncoder()
feats = [ "creativeID","adID", "camgaignID", "advertiserID", "appID", "appPlatform"]
for i,feat in enumerate(feats):
    enc.fit(dfAd[feat].values.reshape(-1, 1))
    x_train = enc.transform(dfTrain[feat].values.reshape(-1, 1))
    x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
    X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
    # if i == 0:
    #     X_train, X_test = x_train, x_test
    # else:
    #     X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

print X_train.shape
# model training
lr = LogisticRegression()
losses=[]
for i in range(1):
    X_train_t, X_test_t, y_train_t, y_test_t = cross_validation.train_test_split(X_train, y_train, test_size=0.4,
                                                                         random_state=2)
    lr.fit(X_train_t, y_train_t)
    proba_test = lr.predict_proba(X_test_t)[:, 1]
    l = loss.logloss(np.array(y_test_t), np.array(lr.predict_proba(X_test_t)[:, 1]))
    print "interator:",i," ",l
    losses.append(l)
print "average:",sum(losses)/1

lr.fit(X_train, y_train)
proba_test = lr.predict_proba(X_test)[:,1]

l=loss.logloss(np.array(y_train),np.array(lr.predict_proba(X_train)[:,1]))
print "total: ",l


# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)




#0.103738655282
#0.102650618717
