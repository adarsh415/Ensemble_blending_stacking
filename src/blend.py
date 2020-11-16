import glob
import pandas as pd
import numpy as np
from sklearn import metrics

if __name__=='__main__':
    files = glob.glob("models/*.csv")

    df = None

    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            tem_df = pd.read_csv(f)
            df = df.merge(tem_df, on="id", how="left")
    #print(df.head(10))

    pred_cols = ["lr_pred", "lr_cnt_pred", "rf_svd_pred"]

    targets = df.sentiment.values

    for col in pred_cols:
        auc = metrics.roc_auc_score(df.sentiment.values, df[col].values)
        print(f"pred col = {col} overall auc= {auc}")
    
    print("average")
    avg_pred = np.mean(df[pred_cols].values, axis=1)
    #print(avg_pred[0])
    print(metrics.roc_auc_score(targets, avg_pred))

    print("weighted average")
    lr_pred = df.lr_pred.values 
    lr_cnt_pred = df.lr_cnt_pred.values 
    rf_svd_pred = df.rf_svd_pred.values 

    avg_pred = (3*lr_pred + lr_cnt_pred + rf_svd_pred)/5
    print(metrics.roc_auc_score(targets, avg_pred))

    print("rank averaging")
    lr_pred = df.lr_pred.rank().values 
    lr_cnt_pred = df.lr_cnt_pred.rank().values 
    rf_svd_pred = df.rf_svd_pred.rank().values 

    avg_pred = (lr_pred + lr_cnt_pred + rf_svd_pred)/3
    print(metrics.roc_auc_score(targets, avg_pred))

    print("Weighted rank averaging")
    lr_pred = df.lr_pred.rank().values 
    lr_cnt_pred = df.lr_cnt_pred.rank().values 
    rf_svd_pred = df.rf_svd_pred.rank().values 

    avg_pred = (3*lr_pred + lr_cnt_pred + rf_svd_pred)/5
    print(metrics.roc_auc_score(targets, avg_pred))

