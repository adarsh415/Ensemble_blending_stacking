import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

    
def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]]
    xvalid = valid_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]]

    scl1 = StandardScaler()
    xtrain = scl1.fit_transform(xtrain)
    xvalid = scl1.transform(xvalid)

    opt = LogisticRegression()
    opt.fit(xtrain, train_df.sentiment.values)
    preds = opt.predict_proba(xvalid)[:,1]
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold}, {auc}")

    return opt.coef_



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

    coef = []
    for j in range(5):
        coef.append(run_training(df, j))
    coef =np.array(coef)
    print(coef)
    coefs = np.mean(coef, axis=0)
    print(coefs)

    wt_avg = (
        coefs[0][0] * df.lr_pred.values +
        coefs[0][1] * df.lr_cnt_pred.values +
        coefs[0][2] * df.rf_svd_pred.values
    )
    print("optimized weight")
    print(metrics.roc_auc_score(targets, wt_avg))