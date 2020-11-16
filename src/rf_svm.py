import pandas as pd
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics

def run_training(folds):
    df = pd.read_csv("F:\AI_repos\EnsembleNStack\data\\train_folds.csv")
    df['review'] = df.review.apply(str)
    df_train = df[df.kfold != folds].reset_index(drop=True)
    df_valid = df[df.kfold == folds].reset_index(drop=True)

    tfv = TfidfVectorizer()
    tfv.fit(df_train.review.values)

    xtrain = tfv.transform(df_train.review.values)
    xvalid = tfv.transform(df_valid.review.values)

    svd = TruncatedSVD(n_components=120)
    svd.fit(xtrain)

    svd_xtrain = svd.transform(xtrain)
    svd_xvalid = svd.transform(xvalid)

    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values


    clf = ensemble.RandomForestClassifier(n_estimators=101, n_jobs=-1)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict_proba(xvalid)[:,1]


    auc = metrics.roc_auc_score(yvalid, ypred)
    df_valid.loc[:, "rf_svd_pred"] = ypred
    print(f"fold: {folds}, auc_score: {auc}")

    return df_valid[["id", "sentiment", "kfold", 'rf_svd_pred']]


if __name__=="__main__":
    dfs =[]
    for j in range(5):
        dfs.append(run_training(j))

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("models/rf_svd.csv", index=False)