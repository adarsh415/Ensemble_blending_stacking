import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

def run_training(folds):
    df = pd.read_csv("F:\AI_repos\EnsembleNStack\data\\train_folds.csv")
    df['review'] = df.review.apply(str)
    df_train = df[df.kfold != folds].reset_index(drop=True)
    df_valid = df[df.kfold == folds].reset_index(drop=True)

    tfv = CountVectorizer()
    tfv.fit(df_train.review.values)

    xtrain = tfv.transform(df_train.review.values)
    xvalid = tfv.transform(df_valid.review.values)

    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values


    clf = linear_model.LogisticRegression()
    clf.fit(xtrain, ytrain)
    ypred = clf.predict_proba(xvalid)[:,1]


    auc = metrics.roc_auc_score(yvalid, ypred)
    df_valid.loc[:, "lr_cnt_pred"] = ypred
    print(f"fold: {folds}, auc_score: {auc}")

    return df_valid[["id", "sentiment", "kfold", 'lr_cnt_pred']]


if __name__=="__main__":
    dfs =[]
    for j in range(5):
        dfs.append(run_training(j))

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("models/lr_cnt.csv", index=False)