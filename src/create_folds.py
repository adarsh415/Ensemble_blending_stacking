import pandas as pd
import numpy as np
from sklearn import model_selection

if __name__=='__main__':
    df = pd.read_csv("data\labeledTrainData.tsv", sep="\t")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.sentiment.values

    skf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_,v_) in enumerate(skf.split(X=df, y=y)):
        df.loc[v_,"kfold"]= f
    
    df.to_csv("data\\train_folds.csv", index=False)
