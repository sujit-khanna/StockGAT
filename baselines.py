import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import glob
import collections


EDGE_THRESH = 0
LABEL_THRESH = 0.0005
DIST_THRESH = 0
TECH_IND = ["rsi", "macd", "adx"]
FUN_IND = ["rating_Score", "pe"]
START_DATE = "2017-01-01"

TRAIN, VALID, TEST = ("2017-01-01", "2018-12-31"), ("2019-01-01", "2019-08-31"), ("2019-09-01", "2021-04-31")


def preprocess_datafiles(node_file, fun_file, tech_file):
    technicals = {key.split("/")[-1].split(".")[0]: pd.read_csv(key) for key in
                     glob.glob(tech_file)}

    fundamentals = {key.split("/")[-1].split(".")[0]: pd.read_csv(key) for key in
                    glob.glob(fun_file)}

    nodes = {key.split("/")[-1].split(".")[0]: pd.read_csv(key) for key in
                   glob.glob(node_file)}
    tickers = list(nodes.keys())
    graph_dict_train = collections.defaultdict(dict)
    graph_dict_test = collections.defaultdict(dict)
    for ticker in tickers:
        ret_df = nodes[ticker][["Date", "weekly_ret"]].iloc[1:]
        ret_df.loc[ret_df["weekly_ret"] >= LABEL_THRESH, "labels"] = 2
        ret_df.loc[ret_df["weekly_ret"] <= -LABEL_THRESH, "labels"] = 0
        ret_df["labels"] = ret_df["labels"].fillna(1)

        tech_df = technicals[ticker][["Date"] + TECH_IND]
        fun_df = fundamentals[ticker][["date"] + FUN_IND]
        fun_df = fun_df.rename(columns={"date": "Date"})
        full_df = pd.merge(ret_df[["Date", "labels"]], tech_df, on="Date", how="left").ffill()
        full_df = pd.merge(full_df, fun_df, on="Date", how="left").ffill()
        full_df = full_df.set_index("Date")
        graph_dict_train[ticker] = full_df.loc[(full_df.index >= TRAIN[0]) & (full_df.index <= TRAIN[1])]
        graph_dict_test[ticker] = full_df.loc[(full_df.index >= TEST[0]) & (full_df.index <= TEST[1])]

    return graph_dict_train, graph_dict_test, tickers


def baseline_training(tickers, train_graph_data, test_graph_data, classifier="RF"):

    y_true, y_pred = [], []
    for ticker in tickers:
        print(f"Evaluating ticker {ticker}")
        if ticker=="BRK_B":
            print()
        df_train = train_graph_data[ticker]
        df_train = df_train.fillna(0)
        df_test = test_graph_data[ticker]
        df_test = df_test.fillna(0)
        X_train, y_train = df_train[["rsi", "macd", "adx", "rating_Score", "pe"]],  df_train["labels"]
        X_test, y_test = df_test[["rsi", "macd", "adx", "rating_Score", "pe"]], df_test["labels"]
        if classifier=="RF":
            clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=2, random_state=0))
        else:
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        y_true.append(y_test.values.tolist())
        y_pred.append(preds.tolist())
        print(f"Test Set accuracy = {accuracy_score(y_test, preds)}")

    y_true_full = np.asarray(y_true)
    y_pred_full = np.asarray(y_pred)

    print(f"Overall Classification Report for {classifier} model")
    print(classification_report(y_true_full.flatten(), y_pred_full.flatten()))
    print(f"jaccard score {jaccard_score(y_true_full.flatten(), y_pred_full.flatten(), average='weighted')}")


if __name__ == '__main__':
    indicator_file_path = "./data/write_files/csv/indicators/*"
    rating_file_path = "./data/write_files/csv/ratings/*"
    label_file_path = "./data/write_files/csv/labels/*"
    graph_dict_train, graph_dict_test, tickers = preprocess_datafiles(label_file_path, rating_file_path, indicator_file_path)
    classification_models = ["RF", "SVM"]
    for models in classification_models:
        baseline_training(tickers, graph_dict_train, graph_dict_test, classifier=models)



