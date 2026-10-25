import pandas as pd

def data():
    # load data
    X_test_raw = pd.read_csv("project1/data/X_test.csv", header=0)
    X_train_raw = pd.read_csv("project1/data/X_train.csv", header=0)
    y_train_raw = pd.read_csv("project1/data/y_train.csv", header=0)

    # assert alingment and shape
    assert X_train_raw.iloc[:, 0].equals(y_train_raw.iloc[:, 0])
    assert X_train_raw.shape[1] == X_test_raw.shape[1]

    # store test ids for submission
    test_ids = X_test_raw.iloc[:, 0].astype(int).to_numpy()

    # drop index
    X_test = X_test_raw.iloc[:, 1:]
    X_train = X_train_raw.iloc[:, 1:]
    y_train = y_train_raw.iloc[:, 1]

    # get non-unique columns
    non_constant = X_train.nunique() > 1

    # get almost uniform columns
    uniform_like = (X_train.kurt() < -0.7) & (X_train.skew().abs() < 0.1)

    # drop columns
    X_train = X_train.loc[:, non_constant & ~uniform_like]
    X_test = X_test.loc[:, non_constant & ~uniform_like]

    return X_train, y_train, X_test, test_ids