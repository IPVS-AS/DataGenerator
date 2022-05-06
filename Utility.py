from sklearn.model_selection import train_test_split


def get_train_test_X_y(df_train, df_test, n_features=100):
    X_train = df_train[[f"F{i}" for i in range(n_features)]].to_numpy()
    y_train = df_train["target"].to_numpy()

    X_test = df_test[[f"F{i}" for i in range(n_features)]].to_numpy()
    y_test = df_test["target"].to_numpy()
    return X_train, X_test, y_train, y_test


def _train_test_splitting(df, train_size=0.7):
    try:
        df_train, df_test = train_test_split(df, train_size=train_size, stratify=df[["group", "target"]])
    except ValueError as e:
        print(e)
        try:
            df_train, df_test = train_test_split(df, train_size=train_size, stratify=df["group"])
        except ValueError as e:
            print(e)
            try:
                df_train, df_test = train_test_split(df, train_size=train_size, stratify=df["target"])
            except ValueError as e:
                print(e)
                df_train, df_test = train_test_split(df, train_size=train_size)
    return df_train, df_test
