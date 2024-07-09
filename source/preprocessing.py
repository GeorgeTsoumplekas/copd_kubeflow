from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


def check_extra_target_col(data):
    if "copd" in data.columns.tolist():
        return data.drop(columns=["copd"])
    return data


def transform_features(X_train, X_test):
    scale_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
    cat_features = [13]

    transformers = [
        ("one_hot", OneHotEncoder(), cat_features),
        ("scale", MinMaxScaler(), scale_features),
    ]
    col_transform = ColumnTransformer(
        transformers=transformers, remainder="passthrough"
    )

    imputer = SimpleImputer(strategy="mean")

    pipeline = Pipeline(steps=[("imp", imputer), ("preproc", col_transform)])

    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    return X_train_proc, X_test_proc


def transform_targets(y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_proc = label_encoder.fit_transform(y_train)
    y_test_proc = label_encoder.transform(y_test)

    label_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )

    return y_train_proc, y_test_proc, label_mapping


def preprocess_copd(data, test_size):
    data = check_extra_target_col(data)

    X = data.loc[:, data.columns != "COPDSEVERITY"]
    y = data["COPDSEVERITY"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train_proc, X_test_proc = transform_features(X_train, X_test)

    y_train_proc, y_test_proc, label_mapping = transform_targets(y_train, y_test)

    return X_train_proc, X_test_proc, y_train_proc, y_test_proc, label_mapping
