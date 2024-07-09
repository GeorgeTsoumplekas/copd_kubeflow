import argparse
import joblib
import os

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from preprocessing import preprocess_copd


def hp_tuning_grid(model, param_space, X_train, y_train):
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gc_cv = GridSearchCV(
        model, param_space, cv=strat_cv, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    gc_cv.fit(X_train, y_train)

    return gc_cv


def hp_tuning_rand(model, param_space, X_train, y_train):
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rc_cv = RandomizedSearchCV(
        model,
        param_space,
        cv=strat_cv,
        scoring="f1_macro",
        n_iter=50,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    rc_cv.fit(X_train, y_train)
    return rc_cv


def eval_optimal(opt_model, X_train, y_train, X_test, y_test):
    # Fit optimal model
    opt_model.fit(X_train, y_train)

    # Train set metrics
    y_pred_train = opt_model.predict(X_train).reshape((-1, 1))
    print("\nTraining set metrics")
    print(classification_report(y_train, y_pred_train, digits=4))

    # Test set metrics
    y_pred_test = opt_model.predict(X_test).reshape((-1, 1))
    print("\nTest set metrics")
    print(classification_report(y_test, y_pred_test, digits=4))
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def train_random_forest(X_train, y_train, X_test, y_test):
    n_estimators_values = np.array([10, 20, 30, 40])
    max_depth_values = np.array([2, 3, 4, 5, 6, 7])
    min_samples_split_values = np.array([2, 3, 4, 5])
    min_samples_leaf_values = np.array([1, 2, 3, 4])
    max_features_values = np.array(["auto", "sqrt"])
    param_space = {
        "n_estimators": n_estimators_values,
        "max_features": max_features_values,
        "max_depth": max_depth_values,
        "min_samples_split": min_samples_split_values,
        "min_samples_leaf": min_samples_leaf_values,
    }

    rf = RandomForestClassifier(random_state=42)

    gc_cv = hp_tuning_grid(rf, param_space, X_train, y_train)

    print("Tuned Random Forest model parameters: {}".format(gc_cv.best_params_))

    # Optimal model
    n_estimators_opt = gc_cv.best_params_["n_estimators"]
    max_depth_opt = gc_cv.best_params_["max_depth"]
    min_samples_split_opt = gc_cv.best_params_["min_samples_split"]
    min_samples_leaf_opt = gc_cv.best_params_["min_samples_leaf"]
    max_features_opt = gc_cv.best_params_["max_features"]
    rf_opt = RandomForestClassifier(
        n_estimators=n_estimators_opt,
        max_depth=max_depth_opt,
        min_samples_leaf=min_samples_leaf_opt,
        min_samples_split=min_samples_split_opt,
        max_features=max_features_opt,
    )

    eval_optimal(rf_opt, X_train, y_train, X_test, y_test)

    return rf_opt


def train_xgboost(X_train, y_train, X_test, y_test):
    param_space = {
        "max_depth": randint(low=3, high=8),
        "learning_rate": uniform(loc=1e-3, scale=0.5),
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "min_child_weight": [1, 3, 5],
        "gamma": uniform(),
        "lambda": uniform(loc=2, scale=8),
        "alpha": uniform(loc=2, scale=8),
        "n_estimators": randint(low=10, high=101),
    }

    xgb = XGBClassifier(objective="multi:softmax", random_state=42)

    rc_cv = hp_tuning_rand(xgb, param_space, X_train, y_train)

    print("Tuned XGBoost model parameters: {}".format(rc_cv.best_params_))

    # Optimal model
    max_depth_opt = rc_cv.best_params_["max_depth"]
    learning_rate_opt = rc_cv.best_params_["learning_rate"]
    subsample_opt = rc_cv.best_params_["subsample"]
    colsample_bytree_opt = rc_cv.best_params_["colsample_bytree"]
    min_child_weight_opt = rc_cv.best_params_["min_child_weight"]
    gamma_opt = rc_cv.best_params_["gamma"]
    lambda_opt = rc_cv.best_params_["lambda"]
    alpha_opt = rc_cv.best_params_["alpha"]
    n_estimators_opt = rc_cv.best_params_["n_estimators"]

    early_stopping = EarlyStopping(
        rounds=5,
        data_name="validation_2",
        maximize=True,
        save_best=True,
        min_delta=1e-3,
    )

    xgb_opt = XGBClassifier(
        max_depth=max_depth_opt,
        learning_rate=learning_rate_opt,
        subsample=subsample_opt,
        colsample_bytree=colsample_bytree_opt,
        min_child_weight=min_child_weight_opt,
        gamma=gamma_opt,
        reg_lambda=lambda_opt,
        alpha=alpha_opt,
        n_estimators=n_estimators_opt,
        callbacks=[early_stopping],
        random_state=42,
    )

    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    eval_set = [(X_train_xgb, y_train_xgb), (X_test, y_test), (X_val_xgb, y_val_xgb)]

    # Eary Stopping to avoid overfitting,
    xgb_opt.fit(X_train_xgb, y_train_xgb, eval_set=eval_set, verbose=False)

    # Train set metrics
    y_pred_train = xgb_opt.predict(X_train).reshape((-1, 1))
    print("\nTraining set metrics")
    print(classification_report(y_train, y_pred_train, digits=4))
    cm = confusion_matrix(y_train, y_pred_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # Test set metrics
    y_pred_test = xgb_opt.predict(X_test).reshape((-1, 1))
    print("\nTest set metrics")
    print(classification_report(y_test, y_pred_test, digits=4))
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    return xgb_opt


def train_svm(X_train, y_train, X_test, y_test):
    param_space = [
        {
            "C": [1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3],
            "gamma": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
            "kernel": ["linear", "rbf"],
        },
    ]

    svm = SVC(random_state=42)

    gc_cv = hp_tuning_grid(svm, param_space, X_train, y_train)

    print("Tuned SVM model parameters: {}".format(gc_cv.best_params_))

    # Optimal model
    c_opt = gc_cv.best_params_["C"]
    kernel_opt = gc_cv.best_params_["kernel"]
    gamma_opt = gc_cv.best_params_["gamma"]

    svm_opt = SVC(C=c_opt, kernel=kernel_opt, gamma=gamma_opt, random_state=42)
    eval_optimal(svm_opt, X_train, y_train, X_test, y_test)

    return svm_opt


def train_logistic_regression(X_train, y_train, X_test, y_test):
    c_values = np.array([1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3])
    l1_ratio_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    param_space = {"l1_ratio": l1_ratio_values, "C": c_values}

    lr = LogisticRegression(penalty="elasticnet", solver="saga", random_state=42)

    gc_cv = hp_tuning_grid(lr, param_space, X_train, y_train)

    print("Tuned Logistic Regression model parameters: {}".format(gc_cv.best_params_))

    # Optimal model
    c_opt = gc_cv.best_params_["C"]
    l1_ratio_opt = gc_cv.best_params_["l1_ratio"]
    lr_opt = LogisticRegression(
        l1_ratio=l1_ratio_opt,
        C=c_opt,
        penalty="elasticnet",
        solver="saga",
        random_state=42,
    )

    eval_optimal(lr_opt, X_train, y_train, X_test, y_test)

    return lr_opt


def train_lightgbm(X_train, y_train, X_test, y_test):
    param_space = {
        "lambda_l1": [0.01, 0.05, 0.1, 0.2, 0.5],
        "lambda_l2": [1, 5, 10, 15, 20],
        "bagging_fraction": [0, 4, 0.6, 0.8, 1.0],
        "feature_fraction": [0.4, 0.6, 0.8, 1.0],
        "bagging_freq": [10, 25, 40],
        "max_depth": [2, 3, 4],
        "num_leaves": [5, 8, 10, 15, 20],
        "min_data_in_leaf": [5, 10, 15, 20, 25],
        "learning_rate": [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1],
        "n_estimators": [5, 10, 15, 20, 25, 30, 35],
        "seed": [42],
    }

    lgbm = LGBMClassifier(objective="multiclass", verbose=-1, random_state=42)

    rc_cv = hp_tuning_rand(lgbm, param_space, X_train, y_train)

    print("Tuned LightGBM model parameters: {}".format(rc_cv.best_params_))

    # Optimal model
    lambda_l1_opt = rc_cv.best_params_["lambda_l1"]
    lambda_l2_opt = rc_cv.best_params_["lambda_l2"]
    bagging_fraction_opt = rc_cv.best_params_["bagging_fraction"]
    feature_fraction_opt = rc_cv.best_params_["feature_fraction"]
    bagging_freq_opt = rc_cv.best_params_["bagging_freq"]
    max_depth_opt = rc_cv.best_params_["max_depth"]
    num_leaves_opt = rc_cv.best_params_["num_leaves"]
    min_data_in_leaf_opt = rc_cv.best_params_["min_data_in_leaf"]
    learning_rate_opt = rc_cv.best_params_["learning_rate"]
    n_estimators_opt = rc_cv.best_params_["n_estimators"]
    seed = rc_cv.best_params_["seed"]

    lgbm_opt = LGBMClassifier(
        objective="multiclass",
        lambda_l1=lambda_l1_opt,
        lambda_l2=lambda_l2_opt,
        bagging_fraction=bagging_fraction_opt,
        feature_fraction=feature_fraction_opt,
        bagging_freq=bagging_freq_opt,
        max_depth=max_depth_opt,
        num_leaves=num_leaves_opt,
        min_data_in_leaf=min_data_in_leaf_opt,
        learning_rate=learning_rate_opt,
        n_estimators=n_estimators_opt,
        seed=seed,
    )

    eval_optimal(lgbm_opt, X_train, y_train, X_test, y_test)

    return lgbm_opt


def train_catboost(X_train, y_train, X_test, y_test):
    param_space = {
        "loss_function": ["MultiClass"],
        "eval_metric": ["TotalF1"],
        "iterations": [10, 50, 100],
        "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 3e-1],
        "l2_leaf_reg": [1e-2, 1e-1, 1, 1e1, 1e2],
        "depth": [2, 3, 4, 5],
        "min_data_in_leaf": [5, 10, 15, 20, 25],
        "rsm": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "random_seed": [42],
    }

    cboost = CatBoostClassifier(objective="MultiClass", verbose=0)

    rc_cv = hp_tuning_rand(cboost, param_space, X_train, y_train)

    print("Tuned CatBoost model parameters: {}".format(rc_cv.best_params_))

    catboost_opt = CatBoostClassifier(**(rc_cv.best_params_))

    eval_optimal(catboost_opt, X_train, y_train, X_test, y_test)

    return catboost_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/copd.csv", dest="data_path")
    parser.add_argument("--test_size", default=0.3, dest="test_size")
    args = parser.parse_args()

    models_dir_name = "../models/"
    if not os.path.exists(models_dir_name):
        os.makedirs(models_dir_name)

    data = pd.read_csv(args.data_path, index_col=0)

    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_copd(
        data, test_size=args.test_size
    )

    # Random Forest
    rf_opt = train_random_forest(X_train, y_train, X_test, y_test)
    joblib.dump(rf_opt, "../models/random_forest.joblib")

    # XGBoost
    xgb_opt = train_xgboost(X_train, y_train, X_test, y_test)
    joblib.dump(xgb_opt, "../models/xgboost.joblib")

    # SVM
    svm_opt = train_svm(X_train, y_train, X_test, y_test)
    joblib.dump(svm_opt, "../models/svm.joblib")

    # Logistic Regression
    lr_opt = train_logistic_regression(X_train, y_train, X_test, y_test)
    joblib.dump(lr_opt, "../models/logistic_regression.joblib")

    # LightGBM
    lgbm_opt = train_lightgbm(X_train, y_train, X_test, y_test)
    joblib.dump(lgbm_opt, "../models/light_gbm.joblib")

    # CatBoost
    cboost_opt = train_catboost(X_train, y_train, X_test, y_test)
    joblib.dump(cboost_opt, "../models/catboost.joblib")


if __name__ == "__main__":
    main()
