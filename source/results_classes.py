from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel


class ExpertKnowledgeResults(BaseModel):
    mwt1_error_idx: List[int] = []
    mwt2_error_idx: List[int] = []
    mwt1best_error_idx: List[int] = []
    mwt_description: str = ""
    fev1pred_error_idx: List[int] = []
    fev1pred_description: str = ""
    sgrq_error_idx: List[int] = []
    sgrq_description: str = ""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mwt1_error_idx": [12, 46],
                    "mwt2_error_idx": [19],
                    "mwt1best_error_idx": [3, 9, 52],
                    "mwt_description": "The human possible range of MWT is [A, B] accroding to (Ref. X) and (Ref. Y).",
                    "fev1pred_error_idx": [],
                    "fev1pred_description": "The human possible range of FEV1PRED is [A, B] accroding to (Ref. X) and (Ref. Y).",
                    "sgrq_error_idx": [6, 18],
                    "sgrq_descrition": "The human possible range of SGRQ is [A, B] accroding to (Ref. X) and (Ref. Y).",
                }
            ]
        }
    }


class StatisticalAnalysisResults(BaseModel):
    duplicates_idx: List[int] = []
    null_idx: Dict[str, List[int]]
    feature_data_type: Dict[str, Dict[str, int]]
    zero_cardinality_columns: List[str] = []
    outliers_idx: Dict[str, List[int]]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "duplicates_idx": [33],
                    "null_idx": {
                        "ID": [],
                        "AGE": [],
                        "PackHistory": [],
                        "COPDSEVERITY": [],
                        "MWT1": [],
                        "MWT2": [],
                        "MWT1Best": [],
                        "FEV1": [32, 45],
                        "FEV1PRED": [7, 19, 46, 92],
                        "FVC": [],
                        "FVCPRED": [],
                        "CAT": [17, 29, 46, 85],
                        "HAD": [],
                        "SGRQ": [],
                        "AGEquartiles": [],
                        "gender": [27, 39, 66, 73],
                        "smoking": [],
                        "Diabetes": [],
                        "muscular": [],
                        "hypertension": [],
                        "AtrialFib": [],
                        "IHD": [],
                    },
                    "feature_data_type": {
                        "ID": {"int": 101},
                        "AGE": {"float": 101},
                        "PackHistory": {"int": 101},
                        "COPDSEVERITY": {"str": 101},
                        "MWT1": {"int": 101},
                        "MWT2": {"int": 101},
                        "MWT1Best": {"int": 101},
                        "FEV1": {"float": 95, "str": 4, "int": 2},
                        "FEV1PRED": {"float": 97, "str": 4},
                        "FVC": {"float": 101},
                        "FVCPRED": {"int": 101},
                        "CAT": {"float": 97, "str": 4},
                        "HAD": {"int": 101},
                        "SGRQ": {"float": 101},
                        "AGEquartiles": {"int": 101},
                        "gender": {"float": 97, "str": 4},
                        "smoking": {"int": 101},
                        "Diabetes": {"int": 101},
                        "muscular": {"int": 101},
                        "hypertension": {"int": 101},
                        "AtrialFib": {"int": 101},
                        "IHD": {"int": 101},
                    },
                    "zero_cardinality_columns": ["muscular"],
                    "outliers_idx": {
                        "AGE": [32, 78],
                        "FEV1": [],
                        "FEV1PRED": [32, 71],
                        "FVC": [28, 56, 61],
                        "CAT": [],
                        "SGRQ": [63],
                        "gender": [],
                    },
                }
            ]
        }
    }


class AdversarialEvaluationResults(BaseModel):
    real: Dict[str, Any]
    synthetic: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "real": {
                        "accuracy": 0.87,
                        "macro avg": {
                            "precision": 0.66,
                            "recall": 0.714,
                            "f1-score": 0.68,
                            "support": 31,
                        },
                        "weighted avg": {
                            "precision": 0.804,
                            "recall": 0.8709,
                            "f1-score": 0.829,
                            "support": 31,
                        },
                        "MILD": {
                            "precision": 1,
                            "recall": 0.857,
                            "f1-score": 0.92307,
                            "support": 7,
                        },
                        "MODERATE": {
                            "precision": 0.92857,
                            "recall": 1,
                            "f1-score": 0.962,
                            "support": 13,
                        },
                        "SEVERE": {
                            "precision": 0.727,
                            "recall": 1,
                            "f1-score": 0.84,
                            "support": 8,
                        },
                        "VERY SEVERE": {
                            "precision": 0.92857,
                            "recall": 0.727,
                            "f1-score": 0.84,
                            "support": 3,
                        },
                    },
                    "synthetic": {
                        "accuracy": 0.483,
                        "macro avg": {
                            "precision": 0.248,
                            "recall": 0.36,
                            "f1-score": 0.2801,
                            "support": 31,
                        },
                        "weighted avg": {
                            "precision": 0.35,
                            "recall": 0.483,
                            "f1-score": 0.3898,
                            "support": 31,
                        },
                        "MILD": {
                            "precision": 0.35,
                            "recall": 0.483,
                            "f1-score": 0.3898,
                            "support": 5,
                        },
                        "MODERATE": {
                            "precision": 0.5454,
                            "recall": 0.461,
                            "f1-score": 0.5,
                            "support": 13,
                        },
                        "SEVERE": {
                            "precision": 0.45,
                            "recall": 0.461,
                            "f1-score": 0.62,
                            "support": 9,
                        },
                        "VERY SEVERE": {
                            "precision": 0.5454,
                            "recall": 0.3898,
                            "f1-score": 0.483,
                            "support": 4,
                        },
                    },
                }
            ]
        }
    }


class ModelName(str, Enum):
    catboost = "CatBoost"
    lightgbm = "LightGBM"
    logistic_regression = "Logistic Regression"
    random_forest = "Random Forest"
    svm = "SVM"
    xgboost = "XGBoost"
