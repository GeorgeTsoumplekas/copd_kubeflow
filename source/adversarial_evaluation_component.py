import kfp
import kfp.dsl as dsl
from kfp.dsl import Input, Output, Dataset, Artifact


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["joblib==1.3.2", "pandas==2.2.0", "scikit-learn==1.4.0"],
)
def adversarial_evaluation_component(
    real_data_csv: Input[Dataset],
    synthetic_data_csv: Input[Dataset],
    model_name: Input[str],
    test_size: Input[float],
    adversarial_evaluation_results: Output[Artifact],
):
    import pandas as pd
    from adversarial_evaluation import check_adversarial_evaluation

    real_data_df = pd.read_csv(real_data_csv.path, index_col=0)
    synthetic_data_df = pd.read_csv(synthetic_data_csv.path, index_col=0)

    if model_name == "catboost":
        model_path = "./models/catboost.joblib"
    elif model_name == "lightgmb":
        model_path = "./models/light_gbm.joblib"
    elif model_name == "logistic_regression":
        model_path = "./models/logistic_regression.joblib"
    elif model_name == "random_forest":
        model_path = "./models/random_forest.joblib"
    elif model_name == "svm":
        model_path = "./models/svm.joblib"
    elif model_name == "xgboost":
        model_path = "./models/xgboost.joblib"
    else:
        raise Exception("Model not found.")

    adversarial_evaluation_results = check_adversarial_evaluation(
        real_data=real_data_df,
        synthetic_data=synthetic_data_df,
        model_path=model_path,
        test_size=test_size,
    )

    return adversarial_evaluation_results


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        adversarial_evaluation_component, "adversarial_evaluation_component.yaml"
    )
