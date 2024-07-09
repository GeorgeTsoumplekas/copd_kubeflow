import csv
import codecs
from typing import Annotated, Dict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response, JSONResponse
import pandas as pd
import uvicorn

from source.adversarial_evaluation import (
    check_adversarial_evaluation,
    get_adversarial_evaluation_plots,
)
from source.expert_knowledge import check_expert_knowledge, get_expert_knowledge_plots
from source.results_classes import (
    AdversarialEvaluationResults,
    ExpertKnowledgeResults,
    ModelName,
    StatisticalAnalysisResults,
)
from source.statistical_analysis import (
    check_statistical_analysis,
    get_statistical_analysis_heatmaps,
    get_statistical_analysis_barplots,
)


tags_metadata = [
    {
        "name": "Expert Knowledge",
        "description": "Operations related to error detection based on expert knowledge.",
    },
    {
        "name": "Statistical Analysis",
        "description": "Operations related to error detection following a statistical analysis approach.",
    },
    {
        "name": "Adversarial Evaluation",
        "description": "Operations related to evaluating the synthetic dataset based on model performance.",
    },
]


app = FastAPI(title="Post-Market Evaluation", openapi_tags=tags_metadata)


@app.post(
    "/expert_knowledge/logs/",
    response_model=ExpertKnowledgeResults,
    tags=["Expert Knowledge"],
)
def perform_expert_knowledge_checks(
    data: UploadFile = File(description="Examined dataset in csv format"),
) -> Dict:
    try:
        data_df = pd.read_csv(data.file, index_col=0)

        if "copd" in data_df.columns.tolist():
            data_df = data_df.drop(columns=["copd"])

        expert_knowledge_results = check_expert_knowledge(data_df)
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        data.file.close()

    return expert_knowledge_results


@app.post(
    "/expert_knowledge/plots/",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the plots.",
        }
    },
    response_class=Response,
    tags=["Expert Knowledge"],
)
def expert_knowledge_plots(
    data: UploadFile = File(description="Examined dataset in csv format"),
):
    try:
        data_df = pd.read_csv(data.file, index_col=0)

        if "copd" in data_df.columns.tolist():
            data_df = data_df.drop(columns=["copd"])

        expert_knowledge_results = check_expert_knowledge(data_df)

        img_buffer: bytes = get_expert_knowledge_plots(
            data_df, expert_knowledge_results
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        data.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")


@app.post(
    "/statistical_analysis/logs/",
    response_model=StatisticalAnalysisResults,
    tags=["Statistical Analysis"],
)
def perform_statistical_analysis_checks(
    uploaded_data: UploadFile = File(description="Examined dataset in csv format"),
) -> Dict:
    try:
        data_df = pd.read_csv(uploaded_data.file, index_col=0)
        if "copd" in data_df.columns.tolist():
            data_df = data_df.drop(columns=["copd"])

        uploaded_data.file.seek(0)

        csv_reader = csv.DictReader(codecs.iterdecode(uploaded_data.file, "utf-8"))

        statistical_analysis_results = check_statistical_analysis(data_df, csv_reader)
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        uploaded_data.file.close()

    return statistical_analysis_results


@app.post(
    "/statistical_analysis/plots/heatmap/",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the plots.",
        }
    },
    response_class=Response,
    tags=["Statistical Analysis"],
)
def statistical_analysis_heatmap_plots(
    uploaded_data: UploadFile = File(description="Examined dataset in csv format"),
):
    try:
        data_df = pd.read_csv(uploaded_data.file, index_col=0)
        if "copd" in data_df.columns.tolist():
            data_df = data_df.drop(columns=["copd"])

        uploaded_data.file.seek(0)

        csv_reader = csv.DictReader(codecs.iterdecode(uploaded_data.file, "utf-8"))

        statistical_analysis_results = check_statistical_analysis(data_df, csv_reader)

        img_buffer: bytes = get_statistical_analysis_heatmaps(
            data_df, statistical_analysis_results
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        uploaded_data.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")


@app.post(
    "/statistical_analysis/plots/barplot/",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the plots.",
        }
    },
    response_class=Response,
    tags=["Statistical Analysis"],
)
def statistical_analysis_barplot_plots(
    uploaded_data: UploadFile = File(description="Examined dataset in csv format"),
):
    try:
        data_df = pd.read_csv(uploaded_data.file, index_col=0)
        if "copd" in data_df.columns.tolist():
            data_df = data_df.drop(columns=["copd"])

        uploaded_data.file.seek(0)

        csv_reader = csv.DictReader(codecs.iterdecode(uploaded_data.file, "utf-8"))

        statistical_analysis_results = check_statistical_analysis(data_df, csv_reader)

        img_buffer: bytes = get_statistical_analysis_barplots(
            data_df, statistical_analysis_results
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        uploaded_data.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")


@app.post(
    "/adversarial_evaluation/logs/",
    response_model=AdversarialEvaluationResults,
    tags=["Adversarial Evaluation"],
)
async def perform_adversarial_evaluation(
    model_name: ModelName,
    test_size: Annotated[float, Query(gt=0, lt=1)] = 0.3,
    real_data: UploadFile = File(description="Real dataset in csv format"),
    synthetic_data: UploadFile = File(description="Synthetic dataset in csv format"),
) -> Dict:
    try:
        real_data_df = pd.read_csv(real_data.file, index_col=0)
        synthetic_data_df = pd.read_csv(synthetic_data.file, index_col=0)

        if model_name is ModelName.catboost:
            model_path = "./models/catboost.joblib"
        elif model_name is ModelName.lightgbm:
            model_path = "./models/light_gbm.joblib"
        elif model_name is ModelName.logistic_regression:
            model_path = "./models/logistic_regression.joblib"
        elif model_name is ModelName.random_forest:
            model_path = "./models/random_forest.joblib"
        elif model_name is ModelName.svm:
            model_path = "./models/svm.joblib"
        elif model_name is ModelName.xgboost:
            model_path = "./models/xgboost.joblib"
        else:
            raise HTTPException(status_code=404, detail="Model not found.")

        adversarial_evaluation_results = check_adversarial_evaluation(
            real_data=real_data_df,
            synthetic_data=synthetic_data_df,
            model_path=model_path,
            test_size=test_size,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        real_data.file.close()
        synthetic_data.file.close()

    return adversarial_evaluation_results


@app.post(
    "/adversarial_evaluation/plots/",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the plots.",
        }
    },
    response_class=Response,
    tags=["Adversarial Evaluation"],
)
def adversarial_evaluation_plots(
    model_name: ModelName,
    test_size: Annotated[float, Query(gt=0, lt=1)] = 0.3,
    real_data: UploadFile = File(description="Real dataset in csv format"),
    synthetic_data: UploadFile = File(description="Synthetic dataset in csv format"),
):
    try:
        real_data_df = pd.read_csv(real_data.file, index_col=0)
        synthetic_data_df = pd.read_csv(synthetic_data.file, index_col=0)

        if model_name is ModelName.catboost:
            model_path = "./models/catboost.joblib"
        elif model_name is ModelName.lightgbm:
            model_path = "./models/light_gbm.joblib"
        elif model_name is ModelName.logistic_regression:
            model_path = "./models/logistic_regression.joblib"
        elif model_name is ModelName.random_forest:
            model_path = "./models/random_forest.joblib"
        elif model_name is ModelName.svm:
            model_path = "./models/svm.joblib"
        elif model_name is ModelName.xgboost:
            model_path = "./models/xgboost.joblib"
        else:
            raise HTTPException(status_code=404, detail="Model not found.")

        adversarial_evaluation_results = check_adversarial_evaluation(
            real_data=real_data_df,
            synthetic_data=synthetic_data_df,
            model_path=model_path,
            test_size=test_size,
        )
        img_buffer: bytes = get_adversarial_evaluation_plots(
            adversarial_evaluation_results
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        real_data.file.close()
        synthetic_data.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8008)


if __name__ == "__main__":
    main()
