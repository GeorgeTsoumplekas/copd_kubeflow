import kfp
import kfp.dsl as dsl
from kfp.dsl import Output, Dataset, Artifact, Input


@dsl.component(base_image="python:3.10", packages_to_install=["pandas==2.2.0"])
def statistical_analysis_component(
    data_csv: Input[Dataset], statistical_analysis_results: Output[Artifact]
):
    import csv
    import codecs
    import pandas as pd
    from statistical_analysis import check_statistical_analysis

    data = pd.read_csv(data_csv.path, index_col=0)
    if "copd" in data.columns.tolist():
        data = data.drop(columns=["copd"])

    csv_reader = csv.DictReader(codecs.iterdecode(data_csv.path, "utf-8"))

    statistical_analysis_results = check_statistical_analysis(data, csv_reader)

    return statistical_analysis_results


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        statistical_analysis_component, "statistical_analysis_component.yaml"
    )
