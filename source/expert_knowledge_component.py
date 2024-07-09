import kfp
import kfp.dsl as dsl
from kfp.dsl import Input, Output, Dataset, Artifact


@dsl.component(base_image="python:3.10", packages_to_install=["pandas==2.2.0"])
def expert_knowledge_component(
    data_csv: Input[Dataset], expert_knowledge_results: Output[Artifact]
):
    import pandas as pd
    from source.expert_knowledge import check_expert_knowledge

    data = pd.read_csv(data_csv.path, index_col=0)

    if "copd" in data.columns.tolist():
        data = data.drop(columns=["copd"])

    expert_knowledge_results = check_expert_knowledge(data)

    return expert_knowledge_results


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        expert_knowledge_component, "expert_knowledge_component.yaml"
    )
