import kfp
from kfp import dsl
from kfp.components import load_component_from_file
from kfp.dsl import Dataset

expert_knowledge_op = load_component_from_file("/home/tsoump/post_market/source/expert_knowledge_component.yaml")
statistical_analysis_op = load_component_from_file("/home/tsoump/post_market/source/statistical_analysis_component.yaml")
adversarial_evaluation_op = load_component_from_file("/home/tsoump/post_market/source/adversarial_evaluation_component.yaml")

@dsl.pipeline(
    name='Post Market Evaluation Pipelien',
    description='A pipeline that ingests the real and synthetic data and produces the post market evaluation results based on the three evaluation axes defined.'
)
def post_market_evaluation_pipeline(
    real_data_csv: Dataset,
    synthetic_data_csv: Dataset,
    model_name: str,
    test_size: float,
):
    expert_knowledge_task = expert_knowledge_op(data_csv=synthetic_data_csv)

    statistical_analysis_task = statistical_analysis_op(data_csv=synthetic_data_csv)

    adversarial_evaluation_task = adversarial_evaluation_op(real_data_csv=real_data_csv,
                                                            synthetic_data_csv=synthetic_data_csv,
                                                            model_name=model_name,
                                                            test_size=test_size)

# Compile pipeline
kfp.compiler.Compiler().compile(post_market_evaluation_pipeline, 'post_market_evaluation_pipeline.yaml')
