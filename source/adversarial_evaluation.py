import io
import joblib
import warnings

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import classification_report

from .preprocessing import preprocess_copd


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def transfom_report(cls_report, label_mapping):
    for key, value in label_mapping.items():
        cls_report[key] = cls_report.pop(str(value))
    return cls_report


def check_adversarial_evaluation(real_data, synthetic_data, model_path, test_size):
    # Load model
    model = joblib.load(model_path)

    # Preprocess data
    _, X_test_real, _, y_test_real, label_mapping_real = preprocess_copd(
        real_data, test_size=test_size
    )
    _, X_test_synthetic, _, y_test_synthetic, label_mapping_synthetic = preprocess_copd(
        synthetic_data, test_size=test_size
    )

    # Model evaluation
    y_test_real_pred = model.predict(X_test_real).reshape((-1, 1))
    y_test_synthetic_pred = model.predict(X_test_synthetic).reshape((-1, 1))

    y_test_real_pred = model.predict(X_test_real).reshape((-1, 1))
    y_test_synthetic_pred = model.predict(X_test_synthetic).reshape((-1, 1))

    cls_report_real = classification_report(
        y_test_real, y_test_real_pred, output_dict=True
    )
    cls_report_synthetic = classification_report(
        y_test_synthetic, y_test_synthetic_pred, output_dict=True
    )

    cls_report_real = transfom_report(cls_report_real, label_mapping_real)
    cls_report_synthetic = transfom_report(
        cls_report_synthetic, label_mapping_synthetic
    )

    # Final results
    adversarial_evaluation_results = {
        "real": cls_report_real,
        "synthetic": cls_report_synthetic,
    }

    return adversarial_evaluation_results


def get_adversarial_evaluation_plots(adversarial_evaluation_results):
    real_results = adversarial_evaluation_results["real"]
    synthetic_results = adversarial_evaluation_results["synthetic"]

    num_plots = len(real_results.keys()) - 2
    nrows = num_plots // 2 + 1
    ncols = 2

    dataset_labels = ["Real", "Synthetic"]
    classes = list(real_results.keys())
    classes = [x for x in classes if x not in ["accuracy", "macro avg", "weighted avg"]]
    class_counter = 0

    axes = [None] * num_plots
    bar_width = 0.2
    colors = ["#34495E", "#E39E21"]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(nrows, 2 * ncols, hspace=0.5, wspace=0.7)

    for i in range(num_plots):
        if i == 0:
            xlabels = ["Accuracy", "Precision", "Recall", "F-Score"]
            x_axis = np.arange(len(xlabels))
            real_metrics = [
                real_results["accuracy"],
                real_results["macro avg"]["precision"],
                real_results["macro avg"]["recall"],
                real_results["macro avg"]["f1-score"],
            ]
            synthetic_metrics = [
                synthetic_results["accuracy"],
                synthetic_results["macro avg"]["precision"],
                synthetic_results["macro avg"]["recall"],
                synthetic_results["macro avg"]["f1-score"],
            ]

            axes[i] = fig.add_subplot(gs[0, 1:3])

            for label in dataset_labels:
                if label == "Real":
                    axes[i].bar(
                        x_axis - 0.1,
                        real_metrics,
                        bar_width,
                        label="Real Data",
                        color=colors[0],
                        edgecolor="black",
                        zorder=3,
                    )
                else:
                    axes[i].bar(
                        x_axis + 0.1,
                        synthetic_metrics,
                        bar_width,
                        label="Synthetic Data",
                        color=colors[1],
                        edgecolor="black",
                        zorder=3,
                    )

            axes[i].legend(
                loc="upper left", bbox_to_anchor=(1.05, 0.7), title="Dataset Type"
            )
            axes[i].set_xticks(
                [r + bar_width / 2 for r in range(len(xlabels))], xlabels
            )
            axes[i].grid(axis="x")

            axes[i].set_title("Total Population Metrics")
            axes[i].set_ylim([0, 1])
            axes[i].set_ylabel("Metric Value")
            axes[i].set_xlabel("Metric")
        else:
            row = (i + 1) // 2
            col = (i + 1) % 2

            xlabels = ["Precision", "Recall", "F-Score"]
            x_axis = np.arange(len(xlabels))

            target_class = classes[class_counter]
            real_metrics = [
                real_results[target_class]["precision"],
                real_results[target_class]["recall"],
                real_results[target_class]["f1-score"],
            ]

            synthetic_metrics = [
                synthetic_results[target_class]["precision"],
                synthetic_results[target_class]["recall"],
                synthetic_results[target_class]["f1-score"],
            ]

            if col == 0:
                axes[i] = fig.add_subplot(gs[row, :2])

            else:
                axes[i] = fig.add_subplot(gs[row, 2:])

            for label in dataset_labels:
                if label == "Real":
                    axes[i].bar(
                        x_axis - 0.1,
                        real_metrics,
                        bar_width,
                        label="Real Data",
                        color=colors[0],
                        edgecolor="black",
                        zorder=3,
                    )
                else:
                    axes[i].bar(
                        x_axis + 0.1,
                        synthetic_metrics,
                        bar_width,
                        label="Synthetic Data",
                        color=colors[1],
                        edgecolor="black",
                        zorder=3,
                    )

            axes[i].set_xticks(
                [r + bar_width / 2 for r in range(len(xlabels))], xlabels
            )
            axes[i].grid(axis="x")

            axes[i].set_title("Metrics for class " + target_class)
            axes[i].set_ylim([0, 1])
            axes[i].set_ylabel("Metric Value")
            axes[i].set_xlabel("Metric")
            class_counter += 1

    plt.suptitle("Adversarial Evaluation Report", x=0.515, fontsize="x-large")

    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches="tight", format="png")
    plt.close(fig)

    return img_buf
