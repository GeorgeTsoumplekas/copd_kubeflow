import io

from matplotlib import pyplot as plt
import matplotlib.lines as lines
import pandas as pd
import seaborn as sns
from yellowbrick.style import set_palette

set_palette("flatui")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

pd.set_option("display.float_format", "{:.3f}".format)


def convert_to_numeric(value):
    if value is None:
        return float("nan")
    if "." in value:
        try:
            return float(value)
        except ValueError:
            return value
    else:
        try:
            return int(value)
        except ValueError:
            return value


def read_csv_to_dict(csv_reader):
    data_dict = {}
    for row in csv_reader:
        for key, value in row.items():
            if key not in data_dict:
                data_dict[key] = []

            # Convert to int or float if possible
            converted_value = convert_to_numeric(value)
            data_dict[key].append(converted_value)

    return data_dict


def check_duplicates(data):
    duplicates_idx = data[data.duplicated()].index.tolist()

    return duplicates_idx


def check_null_values(data):
    null_results = {}

    for column in data.columns:
        column_null_idx = data[data[column].isnull()].index.tolist()
        null_results[column] = column_null_idx

    return null_results


def check_data_type_consistency(csv_reader):
    data_dict = read_csv_to_dict(csv_reader)
    true_datatypes_df = pd.DataFrame(data_dict).drop(columns=[""], axis=1)
    data_type_results = {}
    for column in true_datatypes_df.columns:
        column_types = {}
        for value in true_datatypes_df[column]:
            if type(value).__name__ in list(column_types.keys()):
                column_types[type(value).__name__] += 1
            else:
                column_types[type(value).__name__] = 1
        data_type_results[column] = column_types

    return data_type_results


def check_zero_cardinality(data):
    zero_cardinality_columns = []

    for column in data.columns:
        if data[column].nunique() == 1:
            zero_cardinality_columns.append(column)
    return zero_cardinality_columns


def check_feature_outliers(data, feature, iqr_factor=1.7):
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr

    return data[
        (data[feature] > upper_bound) | (data[feature] < lower_bound)
    ].index.tolist()


def check_outliers(data, statistical_analysis_results):
    outliers_results = {}

    for column in data.columns:
        feature_data_types = statistical_analysis_results["feature_data_type"][column]

        total_counter = 0
        float_counter = 0

        for key, value in feature_data_types.items():
            total_counter += value

            if key == "float":
                float_counter += value
        float_fraction = float_counter / total_counter
        if float_fraction >= 0.9:
            outliers_idx = check_feature_outliers(data, column)
            outliers_results[column] = outliers_idx

    return outliers_results


def check_statistical_analysis(data, csv_reader):
    statistical_analysis_results = {}

    statistical_analysis_results["duplicates_idx"] = check_duplicates(data)
    statistical_analysis_results["null_idx"] = check_null_values(data)
    statistical_analysis_results["feature_data_type"] = check_data_type_consistency(
        csv_reader
    )
    statistical_analysis_results["zero_cardinality_columns"] = check_zero_cardinality(
        data
    )
    statistical_analysis_results["outliers_idx"] = check_outliers(
        data, statistical_analysis_results
    )
    return statistical_analysis_results


def get_statistical_analysis_heatmaps(data, statistical_analysis_results):
    null_values = pd.DataFrame(False, index=data.index, columns=data.columns)
    for col, indices in statistical_analysis_results["null_idx"].items():
        for idx in indices:
            null_values.at[idx, col] = True

    duplicate_values = pd.DataFrame(False, index=data.index, columns=data.columns)
    for duplicate_row_idx in statistical_analysis_results["duplicates_idx"]:
        duplicate_values.loc[duplicate_row_idx] = True

    outlier_values = pd.DataFrame(False, index=data.index, columns=data.columns)
    for col, indices in statistical_analysis_results["outliers_idx"].items():
        for idx in indices:
            outlier_values.at[idx, col] = True

    zero_cardinality_values = pd.DataFrame(
        False, index=data.index, columns=data.columns
    )
    for zero_cardinality_col in statistical_analysis_results[
        "zero_cardinality_columns"
    ]:
        zero_cardinality_values[zero_cardinality_col] = True

    colours = ["#34495E", "tomato"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    heatmap_1 = sns.heatmap(
        null_values, ax=axes[0, 0], cmap=sns.color_palette(colours), yticklabels=10
    )
    heatmap_2 = sns.heatmap(
        duplicate_values, ax=axes[0, 1], cmap=sns.color_palette(colours), yticklabels=10
    )
    heatmap_3 = sns.heatmap(
        outlier_values, ax=axes[1, 0], cmap=sns.color_palette(colours), yticklabels=10
    )
    heatmap_4 = sns.heatmap(
        zero_cardinality_values,
        ax=axes[1, 1],
        cmap=sns.color_palette(colours),
        yticklabels=10,
    )

    for i in range(1, data.shape[0]):
        axes[0, 0].add_line(
            lines.Line2D([i, i], [0, len(data)], color="white", linewidth=1)
        )
        axes[0, 1].add_line(
            lines.Line2D([i, i], [0, len(data)], color="white", linewidth=1)
        )
        axes[1, 0].add_line(
            lines.Line2D([i, i], [0, len(data)], color="white", linewidth=1)
        )
        axes[1, 1].add_line(
            lines.Line2D([i, i], [0, len(data)], color="white", linewidth=1)
        )

    colorbar_1 = heatmap_1.collections[0].colorbar
    colorbar_1.set_ticks([0, 1])
    colorbar_1.set_ticklabels(["Valid", "Missing"])

    colorbar_2 = heatmap_2.collections[0].colorbar
    colorbar_2.set_ticks([0, 1])
    colorbar_2.set_ticklabels(["Valid", "Duplicate"])

    colorbar_3 = heatmap_3.collections[0].colorbar
    colorbar_3.set_ticks([0, 1])
    colorbar_3.set_ticklabels(["Valid", "Outlier"])

    colorbar_4 = heatmap_4.collections[0].colorbar
    colorbar_4.set_ticks([0, 1])
    colorbar_4.set_ticklabels(["Valid", "Zero Cardinality"])

    for ax in axes.flatten():
        ax.tick_params(axis="y", labelrotation=0)
    for ax in axes.flatten():
        ax.tick_params(axis="x", labelrotation=60)

    axes[0, 0].set_ylabel("Index")
    axes[0, 0].set_title("Missing Values")

    axes[0, 1].set_ylabel("Index")
    axes[0, 1].set_title("Duplicate Values")

    axes[1, 0].set_ylabel("Index")
    axes[1, 0].set_title("Outlier Values")

    axes[1, 1].set_ylabel("Index")
    axes[1, 1].set_title("Zero Cardinality Features")

    plt.suptitle("Statistical Analysis Checks")
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches="tight", format="png")
    plt.close(fig)

    return img_buf


def get_statistical_analysis_barplots(data, statistical_analysis_results):
    feature_data_types = statistical_analysis_results["feature_data_type"]

    for key, value in feature_data_types.items():
        for inner_key in value:
            value[inner_key] = round(value[inner_key] / len(data) * 100, 2)

    multi_type_features = {}

    for key, value in feature_data_types.items():
        for inner_key, inner_value in value.items():
            if inner_value != 100.0:
                if key not in multi_type_features:
                    multi_type_features[key] = {}
                multi_type_features[key][inner_key] = inner_value

    num_plots = len(multi_type_features)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))

    if num_plots == 1:
        axes = [axes]

    colors = ["#34495E", "#E39E21", "#BAD0E9", "#F1F1E6"]

    for i, (key, inner_dict) in enumerate(multi_type_features.items()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row][col]

        ax.grid(axis="x")
        ax.bar(
            inner_dict.keys(),
            inner_dict.values(),
            color=colors[: len(inner_dict.keys())],
            zorder=2,
        )
        ax.set_title(key)
        ax.set_ylabel("Data Type Ratio (%)")
        ax.set_ylim([0, 100])

    plt.suptitle("Distribution of data types for features with multiple data types")
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches="tight", format="png")
    plt.close(fig)

    return img_buf
