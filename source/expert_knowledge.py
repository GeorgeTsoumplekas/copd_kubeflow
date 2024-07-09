import io

import numpy as np
import matplotlib

matplotlib.use("AGG")
from matplotlib import pyplot as plt


def determine_value(index, idx_list):
    if index in idx_list:
        return "Invalid"
    else:
        return "Valid"


def render_mpl_table(
    data_df,
    col_width=5.0,
    row_height=0.625,
    font_size=14,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="w",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    **kwargs,
):
    if ax is None:
        size = (np.array(data_df.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height]
        )
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")
    mpl_table = ax.table(
        cellText=data_df.values, bbox=bbox, colLabels=data_df.columns, **kwargs
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    fig = ax.get_figure()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches="tight", format="png")
    plt.close(fig)

    return img_buf


def check_mwt1(data):
    mwt1_error_idx = data.index[(data["MWT1"] < 100) | (data["MWT1"] > 790)].tolist()

    return mwt1_error_idx


def check_mwt2(data):
    mwt2_error_idx = data.index[(data["MWT2"] < 100) | (data["MWT2"] > 790)].tolist()

    return mwt2_error_idx


def check_mwt1best(data):
    mwt1best_error_idx = data.index[
        (data["MWT1Best"] < 100) | (data["MWT1Best"] > 790)
    ].tolist()

    return mwt1best_error_idx


def check_fev1pred(data):
    fev1pred_error_idx = data.index[
        (data["FEV1PRED"] < 11) | (data["FEV1PRED"] > 133)
    ].tolist()

    return fev1pred_error_idx


def check_sgrq(data):
    sgrq_error_idx = data.index[(data["SGRQ"] < 0) | (data["SGRQ"] > 93)].tolist()

    return sgrq_error_idx


def check_expert_knowledge(data):
    mwt1_error_idx = check_mwt1(data)
    mwt2_error_idx = check_mwt2(data)
    mwt1best_error_idx = check_mwt1best(data)
    fev1pred_error_idx = check_fev1pred(data)
    sgrq_error_idx = check_sgrq(data)

    mwt_description = "The human possible range of MWT is [100, 790] accroding to (Casanova, Ciro, et al., 2011) and (Azarisman, Mohd Shah, et al., 2007)."
    fev1pred_description = "The human possible range of FEV1PRED is [11, 133] according to (Masuko, Hironori, et al., 2011) and (Azarisman, Mohd Shah, et al., 2007)."
    sgrq_description = "The human possible range of SGRQ is [0, 93] according to (Ferrer, M., et al., 2002) and (Azarisman, Mohd Shah, et al., 2007)."

    expert_knowledge_results = {
        "mwt1_error_idx": mwt1_error_idx,
        "mwt2_error_idx": mwt2_error_idx,
        "mwt1best_error_idx": mwt1best_error_idx,
        "mwt_description": mwt_description,
        "fev1pred_error_idx": fev1pred_error_idx,
        "fev1pred_description": fev1pred_description,
        "sgrq_error_idx": sgrq_error_idx,
        "sgrq_description": sgrq_description,
    }

    return expert_knowledge_results


def get_expert_knowledge_plots(data, expert_knowledge_results):
    expert_knowledge_plot_df = data.loc[
        :, ["ID", "MWT1", "MWT2", "MWT1Best", "FEV1PRED", "SGRQ"]
    ]
    expert_knowledge_plot_df["SGRQ"] = expert_knowledge_plot_df["SGRQ"].round(2)

    expert_knowledge_plot_df["MWT1_Type"] = expert_knowledge_plot_df.index.map(
        lambda index: determine_value(index, expert_knowledge_results["mwt1_error_idx"])
    )

    expert_knowledge_plot_df["MWT2_Type"] = expert_knowledge_plot_df.index.map(
        lambda index: determine_value(index, expert_knowledge_results["mwt2_error_idx"])
    )

    expert_knowledge_plot_df["MWT1Best_Type"] = expert_knowledge_plot_df.index.map(
        lambda index: determine_value(
            index, expert_knowledge_results["mwt1best_error_idx"]
        )
    )

    expert_knowledge_plot_df["FEV1PRED_Type"] = expert_knowledge_plot_df.index.map(
        lambda index: determine_value(
            index, expert_knowledge_results["fev1pred_error_idx"]
        )
    )

    expert_knowledge_plot_df["SGRQ_Type"] = expert_knowledge_plot_df.index.map(
        lambda index: determine_value(index, expert_knowledge_results["sgrq_error_idx"])
    )

    total_error_idx = sorted(
        set(
            expert_knowledge_results["mwt1_error_idx"]
            + expert_knowledge_results["mwt2_error_idx"]
            + expert_knowledge_results["mwt1best_error_idx"]
            + expert_knowledge_results["fev1pred_error_idx"]
            + expert_knowledge_results["sgrq_error_idx"]
        )
    )

    expert_knowledge_plot_df["Overall_Type"] = expert_knowledge_plot_df.index.map(
        lambda index: determine_value(index, total_error_idx)
    )

    img_buf = render_mpl_table(
        expert_knowledge_plot_df, header_columns=0, col_width=2.5
    )

    return img_buf
