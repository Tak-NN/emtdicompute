import matplotlib.pyplot as plt
import pandas as pd

CSV_DIR_A = "./outputs/point_analysis/cs_point_analysis_dirA.csv"
CSV_DIR_B = "./outputs/point_analysis/cs_point_analysis_dirB.csv"
CSV_WITHOUT_FOV = "./outputs/point_analysis/cs_point_analysis_unlimfov.csv"

DICT_LBL_MTR = {
    'visible_volume': "Visible Volume [m$^3$]",
    'vh_ratio': 'V-H Ratio',
    'td_ratio': 'T-D Ratio',
    'v_jaggedness': 'Vertical Jaggedness',
}


def load_metric(csv_path: str, metric: str) -> float:
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not found in {csv_path}")
    return float(df[metric].iloc[0])


def plot_points(metrics: list[str]) -> None:
    y_positions = [1, 2, 3]

    plt.rcParams["font.family"] = 'Calibri'
    plt.rcParams['font.size'] = 20

    fig, axes = plt.subplots(2, 2, figsize=(24/2.54, 14/2.54), sharex=False)
    if len(metrics) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        values = [
            load_metric(CSV_DIR_A, metric),
            load_metric(CSV_DIR_B, metric),
            load_metric(CSV_WITHOUT_FOV, metric),
        ]
        max_val = max(values)
        ax.scatter(values[0], y_positions[0], color="forestgreen", s=70)
        ax.scatter(values[1], y_positions[1], marker="s", color="orange", s=70)
        ax.scatter(values[2], y_positions[2], marker="x", color="gray", s=70)
        ax.yaxis.set_visible(False)
        ax.set_xlabel(DICT_LBL_MTR[metric])
        ax.set_xlim(0, max_val * (1.1))
        ax.set_ylim(0.5, 3.5)
        ax.tick_params(labelsize=16)

    fig.tight_layout()
    plt.savefig('./outputs/point_analysis/cs_point_analysis_plot.pdf')
    plt.show()


if __name__ == "__main__":
    plot_points(["visible_volume", "vh_ratio", "td_ratio", "v_jaggedness"])
