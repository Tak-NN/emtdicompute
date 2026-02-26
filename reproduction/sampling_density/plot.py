import pandas as pd
import numpy as np
import os
import glob
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import objname_setting as objst



obj_basename = objst.OBJ_FILENAME[:-4]
obj_namelen = len(obj_basename)

SEP_DIR = f'./outputs/{obj_basename}/'
UNIFIED_PATH = f'./outputs/{obj_basename}.csv'


def unified_df():
    if not Path(UNIFIED_PATH).exists():
        df_unified = pd.DataFrame()

        files = glob.glob(f"{SEP_DIR}/*.csv")

        for f in files:
            fname = os.path.basename(f)

            density = fname[obj_namelen+1:]
            density = density.split("_", 1)[0]
            print(f"{density}")

            df = pd.read_csv(f)
            df['density'] = float(density)

            df_unified = pd.concat([df_unified, df])

        df_unified = df_unified.sort_values('density')
        df_unified.to_csv(UNIFIED_PATH, index = False)
        print(f"file saved: {UNIFIED_PATH}")

    else:
        df_unified = pd.read_csv(UNIFIED_PATH)

    return df_unified




DICT_LBL_MTR = {
    'visible_volume': 'Visible Volume [m$^3$]'
}

def plot_box(df, mtr):

    df = df.copy()
    df["density"] = pd.to_numeric(df["density"], errors="coerce")
    df = df.dropna(subset=["density", mtr])

    plt.rcParams["font.family"] = "Calibri"

    plt.figure(figsize = (16 / 2.54, 10 / 2.54))
    color = 'crimson'
    strip_kwargs = {
        "data": df,
        "x": "density",
        "y": mtr,
        "jitter": True,
        "size": 1.8,
        "alpha": 0.9,
        "color": "black",
        # "color": "lightsteelblue",
        "zorder": 1,
    }
    box_kwargs = {
        "data": df,
        "x": "density",
        "y": mtr,
        "color": "white",
        "fliersize": 0,
        "linewidth": 1,
        "zorder": 2,
    }
    if "native_scale" in inspect.signature(sns.stripplot).parameters:
        strip_kwargs["native_scale"] = True
    if "native_scale" in inspect.signature(sns.boxplot).parameters:
        box_kwargs["native_scale"] = True

    sns.stripplot(**strip_kwargs)
    box_alpha = 0.2
    sns.boxplot(
        **box_kwargs,
        boxprops=dict(edgecolor=color, facecolor=mcolors.to_rgba("palevioletred", alpha=box_alpha)),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        medianprops=dict(color=color),
    )
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Sampling Density", fontsize=20)
    ax.set_ylabel(DICT_LBL_MTR[mtr], fontsize=20)
    ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig('./outputs/plot.pdf')
    plt.show()




if __name__ == "__main__":
    df_unified = unified_df()

    plot_box(df_unified, 'visible_volume')
