import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Calibri"
plt.rcParams["font.size"] = 10
df_A = pd.read_csv('./outputs/path_analysis/scenario_path_dir_A.csv')
df_B = pd.read_csv('./outputs/path_analysis/scenario_path_dir_B.csv')
df_B = df_B.iloc[::-1].reset_index(drop=True)
df_withoutfov = pd.read_csv('./outputs/path_analysis/scenario_unlimfov.csv')

dict_lbl_mtr = {'visible_volume': 'Visible Volume [m$^3$]', 
                'vh_ratio': "V-H Ratio",
                'td_ratio': "T-D Ratio", 
                'v_jaggedness': "Vertical Jaggedness",
                }

fig, axes = plt.subplots(2, 2, figsize=(12 *2 / 2.54, 8 *2 / 2.54), sharex=True)
axes = axes.flatten()

for ax, metric in zip(axes, dict_lbl_mtr.keys()):
    ax.plot(df_A.index, df_A[metric], color="forestgreen", label="dir_A")
    ax.plot(df_B.index, df_B[metric], "--", color="orange", label="dir_B")
    ax.plot(df_withoutfov.index, df_withoutfov[metric], "-.", color="gray", label="without_fov")

    ax.set_xlabel("Point Index", fontsize=20, fontfamily="Calibri")
    ax.set_ylabel(f"{dict_lbl_mtr[metric]}", fontsize=20, fontfamily="Calibri")
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout()
plt.savefig('./outputs/path_analysis/cs_path_analysis_plot.pdf')
plt.show()
