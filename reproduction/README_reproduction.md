# How to Reproduce the Analyses Presented in the Original Article

Reproduction procedures for the analyses in the original article [(URL, unpublished)] are described below.

## Preparation

### 1. Create a virtual environment and install required packages

First, create a virtual environment and install the `emtdicompute` package. See `README.md` for installation instructions.
In addition, `matplotlib` and `seaborn` are required.

```bash
pip install matplotlib seaborn
```

Don't forget to activate the virtual environment.

### 2. Copy folders

Once you have created the virtual environment, copy the folders and files under `reproduction/` into your environment.

## Validation (Section 2.2: Validation of the Tool)

The required files are located under `validation/`. By running the following commands, you can reproduce the validation analysis.

```bash
cd validation
python ./validation.py
```

You will get the following result files under `validation/outputs/`:

- `01_cube_validation_d50.csv`
- `02_tallcube_validation_d50.csv`
- `03_widecube_validation_d50.csv`

By default, the sampling density is set to 50. You can modify it by changing the value of `DENSITY` in the script (line 18). You can verify that varying the sampling density does not affect the computed values.

## Sampling Density Determination (Section 2.3: Practical Demonstration of Sampling Density Determination)

The required files are located under `sampling_density/`. By running the following commands, you can reproduce the sampling density determination process.

**CAUTION:**
This process outputs a large number of CSV files and is relatively time-consuming compared to [Validation](#validation-section-22-validation-of-the-tool). If you want to reduce the processing time and the number of output files, uncomment line 27 of `sdd.py` and change the number after `":"`, e.g., `seeds = seeds[:20]`. This value controls the number of iterations of visible volume computation with different sampling seeds.

```bash
cd sampling_density
python ./sdd.py
```

Once the process is complete, draw a plot of the visible volume convergence.

```bash
python ./plot.py
```

You will get a plot image of computed visible volumes at varying sampling densities. The image will be saved as `sampling_density/outputs/plot.pdf`.

**Tips:**
This procedure can be reused for your own project by replacing the input OBJ and CSV files. Place your building model (OBJ) in `sampling_density/` and replace `OBJ_FILENAME` in `obj_vpt_setting.py` with your OBJ filename. Similarly, replace `cam_04.csv` with the location of a vantage point in your building, and update `VPT_FILENAME` in `obj_vpt_setting.py`. You can also change the range and step of sampling densities by modifying `densities` on line 19 of `sdd.py`.

## Case Study (Section 2.4: Case Study)

The required files are located under `case_study/`.

### Point Analysis

```bash
cd case_study/
python ./point_analysis.py
```

The outputs will appear in `case_study/outputs/point_analysis/`.

To obtain an arrayed plot of the results, run:

```bash
python ./plot_point_analysis.py
```

The figure will be saved as `case_study/outputs/point_analysis/cs_point_analysis_plot.pdf`.

You can also get visualisations of the visible sampled points from the vantage point by uncommenting lines 46 to 53 in `point_analysis.py`. Press `Enter` when the terminal asks `force_usecache? Type "False" or "True"`.

- Modify the index of `VANTAGE_PATHS[0]` in line 47 to change the view orientation.
- Setting `mode = 'td'` visualises the visible sampled points on horizontal surfaces. Green (red) points are on the bottom (top) surfaces.
- Setting `mode = 'vh'` visualises all visible sampled points. Green (red) points are on vertical (horizontal) surfaces.

### Path Analysis

```bash
cd case_study/
python ./path_analysis.py
```

The outputs will appear in `case_study/outputs/path_analysis/`.

To obtain an arrayed plot of the results, run:

```bash
python ./plot_path_analysis.py
```

The figure will be saved as `case_study/outputs/path_analysis/cs_path_analysis_plot.pdf`.

### Field Analysis

Run the following commands:

```bash
cd case_study/
python ./field_analysis.py
```

Pop-up windows illustrating the results will appear several times. Close each window to proceed with the analysis.

Finally, images for each level and metric will be saved under `outputs/field_analysis/`. The suffixes at the end of each filename indicate the metric: `vh` = v-h ratio, `td` = t-d ratio, `vj` = vertical jaggedness, and `vv` = visible volume.
