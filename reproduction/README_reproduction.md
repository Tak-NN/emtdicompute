# How to reproduce the analyses presented in the original article?
 
Reproduction procedures of analyses in the original article [(URL, unpublished)] are described below. 

## Preparation
### 1. Create a virtual environment and install required packages
At first, creaate a virtual environment and install `emtdicompute` package. See `README.md` for the installation.

In addition, `matplotlib` and `seaborn` is required.
```bash
pip install matplotlib seaborn
```
Don't forget to activate the virtual environment.

### 2. Copy folders
Once you create the virtual environment, copy the folders and files under `reproduction/` into your environment. 

## Validation (Section 2.2. Validation of the tool)

The required files are located under `validation/`. By running following commands, you can reproduce the validation analysis.
```python
cd validation
python ./validation.py
```

You will get the following result files under `validation/outputs/`:
- `01_cube_validation_d50.csv`
- `02_tallcube_validation_d50.csv`
- `03_widecube_validation_d50.csv` 

By default, the sampling density is set to 50, and you can modify it by replacing the value of `DENSITY` in the script (line 18). You can check that varying the sapling density does not change the computed values.




## Sampling density determination (Section 2.3. A practical demonstration of sampling density determination)

The required files are located under `sampling_density/`. By running following commands, you can reproduce the sampling density determination process.

**CAUTION:** \
This process outputs a large number of CSV files and is relatively time-consuming, compared to [Validation](#validation-section-22-validation-of-the-tool). If you want to reduce the processing time and the number of output files, uncomment line 27 of `sdd.py` and change the number right after the ":", e.g., `seeds = seeds[:20]`. It is the number of iteration of visible volume computation with different sampling seeds.


```python
cd sampling_density
python ./sdd.py
```
Once the process completed, draw a plot of the visible volume convergence.
```python
python ./plot.py
```

You will get a plot image of computed visible volumes to varying sampling densities. The image file will be saved in `sampling_density/outputs/plot.pdf`.

**Tips:** \
This procedure can be re-used for your project, by replacing input OBJ and CSV files. Put your building model (OBJ) in `sampling_density/` and replace `OBJ_FILENAME` in `objname_setting.py` with your OBJ filename. In the same manner, replace `cam_04.csv` with the location of a vantage point in your building, and modify `VPT_FILENAME` in `objname_setting.py`. In addition, you can change the range and step of sampling densities modifying `densities` in the line 19 of `sdd.py` script.


## Case study (Section 2.4. A case study)

The required files are located under `case_study/`. 

### Point analysis

You have to run the following commands.
```python 
cd case_study/
python ./point_analysis.py
```

You will get...

### Path analysis

You have to run the following commands.
```python 
cd case_study/
python ./path_analysis.py
```

You will get...

### Field analysis

You have to run the following commands.
```python 
cd case_study/
python ./field_analysis.py
```

You will get...
