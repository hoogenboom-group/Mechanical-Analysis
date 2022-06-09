# Mechanical-Analysis
Drift and vibration analysis of scanning electron microscope images

Created by Daan Boltje  
Made presentable with input from Ryan Lane  

### Installation
* Create a new conda environment (assumes Anaconda or Miniconda is already installed)
```
$ conda create -n manalysis -c conda-forge numpy scipy pandas matplotlib scikit-image pip ipykernel
```
pip is installed to avoid accidentally pip installing in the base environment. ipykernel is used to have this conda environment made available as Jupyter kernel. 

* Activate environment
```
$ conda activate manalysis
```

* Install directly from github repository
```
(manalysis) $ pip install git+git://github.com/hoogenboom-group/Mechanical-Analysis.git
```


### To run in a Jupyter notebook
* Assumes Jupyter Lab is installed and runs from base conda environment
```
(base) $ conda install -c conda-forge jupyterlab
(base) $ conda install -c conda-forge nb_conda_kernels
```

* Installing Jupyter Notebook extensions
```
(base) $ conda install -c conda-forge jupyter_contrib_nbextensions
```

* (Optional) environment setup for enlightened folk according to Master Lane
```
(base) $ conda install -c conda-forge nodejs=15
(base) $ pip install tqdm ipympl ipywidgets imagecodecs
(base) $ jupyter labextension install @jupyter-widgets/jupyterlab-manager
(base) $ jupyter labextension install jupyter-matplotlib
(base) $ jupyter nbextension enable --py widgetsnbextension
```

* Start jupyter lab session from conda base environment; conda environments are available as separate kernels
```
(base) $ jupyter lab
```
