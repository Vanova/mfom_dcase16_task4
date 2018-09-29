Multi-label MFoM framework for DCASE 2016: Task 4
-------------------------------------------------

[The SIPU Lab / University of Eastern Finland](https://www.uef.fi/web/machine-learning)

[Institute for Infocomm Research / A*Star / Singapore](https://www.a-star.edu.sg/i2r)

Author: **Ivan Kukanov**, [Email](mailto:ivan@kukanov.com), [Homepage](http://www.kukanov.com), [GitHub](https://github.com/vanova)

Supervisor: **Ville Hautamäki**, [Email](mailto:villeh@uef.fi), [Homepage](http://cs.joensuu.fi/~villeh/)
  
Description
===========
This framework represents the implementation of the [maximal figure-of-merit](http://cs.joensuu.fi/~villeh/MFoM-ICASSP2017.pdf) (MFoM) approach 
for multi-label tasks. In particular, it is applied to the [domestic audio tagging task](http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging) 
 of the DCASE 2016: task 4 challenge. This project represents the solution for the problem of the domestic audio tagging, 
where one audio recording can contain one or more acoustic events and a recognizer
should output all of those tags. 

Organizers provided the baseline system [DCASE 2016 task 4 baseline](https://github.com/pafoster/dcase2016_task4/tree/master/baseline)
with the basic approach: MFCC-based acoustic features and a GMM-based classifier.
Our baseline model is a *convolutional recurrent neural network* (CRNN)
with sigmoid output units optimized using the binary cross-entropy (BCE)
objective. We embed maximal figure-of-merit approach into the deep
learning objective function and gain more than 10% relative
improvement, compared to the baseline model with the binary
cross-entropy.

The proposed MFoM approaches are used in the series of works:

* 
* ICASSP 18
* DCASE 17
* DCASE 16
* SLT 16

Installation
============

The system is developed for [Python 2.7](https://www.python.org/). 
Currently, the baseline systems are tested only with Linux operating systems.

You can install the python environment using [Conda](https://conda.io/docs/) and the yml setting file:

`$ conda env create -f envs/conda/ai.py2.yml`

and activate the environment 

`$ source activate ai`

Usage
=====

The executable file of the project is: `experiments/run_dcase.py`

The system has two pipeline operating modes: *Development mode* and *Submission (or evaluation) mode* (TBD). 

The usage parameters are shown by executing `python run_dcase.py -h`

The system parameters are defined in `experiments/params/dcase.yaml`. 

## Development mode

In this mode the system is trained and tested within the development dataset. 
This is the default operating mode. To run the system in this mode:

`python run_dcase.py -m dev` or `run_dcase.py -m dev -p params/dcase.yaml`.

System results
==============

* Dataset: ** CHiME-Home-refine --  development set **

* Evaluation setup: 5-fold cross-validation, 7 classes.

### GMM baseline model

Baseline [github](https://github.com/pafoster/dcase2016_task4/tree/master/baseline). 

System main parameters
 
frame size: 20 ms (50% hop size), number of components: 8, 
features: MFCC 14 static coefficients (excluding 0th coefficient)

     | Tag                  | EER  |
     |----------------------|------|
     | Adult female speech  | 0.29 |
     | Adult male speech    | 0.30 | 
     | Broadband noise      | 0.09 |
     | Child speech         | 0.20 |
     | Other                | 0.29 |
     | Percussive sound     | 0.25 | 
     | Video game/tv        | 0.07 | 
     | **Mean error**       | **0.21** | 

### CRNN baseline model



### MFoM approach with CRNN model


The MFoM approaches
===================

## MFoM embedding

## MFoM-microF1 / MFoM-EER / MFoM-Cprim 


Project Organization
====================

    |── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── envs               <- Environment settings files: Anaconda, Dockerfile
    ├── experiments
    │   ├── logs
    │   ├── params         <- Training settings, hyperparameters
    │   ├── submissions    <- Evaluation model results, submission to the challenge leaderboard
    │   ├── system         <- Trained and serialized models, model predictions, or model summaries
    │   └── experiment.py  <- Main file to run the particular experiment, it is based on the framework in 'src' folder
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Manuals, literature and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Main source code for use in this project. Framework structure.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    |
    ├── test_environment.py
    |
    ├── tests              <- Test framework code from 'src' folder
    │   └── data           <- data for testing
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

--------
