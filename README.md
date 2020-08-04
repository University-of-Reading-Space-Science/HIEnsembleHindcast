# HIEnsembleHindcast - Ensemble CME modelling constrained by heliospheric imager observations.

## Introduction
This repository provides the analysis code and data used to investigate how Heliospheric Imager data can be used to constrain an ensemble hindcasts of CMEs using the HUXt model. This study is accepted for publication in AGU Advances. 

## Installation
This project is written in Python 3.7.3, and the specific dependencies are listed in the ``requirements.txt`` and ``environment.yml`` files. Currently ``moviepy v1.0.1`` is not available on ``conda``, but can be downloaded from ``pip``. Additionally, to make animations, ``moviepy`` requires ``ffmpeg`` to be installed. This project also requires our [``stereo_spice``](https://github.com/LukeBarnard/stereo_spice) package, which is not currently available through ``conda`` or ``pip``. 

After cloning or downloading ``HIEnsembleHindcast``, users should update [``code/config.dat``](code/config.dat) so that ``root`` points to the local directory where it is installed.

The simplest way to work with ``HIEnsembleHindcast`` in ``conda`` is to create its own environment. With the anaconda prompt, in the root directory of ``HIEnsembleHindcast``, this can be done as:
```
>>conda env create -f environment.yml
>>conda activate ensemblehindcast
>> pip install https://github.com/LukeBarnard/stereo_spice/archive/master.zip
``` 
Then the study can be reproduced by running the ``ensemble_analysis.ipynb`` notebook
```
>>jupyter lab code/ensemble_analysis.ipynb
```
The complete study should be expected to take up to 4 hours to run on a standard laptop. The production of the ensemble members and tracking the elongation of the CME flanks takes most of this time. After these files are produced, the remaining analysis is much quicker. 

## Contact
Please contact [Luke Barnard](https://github.com/lukebarnard). 

## Citation
Our article based on this analysis (Barnard et al. 2020) has been accepted to pubilcation in AGU Advances.

