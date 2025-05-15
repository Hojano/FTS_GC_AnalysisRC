# DRIFTS-analysis
## Description
This repository contains my your_package. Among others, the package can be used to import .dx files, logfiles and GC data from various IR setups.

## Installation 
- clone the repository:
```
git clone robins_package_link
```
- install the package by **navigating to the cloned repository** in your python environment and executing the following command:

```
pip install -r requirements.txt
pip install -e .
```
- You should now be able to load the package in python by using:

```python
import your_package as ir
```

## Installation (simplest version)
- Download the repository. 
- Copy the *your_package* folder to the folder where the script you wanna use for analysis is located. Example:

```
My_Scripts
│
└───DRIFTS
│   │   FTS.py
│   │   FTS.ipynb
│   │
│   └───*your_package*
```

## Usage
Parse DRIFTS files from a folder using:
```python
DRIFTS_spectra = ir.parse_spectra('*path-to-your-file*')
```
Make a quick plot to check: 

```python
ir.quick_plot(DRIFTS_spectra)
```
Combine with logfile:

```python
merged_data = ir.merge_spectra_logfile('*path to spectra*', '*path to logfile*')
```

Check the example notebook for more info how to use this package.

## Support
If you find bugs or have other questions send me a message or open an issue.

## Contributing
If you want to contribute, let me know.

## Authors
Robin Conradi, Utrecht University

## Acknowledgements
Package structure was adapted from pyTGA, built by Sebastian Rejman