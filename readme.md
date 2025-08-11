# Molecule screening framework.

## A functional screening framework to accelerate the discovery of materials using data driven methods.

This project is the result of a master thesis with the research purpose of accelerating the discovery of Azobenzene-Derived Photoswitches using Bayesian Optimization and Machine Learning


Bayesian Optimization Framework - Setup Instructions
====================================================

To run the BO-Framework code, please set up a virtual environment using `venv`
and install the required dependencies.

Setup Steps:
------------

1. Create a virtual environment:
   > python -m venv bo_env

2. Activate the environment:
   - On Windows:
     > bo_env\Scripts\activate
   - On macOS/Linux:
     > source bo_env/bin/activate

3. Install required packages:
   > pip install -r environment_requirements.txt

You are now ready to execute the BO-Framework code!

Workspace Structure:
--------------------

This workspace contains two main folders:

1. BO-Framework  
   Contains the main notebook file where the Bayesian Optimization (BO) process can be run.
   It includes all implemented combinations and techniques that were tested during the
   development of the Master's thesis.

2. Masterthesis_Notebooks  
   Contains code used to reproduce the experiments, figures, and diagrams presented in
   Section 5 (Experiments and Results) of the thesis.