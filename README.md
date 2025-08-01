# PID Ki Optimization using Metaheuristics and Ensembles

This repository contains Python implementations of several optimization algorithms applied to the tuning of the **Ki** parameter in PID controllers. The goal is to minimize the **Integral of Time-weighted Absolute Error (ITAE)**.

### Plan

### 1. Determination of Ki-ITAE for Each Method
- Grid Search
- Random Search
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Ensemble Methods (Mean, Median, Weighted)

### 2. Comparative Analysis
Ki values obtained from each model alongside ITAE, IAE, and ISE performance metrics
Comparative evaluation based on step response and error signal plots
Decision criteria encompassing accuracy, convergence speed, and consistency

### 3. Ensemble Learning

Integration of predictions from multiple models to produce a refined Ki estimate
Techniques such as weighted averaging, voting, and stacking employed for ensemble modeling

### 4. Visualization and Tabular Outputs

Generation of boxplots, radar charts, and R² tables
Analysis of Ki distributions and performance differentials
Separate and combined visual representations for all models

### Associated Publication:
This code supports the experiments described in the manuscript:

> "**Enhanced PID Control for Induction Motor Drives: Classical and Intelligent Tuning Approaches**"  
> Targeted for submission to Journal of Artificial Intelligence Research
