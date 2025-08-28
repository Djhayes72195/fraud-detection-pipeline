# End-to-End Fraud Detection Service


## Overview

A ML-powered fraud detection service for credit card transactions, built with an emphasis on production realism. Includes components for model training, model serving, model versioning, monitoring, and automated retraining (WIP).

## Project Goals

This project was built to explore the challenges that arise when productionizing an ML powered application. As a student, most of my time studying Machine Learning was invested in understanding the algorithms themselves. As Chip Huyen writes in *Designing Machine Learning Systems*, however, the algorithm is only a small part of a production ML system. Realizing ML solutions in practice involve answering questions like:

1. How are predictions actually delivered to the user?
2. How does the system react when the underlying data changes?
3. How will we determine ground truth and use it to gauge performance?
4. How do we make our predictions fast enough for our use case?

This project was built with the aim of simulating these challenges and offering realistic solutions.


### Architecture

The system consists of 5 services which run in parallel:

- **Transaction streaming service**: Reads transaction records from S3 and delivers them to the prediction service one record at a time.
- **Prediction service**: Receives transactions from the streaming service and uses XGBoost to classify them as fraudulent or not.
- **Label patcher service**: A lightweight component that applies ground truth labels to our input data after a specified time interval. 
- **Drift monitoring service**: Performs hypothesis tests on a rolling window of records delivered by the streaming service to detect if statistically significant feature drift has occurred. Lightweight prediction latency monitoring is also performed by this service.
- **Inference monitoring service**: Reads records that have been patched with ground truth classification and runs analysis on a rolling window to determine if model performance has significantly degraded. Currently, the service supports measurement of accuracy, precision, F1-score, ROC-AUC, and recall.

Additional key components include:

- A synthetic data generation module which builds data for our production simulation by duplicating records from a Kaggle dataset and injects random noise to simulate drift and emerging fraud signals.
- A model training pipeline, with model versioning, metadata generation, and careful test/train splitting to ensure that synthetic data points don't cause data leakage.