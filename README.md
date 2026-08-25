# Flight Delay Prediction Modeling

## Project Overview & Dataset
This project analyzes airline flight records to develop machine learning models capable of predicting substantial delays (over 15 minutes). The approach systematically compares various predictive algorithms to identify the most effective model for predicting delays in advance. The dataset contains over 30,000 records and approximately 25 variables covering aircraft technical aspects, flight frequency, weather conditions, and airport traffic. 
* Note: Variables such as `distance_group` and `carrier_name` were excluded, the data contained zero missing values, and the target distribution did not require prior balancing.

## Methodology & Data Preprocessing
The project follows a rigorous machine learning pipeline in R:
* Feature Selection: Applied Random Forest variable importance (VarImp > 20) to filter and select the most relevant predictors.
* Model Building: Trained and tuned eight distinct algorithms, including Decision Tree, LASSO, GLM, Naive Bayes, and Neural Networks.
* Assessment: Evaluated overfitting using Average Squared Error (ASE) and compared model performance via ROC curves and AUC metrics.
* Threshold Tuning: Adjusted the classification threshold to 0.01 based on sensitivity to correctly classify delayed flights and minimize missed delay predictions.

## Models & Results
* 1. Model Comparison: Naive Bayes and GLM emerged as the top models based on ROC metrics. Naive Bayes was selected as the definitive winner after evaluating Lift curves, as it correctly predicted a higher percentage of delayed flights in the top deciles.
* 2. Final Classification: When applied to the new score dataset, the winning model (Naive Bayes) achieved a high sensitivity of 0.9882, accurately classifying the positive class with only 7 misclassifications.

## Repository Structure & Authors
* `data/`: Contains the original flight dataset.
* `src/`: R script containing the data preprocessing and machine learning models.
* `docs/`: Full project report detailing the step-by-step statistical analysis and plots.

## Authors:
* Mirko Iannelli 
* Christian Pettinato
* Riccardo Scampini
