# UC Berkeley ML/AI project: Comparing Classifiers for Bank Marketing Campaign
Overview: In this practical application, my goal is to compare the performance of the classifiers namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. I used a dataset related to marketing bank products over the telephone.
Project location: https://github.com/parthabiswas1/UCBerkeley-project-Comparing-Classifiers-for-Bank-Marketing-Campaign

## Problem Statement

Improve the efficiency of customer direct phone call marketing campaign, by using the best model to identify potential customers who are most likely to create a term deposit at the bank thereby optimizing marketing costs, improving conversion rates and saving money for the bank.

To further simplify:
I want to build a computer model that can guess who will say yes — so that we don’t waste time calling everyone and wasting money. :-)


## Source of data

Dataset comes from the UCI Machine Learning repository link. The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. The dataset represents 17 marketing campaigns conducted by a Portuguese bank between May 2008 and November 2010, comprising 79,354 contacts in total.


## Approach

### Data cleaning Plan (Feature Engineering)

The dataset is highly imbalanced.
<img width="1134" height="776" alt="image" src="https://github.com/user-attachments/assets/7425fa6f-6d44-48e3-9844-0306d83cb177" />

Properly format data into numerical and categorical
If any non numerical data appear in numerical features, replace with NaN
Convert catagorical features to category data type

Replace 'unknown' and 'nonexistent' with NaN so they can be treated as missing data rather than real categories

A value of pdays = 999 means the client was never contacted before, so replace it with NaN to avoid misleading averages

Remove data leakage - Drop the duration column because it’s only known after the marketing call — using it would give the model unfair future information.

Encode the target - Convert the target variable from text (‘yes’, ‘no’) to numbers (1, 0)

Mode impuation - For 'job', 'marital', 'housing', 'loan', 'education', 'default', fill the missing categories with the most common value (mode) so the data stays usable and balanced and we don't have to drop rows.

Is missing data actually meaningful behavior ? is there meaning in missingness ? Check if what is missing is telling us something about customer response.
New columns (pdays_missing, poutcome_missing) mark where data is missing (1 = missing, 0 = present). This helps capture patterns — e.g., clients never contacted before.
'pdays' is filled with -1 so it stays numeric but still signals “no previous contact.”
Create a new category 'no_prev_contact' for 'poutcome' for missing cases instead of deleting them

Define preprocessing for numeric and categorical columns
Scaling of numeric features (StandardScaler)
Correct categorical encoding (OneHotEncoder)
Combine transformations (ColumnTransformer) and create a Pipeline for processing the models.

### GPU CPU Hybrid approach
Model finetuning was long and did not complete in the instance of Support Vector Machine. So I took the approach of keeping LogisticRegression and DecisionTree on CPU and configured KNN and SVM to run on GPU

### Missing data
I did not want to drop missing data. rows and reduce the scope of the dataset. I wanted to understand if missing data is acutally meaningful. Is data missing becaue of a reason ?
<img width="590" height="528" alt="Screenshot 2025-11-09 at 11 46 32 PM" src="https://github.com/user-attachments/assets/77592f39-18a0-4a26-9671-be035850d086" />

Notes The negative correaltions mean that if no previous contact result (poutcome is missing) prospect is less likly to create a FD When pday is missing (never contacted before) prospect is even less liker to create FD
## Obsevations 

### Baseline Model
What is the baseline performance that my classifier should aim to beat? I took three approaches:

**Majority class approach (Most frequent)** - If I guess that everyone I call from this dataset will say 'no', then I will be correct 89% of the time and wrong 11% of the time.
(However my recall for 'yes' customers will be 0 which means with this approach, I am unable to identify which customer will say 'yes' before making the call)

**Stratified Random approach** - I will guess 'yes' for 10% of the cases and 'no' for 90% of the cases to mimic class imbalance. in some of the cases and I may get lucky and and correctly guess 'yes'.
(I will generate some precision, recall, F1 score and AUC)

**Heuristic approach** - I will use domain knowledge to make a prediction. Like if the prospect does not have housing loan and has agreed to create FD bedore, then guess 'yes'.
(It will generate some precision, recall, F1 score and AUC)

**BASELINE MODEL METRICS TO BEAT**

                          Accuracy    Precision       Recall      F1 Score      AUC
**Majority class** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.88 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;       0.0  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            0.0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        0.0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       0.5  

**Stratified Random** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.80 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;0.12  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.11 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.12 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.5

**Heuristic** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.86 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;0.27  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.14 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.19 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.547

**MODEL COMPARISON METRICS USING DEFAULT SETTINGS**

<img width="1802" height="207" alt="image" src="https://github.com/user-attachments/assets/b41e9605-9e24-41e2-8bd0-819f29fd3974" />

**MODEL COMPARISON METRICS AFTER TUNING HYPERPARAMETERS**

<img width="1814" height="199" alt="image" src="https://github.com/user-attachments/assets/1f8c8467-8f17-472a-a05c-5b23b0a453c9" />

## Recommendations
Hyperparameter tuning did not improve the models much, Decision Tree showed the best F1 Score improvement (+0.064)

**Logistic Regression** had the most balanced metrics (highest test accuracy, highest precision, good recall and highest AUC-0.80).

SVM’s long runtime and lower metrics make it inefficient for production use.
