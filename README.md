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


## Recommendations

- Create a marketing campaign targeting young adult drivers who frequently go to bars, are not economically well off, do not have kid and are employed in more urban settings.
- Make the bar coupon more attractive to low income earners by increasing the discount rate or by combining it with resturant discount offers.

## Obsevations - **Coffee House coupons**

![Coffee House coupon acceptance rates](images/coffee_coupon_acceptance.png)

- Frequent coffee house visitors (1 or more per month) accepted more coupons(64.73% - 68.24%)
- Morning coffee house visitors (10AM) have a higher coupon acceptance rate (63.43%).
- Irrespective of marital status, a high rate of acceptance(44.4% - 65.6%) among drivers who go to coffee houses in the morning (10AM).
- Highest rate of acceptance (76.9%) is by divorced drivers who go to coffee houses late in the evening (10PM)
- Drivers who have no urgent place to go in the morning (10AM) or late evening (10PM) have higher coupon acceptance rates (63.4% - 68.7%)
- Young single females divers with some college earning less than $12.5k who went to coffee houses in the afternoon accepted coffee coupons at the highest rate
- Overall **49.5%** of coffee house coupons were accepted. This shows that close to majority of the  drivers were interested in Coffee House coupons</li>

## Recommendations
- Run a targeted marketing campaign focused on young female drivers, drivers who frequented coffee houses often and drives to nowhere particular and divorcees who go to coffee houses late in the evening.
