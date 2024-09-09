# Week 13 Classification Notebook

## Develop a supervised machine learning model to detect spam emails

### Step 1:
* Read in the spam data csv file and create a dataframe.
* Break out dataframe into training and testing.
* Scale the features.

### Initial Prediction
* I think Logistic Regression will perform better because spam has more of features like spelling mistakes, poor grammar, capitals, etc. So I would think the spam versus non-spam groups or clusters could be separated by a straight line. Also dataset looks somewhat balanced and it is not huge. Spam classification decision-making is not too complex for Logistic Regression.

### Step 2: Develop and fit a Logistic Regression model
* TBD

### Step 3: Develop and fit a Random Forest Classifier model
* In progress

### Conclusions:
* Both models had over 90% accuracy score, indicating that both models performed reasonably well, but the Random Forest model performed better than the Logistic Regression model. The Random Forest model achieved an accuracy of 95.65% with testing data and predictions while the Logistic Regression model achieved a score of 92.96%.
