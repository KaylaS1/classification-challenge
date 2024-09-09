# Week 13 Classification Notebook

## Develop a supervised machine learning model to detect spam emails

### Step 1:
* Read in the spam data csv file and create a dataframe.
* Break out dataframe into training and testing.
* I wrote code in my pipeline_utilities Python file to scale the X_train and X_test features.

### Initial Prediction
* I think Logistic Regression will perform better because spam has more of features like spelling mistakes, poor grammar, capitals, etc. So I would think the spam versus non-spam groups or clusters could be separated by a straight line. Also dataset looks somewhat balanced and it is not huge. Spam classification decision-making is not too complex for Logistic Regression.

### Step 2: Develop and fit a Logistic Regression model
* I wrote the Logistic Regression model code in my pipeline_utilities Python file.
* I sent in the scaled data as arguments.
* The Logistic Regression model had an accuracy rate of 93.22%.

### Step 3: Develop and fit a Random Forest Classifier model
* I wrote the Random Forest model code in my pipeline_utilities Python file.
* I used the default number of 100 estimators and sent in the scaled data as arguments.
* The Random Forest CLassifier model had an accuracy rate of 95.22%.

### Conclusions:
* Both models had over 90% accuracy score, indicating that both models performed reasonably well, but the Random Forest model performed better than the Logistic Regression model by 2 percentage points. The Random Forest model achieved an accuracy of 95.22% with testing data and predictions while the Logistic Regression model achieved a score of 93.22%.
* My prediction that the Logistic Regression model would perform better was incorrect!

### Sources for my code:
* I wrote most of my code using AI Bootcamp samples provided in the weekly exercises and the starter file.
