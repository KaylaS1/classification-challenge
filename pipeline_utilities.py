from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def logistic_regression_model_generator(X_train, X_test, y_train, y_test, r_state):
    """
    Kala document this function.
    """
    model = LogisticRegression(random_state=r_state)
    model.fit(X_train, y_train)
    print(f"Logistic Regression Training Data Score: {model.score(X_train, y_train)}")
    print(f"Logistic Regression Testing Data Score: {model.score(X_test, y_test)}")
    predictions = model.predict(X_test)
    print(f"Logistic Regression Predictions: {predictions}")
    print(f"Logistic Regression Predictions: {accuracy_score(y_test, predictions)}")

def random_forest_model_generator(X_train, X_test, y_train, y_test, r_state, estimator_count, X_columns):
    """
    Kala document this function.
    """
    # Create the random forest classifier instance
    model = RandomForestClassifier(n_estimators=estimator_count, random_state=r_state)

    # Fit the model and print the training and testing scores
    model.fit(X_train, y_train)
    print(f"Random Forest Training Data Score: {model.score(X_train, y_train)}")
    print(f"Random Forest Testing Data Score: {model.score(X_test, y_test)}")

    # Make predictions using testing data
    predictions = model.predict(X_test)
    print(f"Random Forest Predictions: {predictions}")

    # Print the accuracy score
    print(f"Random Forest Predictions: {accuracy_score(y_test, predictions)}")
    
    # Get the feature importance array
    importances = model.feature_importances_

    # List the top 10 most important features
    importances_sorted = sorted(zip(model.feature_importances_, X_columns), reverse=True)
    importances_sorted[:10]

def split_train_test(data_df, target_col):
    """
    Splits input data into training and testing sets.
    Uses target column to create labels set `y` and removes it from features DataFrame `X`.
    Returns training and testing data sets.
    """
    y = data_df[target_col].values.reshape(-1, 1)
    X = data.copy().drop(columns=target_col)
    return train_test_split(X, y)

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n = x.shape[0]
    p = y.shape[1]
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def prod_model_generator(prod_df, target_col):
    """
    Defines a series of steps that will preprocess data,
    split data, and train a model for predicting actual productivity
    using linear regression. It will return the trained model
    and print the mean squared error, r-squared, and adjusted
    r-squared scores.
    """
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function
    steps = [("Scale", StandardScaler(with_mean=False)), 
        ("Linear Regression", LinearRegression())]

    # Create a pipeline object
    pipeline = Pipeline(steps)

    # Apply the preprocess_rent_data step
    X_train, X_test, y_train, y_test = preprocess_prod_data(prod_df, target_col)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Use the pipeline to make predictions
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2_value = r2_score(y_test, y_pred)
    r2_adj_value = r2_adj(X_test, y_test, pipeline)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2_value}")
    print(f"Adjusted R-squared: {r2_adj_value}")
    if r2_adj_value < 0.4:
        print("WARNING: LOW ADJUSTED R-SQUARED VALUE")

    # Return the trained model
    print(pipeline)
    return pipeline

def ridge_model_generator(prod_df, target_col):
    """
    Defines a series of steps that will preprocess data,
    split data, and train a model for predicting actual productivity
    using linear regression. It will return the trained model
    and print the mean squared error, r-squared, and adjusted
    r-squared scores.
    """
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function
    steps = [("Scale", StandardScaler(with_mean=False)), 
             ("Ridge Linear Regression", Ridge(alpha=1))] 

    # Create a pipeline object
    pipeline = Pipeline(steps)

    # Apply the preprocess_rent_data step
    X_train, X_test, y_train, y_test = preprocess_prod_data(prod_df, target_col)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Use the pipeline to make predictions
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2_value = r2_score(y_test, y_pred)
    r2_adj_value = r2_adj(X_test, y_test, pipeline)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2_value}")
    print(f"Adjusted R-squared: {r2_adj_value}")
    if r2_adj_value < 0.4:
        print("WARNING: LOW ADJUSTED R-SQUARED VALUE")

    # Return the trained model
    print(pipeline)
    return pipeline

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")

    




