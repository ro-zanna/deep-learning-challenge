# deep-learning-challenge

In this assignment, we built an algorithm using machine learning and neural networks that takes data from a fictitious nonprofit company (Alphabet Soup), using a binary classfier to predict whether applicants will be successful if funded by the organization - so as to help the org select applicants for funding with the best chance of success.

The CSV contains more than 34,000 organizations with the following features and outcome:

- EIN and NAME: Identification columns
- APPLICATION_TYPE: Alphabet Soup application type
- AFFILIATION: Affiliated sector of industry
- CLASSIFICATION: Government organization classification
- USE_CASE: Use case for funding
- ORGANIZATION: Organization type
- STATUS: Active status
- INCOME_AMT: Income classification
- SPECIAL_CONSIDERATIONS: Special considerations for application
- ASK_AMT: Funding amount requested
- IS_SUCCESSFUL: Was the money used effectively

Data Preprocessing

First we took the most common application and classification types, and dropped the EIN and Name features, to trim down our data set to the most relevant data so as not to overwhelm the model. We applied functions to classify categorical datapoints into binary values, split the dataset into features and target arrays, and scaled it before running the model on it. 

Compiling, Training, and Evaluating the Model

For the model, we defined number of input features using the categorical variables. Hidden nodes of 8 and 5 for each layer. Relu was used to evaluate input data (most common activation function transforming values from 0 to infinity), and sigmoid for the output function (outcome of 0 or 1).

In this initial run, we saw an accuracy score of 73%, which was slightly below our goal of 75%, so we looked at factors in the dataset for consideration in the next trial run.

In the second trial, we added the organizations' 'Name' feature back in, without the EIN - as organizations had multiple EINs (with the highest amount being over 1,200 EINs just for one organization). That level of detail in data would have added noise to our model and possibly skewed the accuracy of prediction. Using the same model on this new dataset returned a 96% accuracy.

Other Modeling Possibilities

Considering the dataset and problem we're trying to solve, another ML model that could be used is random forests that combine multiple decision trees to handle nonlinear and high-dimensional data. Ensemble methods can improve accuracy, reduce overfitting, and provide robustness to different types of data.
