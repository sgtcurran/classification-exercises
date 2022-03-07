#%%
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
from tabulate import tabulate
from pydataset import data 
from math import sqrt 
from env import host, user, password, get_db_url
url = f'mysql+pymysql://{user}:{password}@{host}/employees'
# import tensorflow as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# imports
# numpy for vectorized operations
import numpy as np
# pandas for dataframe manipulation of tabular data
import pandas as pd
# matplotlib for visualizations
import matplotlib.pyplot as plt

# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer

# filter out warnings
import warnings
warnings.filterwarnings('ignore')

# our own acquire script:
from acquire import get_titanic_data


#%%
def prep_iris(df):
    df = df.drop(columns = ['species_id','measurement_id'])
    dummy_df = pd.get_dummies(df[['species_name']], dummy_na = False , drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis=1)
    return df.rename(columns={'species_name': 'species'})
#%%

#%%
def prep_titanic(df):
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age'])
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=[True,True])
    df = pd.concat([df, dummy_df], axis=1)
    return df.drop(columns=['sex', 'embark_town'])
#%%
# 20% test, 80% train_validate
# then of the 80% train_validate: 30% validate, 70% train. 
# train, test = train_test_split(df, train_size = 0.8, stractify=df.survived, random_state=1234)
#%%
# split on original dataframe for clarity of the imputer example:
# train, validate = train_test_split(train,train_size = 0.7, stratify=train.survived, random_state=1234)
#%%
def prep_titanic_data(df):
    '''
    takes in a dataframe of the titanic dataset as it is acquired and returns a cleaned dataframe
    arguments: df: a pandas DataFrame with the expected feature names and columns
    return: train, test, split: three dataframes with the cleaning operations performed on them
    '''
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age', 'passenger_id'])
    train, test = train_test_split(df, test_size=0.2, random_state=1349, stratify=df.survived)
    train, validate = train_test_split(train, train_size=0.7, random_state=1349, stratify=train.survived)
#     print(train.info())
#     return train, validate, test
    train, validate, test = impute_mode(train, validate, test)
    dummy_train = pd.get_dummies(train[['sex', 'embark_town']], drop_first=[True,True])
    dummy_validate = pd.get_dummies(validate[['sex', 'embark_town']], drop_first=[True,True])
    dummy_test = pd.get_dummies(test[['sex', 'embark_town']], drop_first=[True,True])
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    train = train.drop(columns=['sex', 'embark_town'])
    validate = validate.drop(columns=['sex', 'embark_town'])
    test = test.drop(columns=['sex', 'embark_town'])
    return train, validate, test
#%%
def clean_titanic_data(df):
    '''
    takes in a dataframe of the titanic dataset as it is acquired and returns a cleaned dataframe
    arguments: df: a pandas DataFrame with the expected feature names and columns
    return: clean_df: a dataframe with the cleaning operations performed on it
    '''
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age'])
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=[True,True])
    df = pd.concat([df, dummy_df], axis=1)
    return df.drop(columns=['sex', 'embark_town'])

#%%
# for titanic_data
def impute_mode(train, validate, test):
    '''
    impute mode for embark_town
    '''
    imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test
#%%
def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test
#%%
## PROBABILITY DISTRIBUTION FUNCTIONS

probability_distribution = stats._distn_infrastructure.rv_frozen

def generate_random_value(distribution: probability_distribution, size = 1):
    '''
    Return a single random value, or array of random values, using the given distribution.

    This function utilizes the rvs method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the rvs function. This function
    in no way changes the behavior of the rvs function.
    '''

    return distribution.rvs(size)
#%%
def prob_of_value_discrete(distribution: probability_distribution, value: int) -> float:
    '''
    Returns the probability that the distribution will randomly generate the given value, given
    that the distribution is a discrete distribution.

    This function utilizes the pmf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the pmf function. This function
    in no way changes the behavior of the pmf function.
    '''

    return distribution.pmf(value)
#%%
def prob_of_value_continuous(distribution: probability_distribution, value: float) -> float:
    '''
    Returns the probability that the distribution will randomly generate the given value, given
    that the distribution is a continuous distribution.

    This function utilizes the pdf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the pdf function. This function
    in no way changes the behavior of the pdf function.
    '''

    return distribution.pdf(value)
#%%
def prob_less_than_value(distribution: probability_distribution, value: float) -> float:
    '''
    Returns the probability that the distribution will randomly generate a value less than
    or equal to the given value.

    This function utilizes the cdf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the cdf function. This function
    in no way changes the behavior of the cdf function.
    '''

    return distribution.cdf(value)
#%%
def value_less_than_prob(distribution: probability_distribution, probability: float) -> float:
    '''
    Given the probability of generating a random value less than or equal to some value n,
    returns the value n.

    This function utilizes the ppf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the ppf function. This function
    in no way changes the behavior of the ppf function.
    '''

    return distribution.ppf(probability)
#%%
def prob_greater_than_value(distribution: probability_distribution, value: float) -> float:
    '''
    Returns the probability that the distribution will randomly generate a value greater than
    the given value.

    This function utilizes the sf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the sf function. This function
    in no way changes the behavior of the sf function.
    '''

    return distribution.sf(value)
#%%
def value_greater_than_prob(distribution: probability_distribution, probability: float) -> float:
    '''
    Given the probability of generating a random value greater than some value n, returns
    the value n.

    This function utilizes the isf method of the distribution object. The purpose of this
    function is simply to provide a more meaningful name to the isf function. This function
    in no way changes the behavior of the isf function.
    '''

    return distribution.isf(probability)
#%%
def evaluate_hypothesis(p: float, alpha: float = 0.05) -> None:
    '''
    Compare the p value to the established alpha value to determine if the null hypothesis
    should be rejected or not.
    '''

    if p < alpha:
        print('\nReject H0')
    else: 
        print('\nFail to Reject H0')
#%%
## CHI-SQUARED TEST FUNCTION

def chi2_test(data_for_category1, data_for_category2, alpha=.05):

    '''
    Given two subgroups from a dataset, conducts a chi-squared test for independence and outputs 
    the relevant information to the console. 

    Utilizes the method provided in the Codeup curriculum for conducting chi-squared test using
    scipy and pandas. 
    '''
    
    # create dataframe of observed values
    observed = pd.crosstab(data_for_category1, data_for_category2)
    
    # conduct test using scipy.stats.chi2_contingency() test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # round the expected values
    expected = expected.round(1)
    
    # output
    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    # evaluate the hypothesis against the established alpha value
    if p < alpha:
        print('\nReject H0')
    else: 
        print('\nFail to Reject H0')
#%%
# PEARSONR CORRELATION TEST FUNCTION

def correlation_test(data_for_category1, data_for_category2, alpha = 0.05):
    '''
    Given two subgroups from a dataset, conducts a correlation test for linear relationship and outputs 
    the relevant information to the console. 
    Utilizes the method provided in the Codeup curriculum for conducting correlation test using
    scipy and pandas. 
    '''

    # conduct test using scipy.stats.peasonr() test
    r, p = stats.pearsonr(data_for_category1, data_for_category2)

    # output
    print(f'r = {r:.4f}')
    print(f'p = {p:.4f}')

    # evaluate the hypothesis against the established alpha value
    evaluate_hypothesis(p, alpha)
#%%
def gimmeThemStats(dFrame):
    """
    Description
    ----
    Outputs the general statistical description of the dataframe,
    outputs the correlation heatmap, and outputs a distribution plot.
    
    Parameters
    ----
    dFrame(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    Nothing.
    
    """
    # Description
    print("Descriptive Stats:")
    display(dFrame.describe().T)
    
    # Heatmap
    plt.figure(figsize=(10, 8)) 
    plt.title("Heatmap", fontsize = 'x-large')
    sns.heatmap(dFrame.corr(), annot =True)
    
    # Distribution
    ### NOTE: I changed histplot to distplot
    fig, axes = plt.subplots(4, 2, figsize=(14,14))
    fig.suptitle("Distribution Plot", y=0.92, fontsize='x-large')
    fig.tight_layout(pad=4.0)

    for i,j in enumerate(ndf.columns[:-1]):
        sns.distplot(dFrame[j], ax=axes[i//2, i%2])
#%%
