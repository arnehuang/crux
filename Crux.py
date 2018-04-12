
# coding: utf-8

# **Crux Python Assesment**

# In[1]:


import requests
import pandas as pd
from pathlib import Path


# In[2]:


data = Path("data/data.csv")
mappings = Path("data/maping.csv")
if not data.is_file():
    source = "https://storage.googleapis.com/coderpad/"
    datafile = "data.csv"
    response = requests.get(source + datafile)
    print("Downloading and Saving " + datafile)
    with open(data, 'wb') as f:
        f.write(response.content)
if not mappings.is_file():
    mapfile = "maping.csv"
    response = requests.get(source + mapfile)
    print("Downloading and Saving " + mapfile)
    with open(mappings, 'wb') as f:
        f.write(response.content)


# In[3]:


dfData = pd.read_csv(data, sep=",")


# In[4]:


dfData.head(5)


# *Investigating Insight Score*

# In[5]:


any(dfData.Insight_Score.isnull())


# In[68]:


dfData.groupby(['Period']).count().head(5)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
dfData.hist(column='Insight_Score', bins=50)


# In[8]:


dfData.groupby(['Industry']).mean()


# In[9]:


dfData.plot(x='Period', y='Insight_Score')


# Not really sure what Insight_Score is supposed to represent. A credit/performance rating of some kind? 
# 
# Insight_Score is a float with high variance. No NA's.
# 
# Filter by removing outliers?

# In[10]:


sd = dfData.Insight_Score.std()
mean = dfData.Insight_Score.mean()
sd, mean


# In[11]:


dfData_filtered = dfData[(dfData.Insight_Score < mean+ 3*sd) & (dfData.Insight_Score > mean-3*sd)]


# In[12]:


dfData_filtered.plot(x='Period', y='Insight_Score')


# *Looking into Mappings*

# In[13]:


dfMap = pd.read_csv(mappings, sep=",")


# In[14]:


dfMap.head(5)


# Inner join (would need more business context to understand if left join makes more sense, etc)

# In[15]:


dfMap.rename(columns={'ProductCode': 'Product_Code'}, inplace=True)

dfmerged = pd.merge(dfData_filtered, dfMap, on='Product_Code')


# In[16]:


dfmerged.head(5)


# *Analysis*

# Simple outlier detection might be using standard deviation, so anything above or below 3 standard deviations could be flagged, aggregated by Industry, Country, Period, etc depending on the business context. For example,

# In[17]:


mat_prices = dfmerged[dfmerged.Industry == 'MATERIALS'].Price
mat_sd = mat_prices.std()
mat_avg = mat_prices.mean()
outliers_materials_prices = dfmerged[(dfmerged.Industry == 'MATERIALS') & 
                    ((dfmerged.Price > mat_avg + 3*mat_sd) | (dfmerged.Price < mat_avg - 3*mat_sd))]


# In[53]:


outliers_materials_prices.head(5)


# Since period is a timeseries, could individually analyze by product codes and use, for example, an ARIMA model to detect outliers. To save time, I'll steal some outlier code from https://www.datascience.com/blog/python-anomaly-detection, refactored for python3

# In[42]:


from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
style.use('fivethirtyeight')

def moving_average(data, window_size):
    """ Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
    -----
            data (pandas.Series): independent variable
            window_size (int): rolling window size

    Returns:
    --------
            ndarray of linear convolution

    References:
    ------------
    [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def explain_anomalies(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using stationary standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies

    """
    avg = moving_average(y, window_size).tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for
                                                       index, y_i, avg_i in zip(count(), y, avg)
              if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}


def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using rolling standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                  testing_std_as_df.ix[window_size - 1]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    return {'stationary standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i)
                                                       for index, y_i, avg_i, rs_i in zip(count(),
                                                                                           y, avg_list, rolling_std)
              if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}


# This function is repsonsible for displaying how the function performs on the given dataset.
def plot_results(x, y, window_size, sigma_value=1,
                 text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
    """ Helps in generating the plot and flagging the anamolies.
        Supports both moving and stationary standard deviation. Use the 'applying_rolling_std' to switch
        between the two.
    Args:
    -----
        x (pandas.Series): dependent variable
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma_value (int): value for standard deviation
        text_xlabel (str): label for annotating the X Axis
        text_ylabel (str): label for annotatin the Y Axis
        applying_rolling_std (boolean): True/False for using rolling vs stationary standard deviation
    """
    plt.figure(figsize=(15, 8))
    plt.plot(x, y, "k.")
    y_av = moving_average(y, window_size)
    plt.plot(x, y_av, color='green')
    plt.xlim(0, 60)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)

    # Query for the anomalies and plot the same
    events = {}
    if applying_rolling_std:
        events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
    else:
        events = explain_anomalies(y, window_size=window_size, sigma=sigma_value)

    x_anomaly = np.fromiter(events['anomalies_dict'], dtype=int, count=len(events['anomalies_dict']))
    y_anomaly = np.fromiter(events['anomalies_dict'].values(), dtype=float,
                                            count=len(events['anomalies_dict']))
    print(x_anomaly, y_anomaly)
    plt.plot(x_anomaly, y_anomaly, "r*", markersize=5)

    # add grid and lines and enable the plot
    plt.grid(True)
    plt.show()


# In[54]:


dfOutliers = dfmerged.sort_values(by='Period')
x = dfOutliers[(dfOutliers.Industry == 'MATERIALS') & (dfOutliers.Product_Code == 'US20.3')].Period
Y = dfOutliers[(dfOutliers.Industry == 'MATERIALS') & (dfOutliers.Product_Code == 'US20.3')].Price


# In[67]:


plot_results(x, y=Y, window_size=10, text_xlabel="Period", sigma_value=1.5,
             text_ylabel="Price")
events = explain_anomalies(Y, window_size=10, sigma=2)


# Above red points show anomalies based on moving average std. Play around with different product codes, window sizes, sigmas, etc.

# Show date (period) of highest and lowest Insight_Score per Industry.

# In[22]:


idx_max = dfOutliers.groupby(['Industry'])['Insight_Score'].transform(max) == dfOutliers['Insight_Score']
idx_min = dfOutliers.groupby(['Industry'])['Insight_Score'].transform(min) == dfOutliers['Insight_Score']

highest = dfOutliers[idx_max][['Industry', 'Period', 'Insight_Score']].sort_values('Industry')
lowest = dfOutliers[idx_min][['Industry', 'Period', 'Insight_Score']].sort_values('Industry')
highest, lowest


# Which industries are the most correlated?

# In[23]:


dfmerged.Industry.unique()


# In[25]:


corr = []
industries = dfmerged.Industry.unique()
dfCorr = dfmerged.groupby(['Period', 'Industry'], as_index=False).mean()
# print(dfCorr)
finaldf = None
counts = 1
for acolumn in dfCorr.columns:
    if(dfCorr[acolumn].dtype == np.number):
        corrcol = dfCorr.pivot(index = 'Period', columns = 'Industry', values = acolumn)
        if finaldf is not None:
            finaldf += corrcol.corr().fillna(value=0)
            counts +=1
        else:
            finaldf = corrcol.corr().fillna(value=0)

import seaborn
finaldf = finaldf/counts
finaldf['ENERGY']['ENERGY'] = 1
seaborn.heatmap(finaldf, annot=True)
plt.show()


# Highest correlation between Information Technology and Industrials, Energy and Materials.
