
# coding: utf-8

# # Crux Python Assesment

# In[2]:


import requests
import pandas as pd
from pathlib import Path


# In[3]:


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


# In[5]:


dfData = pd.read_csv(data, sep=",")
dfData.head(2)


# # Investigating Insight Score

# In[5]:


any(dfData.Insight_Score.isnull())


# In[7]:


dfData.groupby(['Period']).count().head(1)


# In[8]:


dfData.groupby(['Industry']).mean()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
dfData.hist(column='Insight_Score', bins=50)


# In[10]:


dfData.plot(x='Period', y='Insight_Score')


# Being given no business context, cannot conclude what Insight_Score is supposed to represent. A credit/performance rating of some kind? As for typing, Insight_Score is a float with high variance. No NA's.
# 
# 

# Filter Insight_Score by removing outliers

# In[12]:


sd = dfData.Insight_Score.std()
mean = dfData.Insight_Score.mean()
sd, mean


# In[13]:


dfData_filtered = dfData[(dfData.Insight_Score < mean+ 3*sd) & (dfData.Insight_Score > mean-3*sd)]


# In[14]:


dfData_filtered.plot(x='Period', y='Insight_Score')


# The resulting dataframe does not have such drastic spikes as before

# # Looking into Mappings

# In[15]:


dfMap = pd.read_csv(mappings, sep=",")


# In[16]:


dfMap.head(5)


# Below I inner join to map the columns. More business context may guide decisions such as if left joining makes sense, etc

# In[17]:


dfMap.rename(columns={'ProductCode': 'Product_Code'}, inplace=True)

dfmerged = pd.merge(dfData_filtered, dfMap, on='Product_Code')


# In[18]:


dfmerged.head(5)


# # Analysis

# Simple outlier detection might involve using static standard deviation. For Example, any Price above or below 3 standard deviations could be flagged (aggregated by Industry), as shown by the list of outliers below

# In[22]:


mat_prices = dfmerged[dfmerged.Industry == 'MATERIALS'].Price
mat_sd = mat_prices.std()
mat_avg = mat_prices.mean()
outliers_materials_prices = dfmerged[(dfmerged.Industry == 'MATERIALS') & 
                    ((dfmerged.Price > mat_avg + 3*mat_sd) | (dfmerged.Price < mat_avg - 3*mat_sd))]
outliers_materials_prices.head(5)


# However, since period is a timeseries, it would make more sense to individually analyze by product codes and use, for example, an ARIMA model to detect outliers. To save time, I'll steal some outlier code from https://www.datascience.com/blog/python-anomaly-detection, refactored for python3

# In[26]:


from stolen_code import plot_results, explain_anomalies
dfOutliers = dfmerged.sort_values(by='Period')
x = dfOutliers[(dfOutliers.Industry == 'MATERIALS') & (dfOutliers.Product_Code == 'US20.3')].Period
Y = dfOutliers[(dfOutliers.Industry == 'MATERIALS') & (dfOutliers.Product_Code == 'US20.3')].Price


# In[27]:


plot_results(x, y=Y, window_size=10, text_xlabel="Period", sigma_value=1.5,
             text_ylabel="Price")
events = explain_anomalies(Y, window_size=10, sigma=2)


# The above red points show anomalies based on moving average std. Can adjust for window sizes, sigmas, etc.

#  

# *Show date (period) of highest and lowest Insight_Score per Industry.*

# In[22]:


idx_max = dfOutliers.groupby(['Industry'])['Insight_Score'].transform(max) == dfOutliers['Insight_Score']
idx_min = dfOutliers.groupby(['Industry'])['Insight_Score'].transform(min) == dfOutliers['Insight_Score']

highest = dfOutliers[idx_max][['Industry', 'Period', 'Insight_Score']].sort_values('Industry')
lowest = dfOutliers[idx_min][['Industry', 'Period', 'Insight_Score']].sort_values('Industry')
highest, lowest


# *Which industries are the most correlated?*

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
