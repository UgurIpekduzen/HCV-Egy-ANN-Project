import numpy as np # linear algebra
from pandas import *
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization
import os
#plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# print(os.listdir("./input"))




#
# # column(feature) names in data
# print(data.columns)
# #
# # getting an overview of our data
# print(data.info())
#
# # checking for missing values
# print("Are there missing values? {}".format(data.isnull().any().any()))
# # missing value control in features
# print(data.isnull().sum())
#
# #Let's learn about the int values in our dataset.
# print(data.describe()) #include ID feature
# #we don't need istaticsal summary for ID feature
# print(data.iloc[:,1:].describe())
#
# print(data.head())
#
# #we found out how many teams in our data
# print("Team Names in Dataset:")
# print(data.Team.unique())
#
# print("\nYears in Dataset:")
# #we sorted the years  for a better look view.
# print(np.sort(data.Year.unique()))
#
# print("\nSport Types:")
# print(data.Sport.unique())