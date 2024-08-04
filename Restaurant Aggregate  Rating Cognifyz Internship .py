#!/usr/bin/env python
# coding: utf-8

# ##  COGNIFYZ   INTERNSHIP  PROGRAM 

# # Level-1

# ## Data Exploration

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import folium
import warnings 
warnings.filterwarnings


# In[2]:


data=pd.read_csv("D:\\data science\\Dataset.csv")
data.head()


# In[3]:


data.columns


# ##  identify the number of rows and columns.

# In[4]:


num_rows, num_columns = data.shape

print("Number of rows:", num_rows)
print("Number of columns:", num_columns)


# ## Check for missing values in each column 

# In[5]:


missing_values = data.isnull().sum()

print("Missing values in each column:")
print(missing_values)


# In[ ]:





# In[6]:


# Data type conversion if necessary
data['Average Cost for two'] = pd.to_numeric(data['Average Cost for two'], errors='coerce')

# Analyzing the distribution of the target variable ("Aggregate rating")
plt.figure(figsize=(10, 6))
sns.histplot(data['Aggregate rating'], bins=20, kde=True)
plt.title("Distribution of Aggregate Rating")
plt.xlabel("Aggregate Rating")
plt.ylabel("Frequency")
plt.show()

# Identifying class imbalances
class_counts = data['Aggregate rating'].value_counts()
print("Class Imbalances:")
print(class_counts)


# ## Descriptive Analysis:

# ### basic statistical measures 

# In[7]:


numeric_stats = data.describe()

print("Basic Statistical Measures for Numerical Columns:")
print(numeric_stats)


# In[8]:


data.describe()


# ### distribution of categorical variables like "Country Code," "City," and "Cuisines.

# In[9]:


# Distribution of "Country Code"
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Country Code')
plt.title("Distribution of Country Code")
plt.xlabel("Country Code")
plt.ylabel("Count")
plt.show()

# Distribution of "City"
plt.figure(figsize=(14, 8))
sns.countplot(data=data, x='City', order=data['City'].value_counts().index)
plt.title("Distribution of City")
plt.xlabel("City")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

# Distribution of "Cuisines"
plt.figure(figsize=(14, 8))
sns.countplot(data=data, x='Cuisines', order=data['Cuisines'].value_counts().iloc[:10].index)
plt.title("Distribution of Cuisines (Top 10)")
plt.xlabel("Cuisines")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()



# ## Identify top cuisines & Identify cities with the highest number of restaurants

# In[10]:


top_cuisines = data['Cuisines'].value_counts().head(10)
print("Top Cuisines:")
print(top_cuisines)

# Identify cities with the highest number of restaurants
top_cities = data['City'].value_counts().head(10)
print("\nTop Cities with the Highest Number of Restaurants:")
print(top_cities)


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of seaborn
sns.set_style("whitegrid")

# Create subplots for top cuisines and cities
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Plot top cuisines
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, ax=axes[0], palette="viridis")
axes[0].set_title('Top Cuisines with the Highest Number of Restaurants')
axes[0].set_xlabel('Number of Restaurants')
axes[0].set_ylabel('Cuisine')

# Plot top cities
sns.barplot(x=top_cities.values, y=top_cities.index, ax=axes[1], palette="magma")
axes[1].set_title('Top Cities with the Highest Number of Restaurants')
axes[1].set_xlabel('Number of Restaurants')
axes[1].set_ylabel('City')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# ## Geospatial Analysis:

# ### Visualize the locations of restaurants on a map using latitude and longitude information.
# 

# In[12]:


import folium
from folium.plugins import MarkerCluster

# Create a map centered at a specific location (you can adjust the coordinates)
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=10)

# Create a MarkerCluster to add markers for each restaurant location
marker_cluster = MarkerCluster().add_to(restaurant_map)

# Add markers for each restaurant
for _, restaurant in data.iterrows():
    folium.Marker([restaurant['Latitude'], restaurant['Longitude']], 
                  popup=restaurant['Restaurant Name']).add_to(marker_cluster)

# Display the map
restaurant_map


# ###  distribution of restaurants across different cities or countries.

# In[13]:


import folium
from folium.plugins import MarkerCluster

map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=10)

# MarkerCluster object create karna
marker_cluster = MarkerCluster().add_to(restaurant_map)


for idx, row in data.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Restaurant Name']).add_to(marker_cluster)

# Map ko display karna
restaurant_map.save("restaurant_map.html") 
restaurant_map

# Analyzing restaurant distribution across different cities
city_distribution = data['City'].value_counts()
print("Distribution of Restaurants Across Different Cities:")
print(city_distribution)


# ###  correlation between the restaurant's location and its rating.

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the relationship between latitude, longitude, and rating
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data, x='Longitude', y='Latitude', hue='Aggregate rating', palette='viridis', size='Aggregate rating', sizes=(20, 200))
plt.title("Restaurant Locations and Ratings")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title='Rating')
plt.show()

# Calculate the correlation coefficient
correlation = data[['Latitude', 'Longitude', 'Aggregate rating']].corr()
print("Correlation Matrix:")
print(correlation)

# Extract correlation coefficient between latitude, longitude, and rating
rating_lat_corr = correlation.loc['Aggregate rating', 'Latitude']
rating_lon_corr = correlation.loc['Aggregate rating', 'Longitude']
print("\nCorrelation between Aggregate Rating and Latitude:", rating_lat_corr)
print("Correlation between Aggregate Rating and Longitude:", rating_lon_corr)


# # Level-2

# ### Table Booking and Online Delivery:

# ### percentage of restaurants that offer table booking and online delivery.

# In[15]:


# Convert boolean values to numerical representation
data['Has Table booking'] = data['Has Table booking'].replace({'Yes': 1, 'No': 0})
data['Has Online delivery'] = data['Has Online delivery'].replace({'Yes': 1, 'No': 0})

# Calculate the percentage of restaurants offering table booking
table_booking_percentage = (data['Has Table booking'].mean() * 100)
print("Percentage of Restaurants Offering Table Booking:", table_booking_percentage)

# Calculate the percentage of restaurants offering online delivery
online_delivery_percentage = (data['Has Online delivery'].mean() * 100)
print("Percentage of Restaurants Offering Online Delivery:", online_delivery_percentage)


# ###  average ratings of restaurants with table booking and those without.

# In[16]:


# Calculate the average ratings for restaurants with and without table booking
avg_rating_with_booking = data[data['Has Table booking'] == 1]['Aggregate rating'].mean()
avg_rating_without_booking = data[data['Has Table booking'] == 0]['Aggregate rating'].mean()

print("Average rating for restaurants with table booking:", avg_rating_with_booking)
print("Average rating for restaurants without table booking:", avg_rating_without_booking)


# ### availability of online delivery among restaurants with different price ranges.

# In[17]:


# Convert 'Has Online delivery' column to boolean values
data['Has Online delivery'] = data['Has Online delivery'].astype(bool)

# Group the data by price range and calculate the percentage of restaurants offering online delivery
online_delivery_percentage_by_price_range = data.groupby('Price range')['Has Online delivery'].mean() * 100

print("Availability of Online Delivery Among Restaurants with Different Price Ranges:")
print(online_delivery_percentage_by_price_range)


# ## Price Range Analysis:

# ### most common price range among all the restaurants.

# In[18]:


# Determine the most common price range
most_common_price_range = data['Price range'].mode().values[0]

print("The most common price range among all the restaurants is:", most_common_price_range)


# ###  average rating for each price range.

# In[19]:


# Calculate the average rating for each price range
average_rating_by_price_range = data.groupby('Price range')['Aggregate rating'].mean()

print("Average Rating for Each Price Range:")
print(average_rating_by_price_range)


# ### color that represents the highest average rating among different price ranges.

# In[20]:


# Calculate the average rating for each price range
average_rating_by_price_range = data.groupby('Price range')['Aggregate rating'].mean()

# Find the price range with the highest average rating
highest_avg_rating_price_range = average_rating_by_price_range.idxmax()

# Find the color associated with the highest average rating price range
color_of_highest_avg_rating = data[data['Price range'] == highest_avg_rating_price_range]['Rating color'].iloc[0]

print("Color representing the highest average rating among different price ranges:", color_of_highest_avg_rating)


# ## Feature Engineering:

# In[21]:


# Extract additional features: length of restaurant name and address
data['Restaurant Name Length'] = data['Restaurant Name'].apply(lambda x: len(str(x)))
data['Address Length'] = data['Address'].apply(lambda x: len(str(x)))


# In[22]:


# Perform one-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=['Has Table booking', 'Has Online delivery'], drop_first=True)


# # Level-3

# ##  Task-2 : Customer Preference Analysis¶

# ### relationship between the type of cuisine and the restaurant's rating.¶

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
# Group the data by cuisine and calculate the average rating for each cuisine
avg_rating_by_cuisine = data.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)

# Visualize the relationship between cuisine and rating
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_rating_by_cuisine.head(20).index, y=avg_rating_by_cuisine.head(20), palette="mako")
plt.title("Average Rating by Cuisine (Top 20)")
plt.xlabel("Cuisine")
plt.ylabel("Average Rating")
plt.xticks(rotation=90)
plt.show()


# ## most popular cuisines among customers based on the number of votes.¶

# In[24]:


# Group the data by cuisine and calculate the total number of votes for each cuisine
total_votes_by_cuisine = data.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False)

# Display the top 10 most popular cuisines based on the total number of votes
print("Top 10 Most Popular Cuisines based on Total Number of Votes:")
print(total_votes_by_cuisine.head(10))


# ## cuisines that tend to receive higher ratings.

# In[25]:


avg_rating_by_cuisine = data.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)

# Display the top 10 cuisines with the highest average ratings
print("Top 10 Cuisines with the Highest Average Ratings:")
print(avg_rating_by_cuisine.head(10))


# ## Task-3: Data Visualization

# ### visualizations to represent the distribution of ratings using different charts

# ### Histogram

# In[26]:


import matplotlib.pyplot as plt

# Plot a histogram to show the distribution of ratings
plt.figure(figsize=(10, 6))
plt.hist(data['Aggregate rating'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# ### Bar plot

# In[27]:


# Plot a bar plot to show the frequency of each rating
plt.figure(figsize=(10, 6))
data['Aggregate rating'].value_counts().sort_index().plot(kind='bar', color='green', edgecolor='black')
plt.title('Frequency of Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# ## average ratings of different cuisines or cities using appropriate visualizations.

# In[28]:


# Group the data by city and calculate the average rating for each city
avg_rating_by_city = data.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)

# Plot a bar plot to compare the average ratings of different cities
plt.figure(figsize=(12, 6))
avg_rating_by_city.head(20).plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Average Rating by City (Top 20)')
plt.xlabel('City')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()


# ## Relationship between various features and the target variable to gain insights.

# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot for 'Average Cost for two' vs. 'Aggregate rating'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Cost for two', y='Aggregate rating', data=data)
plt.title('Scatter Plot: Average Cost for Two vs. Aggregate Rating')
plt.xlabel('Average Cost for Two')
plt.ylabel('Aggregate Rating')
plt.show()

# Box plot for 'Votes' vs. 'Aggregate rating'
plt.figure(figsize=(10, 6))
sns.boxplot(x='Votes', y='Aggregate rating', data=data)
plt.title('Box Plot: Votes vs. Aggregate Rating')
plt.xlabel('Votes')
plt.ylabel('Aggregate Rating')
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create a decision tree regressor model
tree_model = DecisionTreeRegressor(random_state=42)

# Fit the model using the training data
tree_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_tree = tree_model.predict(X_test)

# Evaluate the model
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("Mean Squared Error (MSE):", mse_tree)
print("R-squared (R2) Score:", r2_tree)



# ## The project involves analyzing a dataset related to restaurants, focusing on various factors such as cuisines, location, average cost, and ratings. The dataset likely contains information about restaurants including their names, locations (city, locality), types of cuisines they offer, average cost for two people, whether they offer table booking and online delivery, aggregate rating, and other related information.
# 
# ## The objective of the project is to gain insights into the factors that influence the ratings of restaurants. This involves several tasks including data preprocessing, exploratory data analysis, feature engineering, and building predictive models. Some specific tasks may include:
# 
# ## Data Cleaning and Preprocessing: Handling missing values, converting data types, and encoding categorical variables.
# ## Exploratory Data Analysis (EDA): Analyzing the distribution of ratings, exploring the relationship between features and ratings, identifying trends and patterns.
# ## Feature Engineering: Creating new features from existing ones, extracting additional information, and selecting relevant features for modeling.
# ## Building Predictive Models: Training regression models to predict the aggregate rating of restaurants based on available features, evaluating model performance, and comparing different algorithms.
# ## Visualization: Creating visualizations such as histograms, bar plots, scatter plots, and pair plots to gain insights and communicate findings effectively.
# ## Overall, the project aims to provide actionable insights for stakeholders such as restaurant owners, food delivery platforms, and customers, helping them understand factors contributing to restaurant ratings and potentially improving their offerings or services based on these insights.
# 

# In[ ]:




