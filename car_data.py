#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install fastf1


# In[2]:


import fastf1


# In[3]:


session = fastf1.get_session(2019, 'Monza', 'R')
session.load()
session.results


# In[4]:


session = fastf1.get_session(2019, 'Monza','R')
session.load(telemetry = True)
#telemetry_data = session.get_telemetry()
car_data = session.car_data
#print(telemetry_data)
print(car_data)


# In[5]:


import pandas as pd


summary_data = []


for driver, data in car_data.items():
    
    avg_speed = data['Speed'].mean()
    avg_RPM = data['RPM'].mean()
    avg_throttle = data['Throttle'].mean()
    prop_time_braking = data['Brake'].mean()  # Brake is a binary flag
    
    
    summary_data.append({
        'Driver': driver,
        'Average Speed': avg_speed,
        'Average RPM': avg_RPM,
        'Average Throttle': avg_throttle,
        'Proportion of Time Braking': prop_time_braking
    })


summary_df = pd.DataFrame(summary_data)

print(summary_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


bottas_laps = session.laps.pick_driver('BOT')
bottas_car = bottas_laps.get_car_data()
print(bottas_car)


# In[7]:


driver_code = 'BOT'  # Replace with the driver code you're interested in
driver_laps = session.laps.pick_driver(driver_code)

lapwise_data = []  # To store processed data for each lap

# If driver_laps is a DataFrame
if isinstance(driver_laps, pd.DataFrame):
    for index, row in driver_laps.iterrows():
        lap_data = row  # Here row will contain all the lap data for that specific lap
        
        # Your actual code to process and analyze the lap_data goes here
        # For example, you could simply append the row to lapwise_data for now
        lapwise_data.append(lap_data)

# Convert to a DataFrame for easier manipulation later
lapwise_df = pd.DataFrame(lapwise_data)
lapwise_df


# In[8]:


driver_laps = session.laps.pick_driver('BOT')

driver_lapwise_car_data = {}


for lap_number, lap_data in driver_laps.iterrows():
    # Get car telemetry data for each lap
    lap_car_data = lap_data.get_car_data()
   
    driver_lapwise_car_data[lap_number] = lap_car_data  # Or average_data

# car data for each lap for Bottas
driver_lapwise_car_data


# In[9]:


driver_lapwise_summary = {}


for lap_number, lap_data in driver_laps.iterrows():
  
    lap_car_data = lap_data.get_car_data()
    
    # Calculate the average for each variable 
    average_speed = lap_car_data['Speed'].mean()
    average_rpm = lap_car_data['RPM'].mean()
    average_throttle = lap_car_data['Throttle'].mean()
    average_brake = lap_car_data['Brake'].mean()
    
    # Store the summarized data in the dictionary
    driver_lapwise_summary[lap_number] = {
        'Average Speed': average_speed,
        'Average RPM': average_rpm,
        'Average Throttle': average_throttle,
        'Average Brake': average_brake,
    }


summary_df = pd.DataFrame(driver_lapwise_summary)
#Transpose
summary_df = summary_df.T

# Reset the index
summary_df.reset_index(inplace=True)

# Rename the columns
summary_df.columns = ['Lap', 'Average Speed', 'Average RPM', 'Average Throttle', 'Average Brake']

summary_df['Driver'] = 77

# Reorder the columns
summary_df = summary_df[['Driver', 'Lap', 'Average Speed', 'Average RPM', 'Average Throttle', 'Average Brake']]

summary_df


# In[ ]:





# In[10]:


final_summary_df = pd.DataFrame()

all_driver_codes = ['HAM', 'BOT', 'VER', 'LEC', 'RIC']  # Add more driver codes here

for driver_code in all_driver_codes:
    driver_laps = session.laps.pick_driver(driver_code)

    driver_lapwise_summary = {}

    for lap_number, lap_data in driver_laps.iterrows():
        lap_car_data = lap_data.get_car_data()
        
        average_speed = lap_car_data['Speed'].mean()
        average_rpm = lap_car_data['RPM'].mean()
        average_throttle = lap_car_data['Throttle'].mean()
        average_brake = lap_car_data['Brake'].mean()
        
        # Store the summarized data in the dictionary
        driver_lapwise_summary[lap_number] = {
            'Average Speed': average_speed,
            'Average RPM': average_rpm,
            'Average Throttle': average_throttle,
            'Average Brake': average_brake,
        }
    
    summary_df = pd.DataFrame(driver_lapwise_summary)
    summary_df = summary_df.T

    # Add driver code to the DataFrame
    summary_df['Driver'] = driver_code
    
    # Reorder the columns
    summary_df = summary_df[['Driver', 'Average Speed', 'Average RPM', 'Average Throttle', 'Average Brake']]
    
    # Append this DataFrame to the final summary DataFrame
    final_summary_df = pd.concat([final_summary_df, summary_df], ignore_index=True)

# Now, final_summary_df will contain the summarized data for all drivers
print(final_summary_df)


# In[11]:


final_summary_df.to_csv('final_summary.csv', index=False)
# so that we can see all of it


# In[15]:


from sklearn.preprocessing import StandardScaler
import pandas as pd

# Your DataFrame name is assumed to be final_summary_df

# 1. Aggregate by Driver
# This will take the mean of 'Average Speed', 'Average RPM', 'Average Throttle', and 'Average Brake' for each driver
aggregated_df = final_summary_df.groupby('Driver').mean().reset_index()

# 2. Standardize the Features
features_to_standardize = ['Average Speed', 'Average RPM', 'Average Throttle', 'Average Brake']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the features and transform them
aggregated_df[features_to_standardize] = scaler.fit_transform(aggregated_df[features_to_standardize])

# Now, aggregated_df is ready for clustering



# In[16]:


# what value of k should we use?
print("Shape of aggregated_df:", aggregated_df.shape)
print("Unique drivers:", aggregated_df['Driver'].nunique())


# In[20]:


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
inertia = []
silhouette_scores = []

# Limit k to 2, 3, and 4 due to the small number of unique drivers
for i in range(2, 5):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(aggregated_df[features_to_standardize])
    silhouette_avg = silhouette_score(aggregated_df[features_to_standardize], cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(2, 5), silhouette_scores, marker='o')
plt.title('Silhouette Analysis for Aggregated Data')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


# Since we have lap-level data for each driver 
# we could perform clustering on that instead of aggregating it to the driver level?
#this would give us more data points to work with?


# In[22]:


silhouette_scores_lap_level = []

# Test for clusters in the range 2 to 10
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(final_summary_df[features_to_standardize])
    silhouette_avg = silhouette_score(final_summary_df[features_to_standardize], cluster_labels)
    silhouette_scores_lap_level.append(silhouette_avg)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores_lap_level, marker='o')
plt.title('Silhouette Analysis for Lap-Level Data')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


# In[23]:


#the idea is to pick the number of clusters (k)
#that gives the highest silhouette score, as this represents the most well-defined clusters.
# clearly k = 2


# In[24]:


from sklearn.cluster import KMeans

# Initialize KMeans with k=3
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the model to aggregated data and predict cluster labels
cluster_labels_aggregated = kmeans.fit_predict(aggregated_df[features_to_standardize])

# Add the cluster labels to the aggregated DataFrame
aggregated_df['Cluster'] = cluster_labels_aggregated

print(aggregated_df['Cluster'])


# In[25]:


from sklearn.cluster import KMeans

# Initialize KMeans with k=2
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the model to lap-level data and predict cluster labels
cluster_labels_lap_level = kmeans.fit_predict(final_summary_df[features_to_standardize])

# Add the cluster labels to the lap-level DataFrame
final_summary_df['Cluster'] = cluster_labels_lap_level
print(final_summary_df['Cluster'])


# In[26]:


#Cluster 1: The four drivers falling into this cluster have similar 
#characteristics in terms of 'Average Speed', 'Average RPM', 'Average Throttle', and 'Average Brake'.
#Cluster 0: The driver in this cluster differs in those same aspects from those in Cluster 1.


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt

# Using pairplot to plot pairwise relationships in the dataset
sns.pairplot(aggregated_df, hue='Cluster', palette='Dark2', diag_kind='kde')

plt.show()


# In[ ]:


#Cluster 0 seems to represent a more conservative style of driving.a lot of braking and low speed.
#This could signify a conservative strategy or laps under safety car conditions.
#Cluster 1 appears to be more balanced, with drivers who adapt to different racing conditions.
#Cluster 2 seems to have aggressive drivers.High speed with moderate braking. This could indicate aggressive driving.


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

# Using pairplot to plot pairwise relationships in the dataset
sns.pairplot(final_summary_df, hue='Cluster', palette='Dark2', diag_kind='kde')

plt.show()


# In[ ]:


#The drivers in Cluster 0 seem to have a more aggressive and consistent driving style,
#possibly sticking to a set strategy. On the other hand, Cluster 1 seems more adaptive, 
#possibly reacting to race conditions Cluster 0 could potentially represent more skilled drivers 
#who can maintain high speeds while minimizing brake usage, showcasing efficient driving. 
#If there was a shifts between clusters for the same driver, it could signify a change in race conditions
#or strategy.For example, moving from Cluster 1 to Cluster 0 could indicate a shift to a more aggressive strategy.


# In[ ]:





# In[29]:


# Now we try DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Standardize the features for clustering
features_to_standardize = ['Average Speed', 'Average RPM', 'Average Throttle', 'Average Brake']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(final_summary_df[features_to_standardize])

# Initialize and fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(scaled_features)

# Add the cluster labels to the original DataFrame
final_summary_df['DBSCAN_Cluster'] = clusters


# In[30]:


print(clusters)


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

# Customize pairplot to use the 'DBSCAN_Cluster' as hue and palette 'Dark2'
# Since we have 5 features including 'Driver', you'll see 5x5 plots
# We'll use Kernel Density Estimation (kde) for diagonal plots
sns.pairplot(final_summary_df, hue='DBSCAN_Cluster', palette='Dark2', diag_kind='kde', vars=['Average Speed', 'Average RPM', 'Average Throttle', 'Average Brake'])

# Show the plot
plt.show()


# In[ ]:


#The dominance of cluster 0 suggests that most laps have similar speed profiles, while the laps in 
#cluster -1 (the outliers) deviate from this norm. This could indicate more aggressive or conservative
#driving styles that stand out from the majority.


# In[ ]:





# In[32]:


#which features are most influential in determining these clusters. 
#This will help understand what primarily differentiates the 'regular' laps from the 'outliers'.
# Generate box plots for each feature split by cluster
for feature in ['Average Speed', 'Average RPM', 'Average Throttle', 'Average Brake']:
    sns.boxplot(x='DBSCAN_Cluster', y=feature, data=final_summary_df)
    plt.title(f'Box plot of {feature} by Cluster')
    plt.show()


# In[ ]:





# In[33]:


#could compute average statistics for each cluster to understand what they signify in terms of driving patterns?
# Compute the mean for each feature in each cluster
cluster_summary = final_summary_df.groupby('DBSCAN_Cluster').mean()
print("Cluster Averages:")
print(cluster_summary)



# In[35]:


# what we make from this
# From average speed, cluster 0 represents laps where drivers are pushing for speed, whereas cluster -1 might represent
#more conservative laps or laps with incidents that reduced speed.
#Braking is less frequent in cluster 0 (~0.13) compared to cluster -1 (~0.17). This aligns with the 
#idea that cluster 0 represents more aggressive driving, as less time is spent on the brakes

#cluster 0 represents more aggressive or speed-focused driving, while cluster -1 represents more conservative
#driving or laps where incidents may have occurred


# In[34]:


# Extract rows belonging to the outlier cluster (-1)
outlier_data = final_summary_df[final_summary_df['DBSCAN_Cluster'] == -1]

# Check which drivers have the most outlier laps
print("Drivers with most outlier laps:")
print(outlier_data['Driver'].value_counts())


# In[ ]:


# HAM and VER Both have 9 outlier laps each. This could indicate either a highly aggressive or highly 
#conservative driving style that deviates from the norm.
# BOT, LEC and RIC each have 5 outlier laps. This could be indicative of specific laps where these 
#drivers adopted a different strategy or faced specific challenges.


# In[ ]:





# In[42]:


from fastf1 import api

# session object already initialized
# for the 2019 Monza race
# session = fastf1.get_session(2019, 'Monza', 'R')

# Using the path from the session object to fetch track status data
track_data = api.track_status_data(session.api_path)

# Now track_data should contain the track status information
print(track_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




