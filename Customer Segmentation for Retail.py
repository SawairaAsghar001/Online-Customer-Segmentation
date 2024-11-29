#----Data Preparation---
import pandas as pd
df = pd.read_excel('E:\Internship\Week 2\Online Retail.xlsx')
#---Data Cleaning----
#checking for null values in dataset
df.isnull().sum()
#dropping null values
df.dropna(inplace=True)
#checking for missing values
df.isnull().sum()
#----Perform RFM Analysis---
import datetime as dt

# Set reference date for Recency calculation
Latest_Date = dt.datetime(2011,12,10)
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

#Create RFM Modelling scores for each customer
rfm_df = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (Latest_Date - x.max()).days,
                                       'InvoiceNo': lambda x: len(x),
                                       'TotalAmount': lambda x: x.sum()})

#Convert Invoice Date into type int
rfm_df['InvoiceDate'] = rfm_df['InvoiceDate'].astype(int)
#Rename column names to Recency, Frequency and Monetary
rfm_df.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalAmount': 'Monetary'}, inplace=True)

rfm_df.reset_index().head()
#Descriptive Statistics (Recency)
rfm_df.Recency.describe()
#-----Normalize RFM Metrics: Use Min-Max Scaling----
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
rfm_scaled = scaler.fit_transform(rfm_df)
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])
#----Apply Clustering Algorithms----
# Importing libraries necessary for clustering
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
#---Applying Elbow Method on Recency and Monetary---
# Elbow Method
distortions = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    distortions.append(kmeans.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()
# Visually inspect the Elbow plot to determine the optimal k
# Let's assume the optimal k is 3 based on the elbow plot
optimal_k = 3  # Assign the value you identified from the Elbow method

# Now you can use optimal_k in your KMeans model
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
rfm_df['DBSCAN_Cluster'] = dbscan.fit_predict(rfm_scaled)
#----Visualization and Insights----
from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm_df['PCA1'], rfm_df['PCA2'] = rfm_pca[:, 0], rfm_pca[:, 1]

# Pass rfm_df to the 'data' argument of sns.scatterplot
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=rfm_df, palette='viridis')
plt.title('Customer Clusters')
plt.show()
#---Analyze Clusters---
cluster_summary = rfm_df.groupby('Cluster').mean()
print(cluster_summary)
