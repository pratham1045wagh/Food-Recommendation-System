import pandas as pd
import numpy as np
import warnings

# Ensure pandas doesn't try to use pyarrow
pd.options.mode.copy_on_write = False
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class CustomerSegmentation:
    """
    Implements K-Means clustering for customer segmentation.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.customer_features = None
        self.cluster_labels = None
    
    def perform_clustering(self, data, n_clusters=4):
        """
        Perform K-Means clustering on customer data.
        
        Args:
            data: Cleaned order data DataFrame
            n_clusters: Number of clusters to create
            
        Returns:
            tuple: (customer_features, cluster_labels, cluster_centers)
        """
        # Extract customer features
        customer_features = self._extract_customer_features(data)
        
        # Prepare features for clustering
        feature_columns = ['Total_Spending', 'Order_Frequency', 'Avg_Basket_Size', 'Item_Variety']
        X = customer_features[feature_columns].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-Means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Store results
        self.customer_features = customer_features
        self.cluster_labels = cluster_labels
        
        # Get cluster centers (in original scale)
        cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        return customer_features, cluster_labels, cluster_centers
    
    def _extract_customer_features(self, data):
        """
        Extract features for each customer (order).
        
        Args:
            data: Order data DataFrame
            
        Returns:
            DataFrame: Customer features
        """
        try:
            # Group by order number to get customer-level features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                customer_features = data.groupby('Order Number').agg({
                    'Line_Total': 'sum',  # Total spending
                    'Item Name': ['count', 'nunique'],  # Number of items and variety
                    'Order Date': 'first'  # Order date
                }).reset_index()
            
            # Flatten column names
            customer_features.columns = ['Order_Number', 'Total_Spending', 'Basket_Size', 'Item_Variety', 'Order_Date']
            
            # Calculate order frequency (for this, we need to identify repeat customers)
            # Since we don't have customer IDs, we'll use a proxy: orders per day
            customer_features['Order_Frequency'] = 1  # Each order counts as 1
            
            # Average basket size (same as basket size for individual orders)
            customer_features['Avg_Basket_Size'] = customer_features['Basket_Size']
            
            # Select final features and ensure standard dtypes
            features = customer_features[[
                'Order_Number',
                'Total_Spending',
                'Order_Frequency',
                'Avg_Basket_Size',
                'Item_Variety'
            ]].copy()
            
            # Convert to standard numpy dtypes to avoid pyarrow
            features['Total_Spending'] = features['Total_Spending'].astype(float)
            features['Order_Frequency'] = features['Order_Frequency'].astype(int)
            features['Avg_Basket_Size'] = features['Avg_Basket_Size'].astype(float)
            features['Item_Variety'] = features['Item_Variety'].astype(int)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_cluster(self, features):
        """
        Predict cluster for new customer features.
        
        Args:
            features: Array of customer features
            
        Returns:
            int: Cluster label
        """
        if self.kmeans is None:
            raise ValueError("Model not trained. Call perform_clustering first.")
        
        # Standardize features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict cluster
        cluster = self.kmeans.predict(features_scaled)[0]
        
        return cluster
    
    def get_cluster_characteristics(self, customer_features, cluster_labels):
        """
        Get characteristics of each cluster.
        
        Args:
            customer_features: Customer features DataFrame
            cluster_labels: Array of cluster labels
            
        Returns:
            DataFrame: Cluster characteristics
        """
        # Add cluster labels to features
        features_with_clusters = customer_features.copy()
        features_with_clusters['Cluster'] = cluster_labels
        
        # Calculate statistics for each cluster
        cluster_stats = features_with_clusters.groupby('Cluster').agg({
            'Total_Spending': ['mean', 'std', 'min', 'max'],
            'Order_Frequency': ['mean', 'std'],
            'Avg_Basket_Size': ['mean', 'std'],
            'Item_Variety': ['mean', 'std'],
            'Order_Number': 'count'
        }).round(2)
        
        return cluster_stats
    
    def get_cluster_profiles(self, customer_features, cluster_labels):
        """
        Get descriptive profiles for each cluster.
        
        Args:
            customer_features: Customer features DataFrame
            cluster_labels: Array of cluster labels
            
        Returns:
            dict: Cluster profiles with descriptions
        """
        features_with_clusters = customer_features.copy()
        features_with_clusters['Cluster'] = cluster_labels
        
        profiles = {}
        
        # Calculate overall medians for comparison
        overall_medians = {
            'spending': customer_features['Total_Spending'].median(),
            'frequency': customer_features['Order_Frequency'].median(),
            'basket_size': customer_features['Avg_Basket_Size'].median(),
            'variety': customer_features['Item_Variety'].median()
        }
        
        for cluster_id in sorted(features_with_clusters['Cluster'].unique()):
            cluster_data = features_with_clusters[features_with_clusters['Cluster'] == cluster_id]
            
            avg_spending = cluster_data['Total_Spending'].mean()
            avg_frequency = cluster_data['Order_Frequency'].mean()
            avg_basket = cluster_data['Avg_Basket_Size'].mean()
            avg_variety = cluster_data['Item_Variety'].mean()
            
            # Determine cluster type based on characteristics
            if avg_spending > overall_medians['spending'] and avg_frequency > overall_medians['frequency']:
                profile_type = "High-Value Customers"
                description = "Frequent orders with high spending. These are your most valuable customers."
            elif avg_spending > overall_medians['spending']:
                profile_type = "Big Spenders"
                description = "High spending but less frequent. Focus on increasing their order frequency."
            elif avg_frequency > overall_medians['frequency']:
                profile_type = "Frequent Buyers"
                description = "Regular customers with moderate spending. Opportunity to upsell."
            else:
                profile_type = "Occasional Customers"
                description = "Lower frequency and spending. Target with promotions to increase engagement."
            
            profiles[cluster_id] = {
                'type': profile_type,
                'description': description,
                'size': len(cluster_data),
                'avg_spending': avg_spending,
                'avg_frequency': avg_frequency,
                'avg_basket_size': avg_basket,
                'avg_variety': avg_variety
            }
        
        return profiles
    
    def find_optimal_clusters(self, data, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            data: Order data DataFrame
            max_clusters: Maximum number of clusters to test
            
        Returns:
            dict: Metrics for different cluster numbers
        """
        customer_features = self._extract_customer_features(data)
        feature_columns = ['Total_Spending', 'Order_Frequency', 'Avg_Basket_Size', 'Item_Variety']
        X = customer_features[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        metrics = {
            'n_clusters': [],
            'inertia': [],
            'silhouette': []
        }
        
        for n in range(2, min(max_clusters + 1, len(X))):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            metrics['n_clusters'].append(n)
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(X_scaled, labels))
        
        return pd.DataFrame(metrics)
    
    def export_clusters(self, filepath='customer_clusters.csv'):
        """Export customer clusters to CSV."""
        if self.customer_features is not None and self.cluster_labels is not None:
            export_df = self.customer_features.copy()
            export_df['Cluster'] = self.cluster_labels
            export_df.to_csv(filepath, index=False)
            return True
        return False

