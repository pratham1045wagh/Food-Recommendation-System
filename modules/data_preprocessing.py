import pandas as pd
import numpy as np
from datetime import datetime

class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for restaurant order data.
    """
    
    def __init__(self):
        self.cleaned_data = None
        self.transactions = None
        self.item_list = None
    
    def preprocess(self, data):
        """
        Main preprocessing pipeline.
        
        Args:
            data: Raw pandas DataFrame
            
        Returns:
            tuple: (cleaned_data, transactions, item_list)
        """
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # 1. Handle missing values
        df = self._handle_missing_values(df)
        
        # 2. Remove duplicates
        df = self._remove_duplicates(df)
        
        # 3. Data type conversions
        df = self._convert_data_types(df)
        
        # 4. Data validation and cleaning
        df = self._validate_data(df)
        
        # 5. Feature engineering
        df = self._engineer_features(df)
        
        # 6. Create transaction format for association rules
        transactions = self._create_transactions(df)
        
        # 7. Get unique item list
        item_list = sorted(df['Item Name'].unique().tolist())
        
        self.cleaned_data = df
        self.transactions = transactions
        self.item_list = item_list
        
        return df, transactions, item_list
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            # Fill missing item names with 'Unknown'
            if 'Item Name' in df.columns:
                df['Item Name'] = df['Item Name'].fillna('Unknown')
            
            # Fill missing quantities with 1
            if 'Quantity' in df.columns:
                df['Quantity'] = df['Quantity'].fillna(1)
            
            # Fill missing prices with median
            if 'Product Price' in df.columns:
                df['Product Price'] = df['Product Price'].fillna(df['Product Price'].median())
            
            # Drop rows with missing order numbers or dates
            df = df.dropna(subset=['Order Number', 'Order Date'])
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate records."""
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        return df
    
    def _convert_data_types(self, df):
        """Convert columns to appropriate data types."""
        # Convert Order Number to integer
        df['Order Number'] = pd.to_numeric(df['Order Number'], errors='coerce').fillna(0).astype(int)
        
        # Convert Order Date to datetime (avoid pyarrow by using standard datetime parsing)
        try:
            df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y %H:%M', errors='coerce')
        except:
            # Fallback to automatic parsing
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        
        # Convert numeric columns
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1).astype(int)
        df['Product Price'] = pd.to_numeric(df['Product Price'], errors='coerce').fillna(0).astype(float)
        
        if 'Total products' in df.columns:
            df['Total products'] = pd.to_numeric(df['Total products'], errors='coerce').fillna(0).astype(int)
        
        return df
    
    def _validate_data(self, df):
        """Validate and clean data values."""
        # Remove records with invalid prices (negative or zero)
        df = df[df['Product Price'] > 0]
        
        # Remove records with invalid quantities
        df = df[df['Quantity'] > 0]
        
        # Clean item names (strip whitespace, title case)
        df['Item Name'] = df['Item Name'].str.strip()
        
        # Remove any records with 'Unknown' items if they exist
        df = df[df['Item Name'] != 'Unknown']
        
        return df
    
    def _engineer_features(self, df):
        """Create additional features for analysis."""
        # Total price per line item
        df['Line_Total'] = df['Quantity'] * df['Product Price']
        
        # Extract time-based features (with error handling)
        try:
            df['Year'] = df['Order Date'].dt.year
            df['Month'] = df['Order Date'].dt.month
            df['Day'] = df['Order Date'].dt.day
            df['Hour'] = df['Order Date'].dt.hour
            df['DayOfWeek'] = df['Order Date'].dt.dayofweek
            
            # Handle week extraction (avoid pyarrow issues)
            try:
                df['WeekOfYear'] = df['Order Date'].dt.isocalendar().week
            except:
                df['WeekOfYear'] = df['Order Date'].dt.week
            
            # Day name
            df['Day_Name'] = df['Order Date'].dt.day_name()
        except Exception as e:
            # If datetime features fail, set defaults
            print(f"Warning: Could not extract all datetime features: {e}")
            if 'Year' not in df.columns:
                df['Year'] = 2019
            if 'Month' not in df.columns:
                df['Month'] = 1
            if 'Day' not in df.columns:
                df['Day'] = 1
            if 'Hour' not in df.columns:
                df['Hour'] = 12
            if 'DayOfWeek' not in df.columns:
                df['DayOfWeek'] = 0
            if 'WeekOfYear' not in df.columns:
                df['WeekOfYear'] = 1
            if 'Day_Name' not in df.columns:
                df['Day_Name'] = 'Monday'
        
        return df
    
    def _create_transactions(self, df):
        """
        Create transaction format for association rule mining.
        Each transaction is a list of items in an order.
        
        Returns:
            list: List of transactions (each transaction is a list of items)
        """
        transactions = []
        
        # Group by order number
        grouped = df.groupby('Order Number')['Item Name'].apply(list).reset_index()
        
        for _, row in grouped.iterrows():
            # Get unique items in the order (in case of duplicates)
            transaction = list(set(row['Item Name']))
            transactions.append(transaction)
        
        return transactions
    
    def get_summary_statistics(self, df):
        """
        Generate summary statistics for the dataset.
        
        Returns:
            dict: Dictionary containing various statistics
        """
        stats = {
            'total_orders': df['Order Number'].nunique(),
            'total_items': df['Item Name'].nunique(),
            'total_revenue': (df['Quantity'] * df['Product Price']).sum(),
            'avg_order_value': df.groupby('Order Number')['Line_Total'].sum().mean(),
            'avg_basket_size': df.groupby('Order Number')['Item Name'].count().mean(),
            'date_range': {
                'start': df['Order Date'].min(),
                'end': df['Order Date'].max()
            },
            'most_popular_item': df.groupby('Item Name')['Quantity'].sum().idxmax(),
            'least_popular_item': df.groupby('Item Name')['Quantity'].sum().idxmin()
        }
        
        return stats
    
    def export_cleaned_data(self, filepath='cleaned_orders.csv'):
        """Export cleaned data to CSV."""
        if self.cleaned_data is not None:
            self.cleaned_data.to_csv(filepath, index=False)
            return True
        return False

