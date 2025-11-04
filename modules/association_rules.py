import pandas as pd
import warnings

# Ensure pandas doesn't try to use pyarrow
pd.options.mode.copy_on_write = False
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

class AssociationRuleMiner:
    """
    Implements Apriori algorithm for association rule mining.
    """
    
    def __init__(self):
        self.frequent_itemsets = None
        self.rules = None
    
    def generate_rules(self, transactions, min_support=0.02, min_confidence=0.3, min_lift=1.2):
        """
        Generate association rules using Apriori algorithm.
        
        Args:
            transactions: List of transactions (each transaction is a list of items)
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
            
        Returns:
            tuple: (rules_dataframe, frequent_itemsets)
        """
        try:
            # Convert transactions to one-hot encoded DataFrame
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            
            # Create DataFrame without pyarrow
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.DataFrame(te_ary, columns=te.columns_)
                # Ensure using standard dtypes, not pyarrow
                df = df.astype(bool)
            
            # Apply Apriori algorithm to find frequent itemsets
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                return pd.DataFrame(), pd.DataFrame()
            
            self.frequent_itemsets = frequent_itemsets
            
            # Generate association rules
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=min_confidence
                )
                
                # Filter by lift
                rules = rules[rules['lift'] >= min_lift]
                
                # Sort by lift (descending)
                rules = rules.sort_values('lift', ascending=False)
                
                # Format antecedents and consequents for better readability
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                # Select relevant columns and ensure standard dtypes
                rules_df = rules[[
                    'antecedents',
                    'consequents',
                    'support',
                    'confidence',
                    'lift'
                ]].reset_index(drop=True)
                
                # Convert to standard dtypes to avoid pyarrow issues
                rules_df = rules_df.copy()
                rules_df['support'] = rules_df['support'].astype(float)
                rules_df['confidence'] = rules_df['confidence'].astype(float)
                rules_df['lift'] = rules_df['lift'].astype(float)
            
            self.rules = rules_df
            
            return rules_df, frequent_itemsets
            
        except ImportError as ie:
            if 'pyarrow' in str(ie):
                print(f"PyArrow not available, but continuing with workaround: {ie}")
                # Retry with explicit copy to avoid pyarrow
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df = pd.DataFrame(te_ary.copy(), columns=list(te.columns_))
                frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
                
                if len(frequent_itemsets) == 0:
                    return pd.DataFrame(), pd.DataFrame()
                
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                rules = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False)
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                rules_df = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].reset_index(drop=True)
                return rules_df, frequent_itemsets
            else:
                raise ie
        except Exception as e:
            print(f"Error generating rules: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), frequent_itemsets
    
    def get_rules_for_item(self, item_name, rules_df=None):
        """
        Get all rules where the specified item appears in antecedents.
        
        Args:
            item_name: Name of the item
            rules_df: Rules DataFrame (uses self.rules if None)
            
        Returns:
            DataFrame: Filtered rules
        """
        if rules_df is None:
            rules_df = self.rules
        
        if rules_df is None or len(rules_df) == 0:
            return pd.DataFrame()
        
        # Filter rules where item appears in antecedents
        filtered_rules = rules_df[rules_df['antecedents'].str.contains(item_name, case=False)]
        
        return filtered_rules
    
    def get_recommendations_from_rules(self, selected_items, rules_df=None, top_n=5):
        """
        Get item recommendations based on selected items and association rules.
        
        Args:
            selected_items: List of currently selected items
            rules_df: Rules DataFrame (uses self.rules if None)
            top_n: Number of recommendations to return
            
        Returns:
            list: List of tuples (item, confidence_score)
        """
        if rules_df is None:
            rules_df = self.rules
        
        if rules_df is None or len(rules_df) == 0:
            return []
        
        recommendations = {}
        
        # For each selected item, find rules where it appears in antecedents
        for item in selected_items:
            item_rules = self.get_rules_for_item(item, rules_df)
            
            # Extract consequents and their confidence scores
            for _, rule in item_rules.iterrows():
                consequents = rule['consequents'].split(', ')
                confidence = rule['confidence']
                
                for consequent in consequents:
                    # Don't recommend items already selected
                    if consequent not in selected_items:
                        # Use maximum confidence if item appears in multiple rules
                        if consequent not in recommendations or recommendations[consequent] < confidence:
                            recommendations[consequent] = confidence
        
        # Sort by confidence and return top N
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return sorted_recommendations
    
    def get_frequent_itemsets_summary(self):
        """
        Get summary statistics of frequent itemsets.
        
        Returns:
            dict: Summary statistics
        """
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            return {}
        
        summary = {
            'total_itemsets': len(self.frequent_itemsets),
            'avg_support': self.frequent_itemsets['support'].mean(),
            'max_support': self.frequent_itemsets['support'].max(),
            'min_support': self.frequent_itemsets['support'].min(),
            'itemset_sizes': self.frequent_itemsets['itemsets'].apply(len).value_counts().to_dict()
        }
        
        return summary
    
    def get_rules_summary(self):
        """
        Get summary statistics of association rules.
        
        Returns:
            dict: Summary statistics
        """
        if self.rules is None or len(self.rules) == 0:
            return {}
        
        summary = {
            'total_rules': len(self.rules),
            'avg_confidence': self.rules['confidence'].mean(),
            'avg_lift': self.rules['lift'].mean(),
            'max_lift': self.rules['lift'].max(),
            'min_lift': self.rules['lift'].min()
        }
        
        return summary
    
    def export_rules(self, filepath='association_rules.csv'):
        """Export association rules to CSV."""
        if self.rules is not None and len(self.rules) > 0:
            self.rules.to_csv(filepath, index=False)
            return True
        return False

