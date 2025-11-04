import pandas as pd
import numpy as np
from collections import Counter

class RecommendationEngine:
    """
    Generates food recommendations based on association rules and patterns.
    """
    
    def __init__(self):
        pass
    
    def get_recommendations(self, selected_items, rules_df, transactions, top_n=5):
        """
        Get recommendations based on selected items using association rules.
        
        Args:
            selected_items: List of currently selected items
            rules_df: Association rules DataFrame
            transactions: List of transactions
            top_n: Number of recommendations to return
            
        Returns:
            list: List of tuples (item, confidence_score)
        """
        if rules_df is None or len(rules_df) == 0:
            # Fallback to frequency-based recommendations
            return self.get_frequency_based_recommendations(selected_items, transactions, top_n)
        
        recommendations = {}
        
        # For each selected item, find rules where it appears in antecedents
        for item in selected_items:
            # Filter rules where item appears in antecedents
            item_rules = rules_df[rules_df['antecedents'].str.contains(item, case=False, na=False)]
            
            # Extract consequents and their confidence scores
            for _, rule in item_rules.iterrows():
                consequents = [c.strip() for c in rule['consequents'].split(',')]
                confidence = rule['confidence']
                
                for consequent in consequents:
                    # Don't recommend items already selected
                    if consequent not in selected_items:
                        # Use maximum confidence if item appears in multiple rules
                        if consequent not in recommendations or recommendations[consequent] < confidence:
                            recommendations[consequent] = confidence
        
        # If no recommendations from rules, use frequency-based approach
        if not recommendations:
            return self.get_frequency_based_recommendations(selected_items, transactions, top_n)
        
        # Sort by confidence and return top N
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return sorted_recommendations
    
    def get_frequency_based_recommendations(self, selected_items, transactions, top_n=5):
        """
        Get recommendations based on item co-occurrence frequency.
        
        Args:
            selected_items: List of currently selected items
            transactions: List of transactions
            top_n: Number of recommendations to return
            
        Returns:
            list: List of tuples (item, frequency_score)
        """
        # Find all items that appear with selected items
        cooccurring_items = []
        
        for transaction in transactions:
            # Check if any selected item is in this transaction
            if any(item in transaction for item in selected_items):
                # Add all other items from this transaction
                for item in transaction:
                    if item not in selected_items:
                        cooccurring_items.append(item)
        
        # Count frequencies
        item_counts = Counter(cooccurring_items)
        
        # Calculate frequency scores (normalized by total occurrences)
        total_occurrences = len(cooccurring_items)
        
        if total_occurrences == 0:
            return []
        
        recommendations = [
            (item, count / total_occurrences)
            for item, count in item_counts.most_common(top_n)
        ]
        
        return recommendations
    
    def get_similar_orders(self, selected_items, transactions, top_n=5):
        """
        Find similar orders based on Jaccard similarity.
        
        Args:
            selected_items: List of currently selected items
            transactions: List of transactions
            top_n: Number of similar orders to return
            
        Returns:
            list: List of similar transactions
        """
        selected_set = set(selected_items)
        similarities = []
        
        for transaction in transactions:
            transaction_set = set(transaction)
            
            # Calculate Jaccard similarity
            intersection = len(selected_set.intersection(transaction_set))
            union = len(selected_set.union(transaction_set))
            
            if union > 0:
                similarity = intersection / union
                similarities.append((transaction, similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def get_complementary_items(self, selected_items, transactions, top_n=5):
        """
        Get items that complement the selected items.
        
        Args:
            selected_items: List of currently selected items
            transactions: List of transactions
            top_n: Number of complementary items to return
            
        Returns:
            list: List of tuples (item, complement_score)
        """
        # Find transactions containing selected items
        relevant_transactions = [
            t for t in transactions
            if any(item in t for item in selected_items)
        ]
        
        # Count item frequencies in relevant transactions
        item_counts = Counter()
        for transaction in relevant_transactions:
            for item in transaction:
                if item not in selected_items:
                    item_counts[item] += 1
        
        # Calculate complement scores
        total_relevant = len(relevant_transactions)
        
        if total_relevant == 0:
            return []
        
        complementary = [
            (item, count / total_relevant)
            for item, count in item_counts.most_common(top_n)
        ]
        
        return complementary
    
    def get_popular_items(self, transactions, top_n=10):
        """
        Get most popular items overall.
        
        Args:
            transactions: List of transactions
            top_n: Number of popular items to return
            
        Returns:
            list: List of tuples (item, frequency)
        """
        all_items = []
        for transaction in transactions:
            all_items.extend(transaction)
        
        item_counts = Counter(all_items)
        total_items = len(all_items)
        
        popular = [
            (item, count / total_items)
            for item, count in item_counts.most_common(top_n)
        ]
        
        return popular
    
    def get_bundle_recommendations(self, selected_items, rules_df, min_items=2, top_n=5):
        """
        Get bundle recommendations (multiple items together).
        
        Args:
            selected_items: List of currently selected items
            rules_df: Association rules DataFrame
            min_items: Minimum number of items in bundle
            top_n: Number of bundles to return
            
        Returns:
            list: List of tuples (bundle, confidence)
        """
        if rules_df is None or len(rules_df) == 0:
            return []
        
        bundles = []
        
        for _, rule in rules_df.iterrows():
            antecedents = [a.strip() for a in rule['antecedents'].split(',')]
            consequents = [c.strip() for c in rule['consequents'].split(',')]
            
            # Check if any selected item is in antecedents
            if any(item in antecedents for item in selected_items):
                # Check if bundle has minimum items
                if len(consequents) >= min_items:
                    # Make sure none of the consequents are already selected
                    if not any(item in selected_items for item in consequents):
                        bundle = tuple(consequents)
                        confidence = rule['confidence']
                        bundles.append((bundle, confidence))
        
        # Sort by confidence and return top N unique bundles
        bundles = list(set(bundles))
        bundles.sort(key=lambda x: x[1], reverse=True)
        
        return bundles[:top_n]
    
    def get_personalized_recommendations(self, customer_cluster, cluster_data, top_n=5):
        """
        Get personalized recommendations based on customer cluster.
        
        Args:
            customer_cluster: Cluster ID of the customer
            cluster_data: Data for all customers with cluster labels
            top_n: Number of recommendations to return
            
        Returns:
            list: List of recommended items
        """
        # Filter data for the same cluster
        same_cluster = cluster_data[cluster_data['Cluster'] == customer_cluster]
        
        # Get most popular items in this cluster
        item_counts = Counter()
        for items in same_cluster['Items']:
            if isinstance(items, list):
                item_counts.update(items)
        
        # Return top N items
        recommendations = [item for item, _ in item_counts.most_common(top_n)]
        
        return recommendations
    
    def explain_recommendation(self, item, selected_items, rules_df):
        """
        Provide explanation for why an item is recommended.
        
        Args:
            item: Recommended item
            selected_items: List of selected items
            rules_df: Association rules DataFrame
            
        Returns:
            str: Explanation text
        """
        if rules_df is None or len(rules_df) == 0:
            return f"Customers who ordered similar items often also ordered {item}."
        
        # Find the best rule that led to this recommendation
        best_rule = None
        best_confidence = 0
        
        for _, rule in rules_df.iterrows():
            if item in rule['consequents']:
                antecedents = [a.strip() for a in rule['antecedents'].split(',')]
                if any(sel_item in antecedents for sel_item in selected_items):
                    if rule['confidence'] > best_confidence:
                        best_confidence = rule['confidence']
                        best_rule = rule
        
        if best_rule is not None:
            antecedents = best_rule['antecedents']
            confidence = best_rule['confidence'] * 100
            return f"Customers who ordered {antecedents} also ordered {item} {confidence:.0f}% of the time."
        else:
            return f"Customers who ordered similar items often also ordered {item}."

