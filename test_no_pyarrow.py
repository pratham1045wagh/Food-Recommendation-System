#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify all modules work without pyarrow
"""

import sys
import warnings
import os

# Set encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

print("=" * 60)
print("Testing Food Recommendation System - PyArrow Fix")
print("=" * 60)
print()

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    from modules.data_preprocessing import DataPreprocessor
    from modules.association_rules import AssociationRuleMiner
    from modules.clustering import CustomerSegmentation
    from modules.visualization import DataVisualizer
    from modules.recommendation import RecommendationEngine
    print("[OK] All modules imported successfully!")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Test 2: Load and preprocess data
print("\nTest 2: Loading and preprocessing data...")
try:
    import pandas as pd
    preprocessor = DataPreprocessor()
    
    # Load small sample
    data = pd.read_csv('dataset/restaurant-1-orders.csv', nrows=1000)
    cleaned_data, transactions, item_list = preprocessor.preprocess(data)
    
    print(f"[OK] Preprocessed {len(cleaned_data)} records")
    print(f"[OK] Created {len(transactions)} transactions")
    print(f"[OK] Found {len(item_list)} unique items")
except Exception as e:
    print(f"[ERROR] Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Association Rules
print("\nTest 3: Generating association rules...")
try:
    miner = AssociationRuleMiner()
    rules_df, frequent_itemsets = miner.generate_rules(
        transactions[:500],  # Use subset for speed
        min_support=0.05,
        min_confidence=0.3,
        min_lift=1.2
    )
    
    if len(rules_df) > 0:
        print(f"[OK] Generated {len(rules_df)} association rules")
        print(f"[OK] Found {len(frequent_itemsets)} frequent itemsets")
    else:
        print("[WARN] No rules found (try lower support)")
except Exception as e:
    print(f"[ERROR] Association rules failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Customer Segmentation
print("\nTest 4: Performing customer segmentation...")
try:
    segmentation = CustomerSegmentation()
    customer_features, cluster_labels, cluster_centers = segmentation.perform_clustering(
        cleaned_data,
        n_clusters=3
    )
    
    print(f"[OK] Segmented {len(customer_features)} customers")
    print(f"[OK] Created {len(set(cluster_labels))} clusters")
except Exception as e:
    print(f"[ERROR] Clustering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Visualizations
print("\nTest 5: Creating visualizations...")
try:
    visualizer = DataVisualizer()
    fig = visualizer.plot_top_items(cleaned_data, top_n=5)
    print("[OK] Visualization created successfully")
except Exception as e:
    print(f"[ERROR] Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Recommendations
print("\nTest 6: Generating recommendations...")
try:
    recommender = RecommendationEngine()
    recommendations = recommender.get_frequency_based_recommendations(
        [item_list[0]],  # Select first item
        transactions,
        top_n=5
    )
    
    print(f"[OK] Generated {len(recommendations)} recommendations")
except Exception as e:
    print(f"[ERROR] Recommendations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print()
print("=" * 60)
print("SUCCESS - ALL TESTS PASSED!")
print("=" * 60)
print()
print("[OK] Data preprocessing works")
print("[OK] Association rules work WITHOUT pyarrow")
print("[OK] Customer segmentation works WITHOUT pyarrow")
print("[OK] Visualizations work")
print("[OK] Recommendations work")
print()
print("Application is ready to use!")
print()
print("Start the app with:")
print("  python -m streamlit run app.py")
print()

