# 📊 Food Recommendation System - Detailed Project Report

**Project Name:** Data Mining-Based Food Recommendation System  
**Technology Stack:** Python, Streamlit, Pandas, Scikit-learn, MLxtend, Plotly  
**Project Type:** Web Application with Data Mining and Machine Learning  
**Date:** December 2024

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Project Objectives](#project-objectives)
4. [System Architecture](#system-architecture)
5. [Technical Implementation](#technical-implementation)
6. [Features and Functionality](#features-and-functionality)
7. [Data Mining Algorithms](#data-mining-algorithms)
8. [User Interface Design](#user-interface-design)
9. [Data Flow and Processing](#data-flow-and-processing)
10. [Key Performance Indicators](#key-performance-indicators)
11. [Implementation Details](#implementation-details)
12. [Testing and Quality Assurance](#testing-and-quality-assurance)
13. [Project Challenges and Solutions](#project-challenges-and-solutions)
14. [Future Enhancements](#future-enhancements)
15. [Conclusion](#conclusion)

---

## 1. Executive Summary

The **Food Recommendation System** is a comprehensive data mining web application designed to provide intelligent food recommendations and business analytics for restaurants. Built using Streamlit and Python, the system leverages advanced data mining techniques including **Apriori Algorithm** for association rule mining and **K-Means Clustering** for customer segmentation.

### Key Achievements
- ✅ Successful implementation of association rule mining for item recommendation
- ✅ Customer segmentation using K-Means clustering
- ✅ Comprehensive business analytics dashboard
- ✅ Interactive data visualizations
- ✅ Dual-mode user interface (Owner Dashboard & Customer Recommendations)
- ✅ Advanced theming system (Dark/Light mode)
- ✅ Real-time recommendation engine

### Business Impact
- Enables data-driven decision making for restaurant owners
- Improves customer experience through personalized recommendations
- Identifies cross-selling opportunities through association rules
- Facilitates targeted marketing through customer segmentation
- Provides actionable business insights through KPIs and analytics

---

## 2. Project Overview

### 2.1 Project Description
The Food Recommendation System is a data mining project that processes restaurant order data to extract meaningful patterns, relationships, and insights. The system serves two primary user groups:

1. **Restaurant Owners**: Business analytics and decision support
2. **Customers**: Personalized food recommendations

### 2.2 Problem Statement
Restaurants face several challenges:
- **Lack of data-driven insights**: Limited understanding of customer preferences and buying patterns
- **Inefficient cross-selling**: Missing opportunities to suggest complementary items
- **Poor customer segmentation**: Inability to target specific customer groups
- **Limited personalization**: Generic recommendations without understanding individual preferences

### 2.3 Solution Approach
The system addresses these challenges through:
- **Association Rule Mining**: Discover which items are frequently bought together
- **Customer Segmentation**: Group customers based on ordering behavior
- **Real-time Recommendations**: Provide personalized suggestions based on selected items
- **Comprehensive Analytics**: Generate actionable insights from order data

---

## 3. Project Objectives

### 3.1 Primary Objectives
1. **Data Mining**: Extract hidden patterns and relationships from restaurant order data
2. **Recommendation System**: Provide intelligent food recommendations to customers
3. **Business Analytics**: Deliver actionable insights for restaurant owners
4. **Customer Segmentation**: Identify distinct customer groups for targeted marketing

### 3.2 Technical Objectives
1. Implement Apriori algorithm for association rule mining
2. Implement K-Means clustering for customer segmentation
3. Develop interactive visualizations for data exploration
4. Create user-friendly web interface using Streamlit
5. Ensure data quality through preprocessing and validation

### 3.3 Business Objectives
1. Improve cross-selling opportunities
2. Enhance customer satisfaction through personalization
3. Optimize menu offerings based on data insights
4. Support data-driven decision making

---

## 4. System Architecture

### 4.1 High-Level Architecture

The system follows a modular architecture with clear separation of concerns:

```
Streamlit Web Interface
    ├── Owner Dashboard
    │   ├── Data Upload
    │   ├── Data Insights
    │   ├── Association Rules
    │   ├── Customer Segmentation
    │   └── Decision Support
    └── Customer Module
        ├── Menu Browsing
        └── Recommendations

Business Logic Layer
    ├── Data Preprocessing Module
    ├── Association Rules Module
    ├── Clustering Module
    ├── Recommendation Engine
    └── Visualization Module

Data Layer
    └── CSV File Input
```

### 4.2 Component Architecture

#### 4.2.1 Frontend Layer
- **Framework**: Streamlit (Python web framework)
- **Visualization**: Plotly (interactive), Matplotlib/Seaborn (static)
- **Theming**: Custom CSS with Dark/Light mode support

#### 4.2.2 Business Logic Layer
- **Data Preprocessing**: `data_preprocessing.py`
- **Association Rules**: `association_rules.py` (Apriori algorithm)
- **Clustering**: `clustering.py` (K-Means)
- **Recommendations**: `recommendation.py`
- **Visualization**: `visualization.py`

#### 4.2.3 Data Layer
- **Input Format**: CSV files
- **Storage**: Pandas DataFrames (in-memory)
- **Processing**: NumPy arrays for numerical operations

---

## 5. Technical Implementation

### 5.1 Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Web Framework** | Streamlit | ≥1.28.0 | UI and application framework |
| **Data Processing** | Pandas | ≥2.0.0 | Data manipulation and analysis |
| **Numerical Computing** | NumPy | ≥1.24.0 | Numerical operations |
| **Visualization** | Plotly | ≥5.14.0 | Interactive charts |
| **Visualization** | Matplotlib | ≥3.7.0 | Static plots |
| **Visualization** | Seaborn | ≥0.12.0 | Statistical visualizations |
| **Machine Learning** | Scikit-learn | ≥1.3.0 | K-Means clustering |
| **Association Rules** | MLxtend | ≥0.22.0 | Apriori algorithm |
| **Data Format** | PyArrow | ≥7.0 | CSV handling |

### 5.2 Module Structure

```
project-dmw/
├── app.py                          # Main application (1535 lines)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── PROJECT_REPORT.md              # This detailed report
├── dataset/
│   └── restaurant-1-orders.csv    # Sample dataset
└── modules/
    ├── __init__.py                 # Package initialization
    ├── data_preprocessing.py       # Data cleaning (211 lines)
    ├── association_rules.py        # Apriori algorithm (233 lines)
    ├── clustering.py               # K-Means clustering (257 lines)
    ├── visualization.py            # Chart generation (509 lines)
    └── recommendation.py           # Recommendation engine (293 lines)
```

### 5.3 Core Classes and Functions

#### 5.3.1 DataPreprocessor (`data_preprocessing.py`)
**Purpose**: Handles data loading, cleaning, and preprocessing

**Key Methods**:
- `preprocess(data)`: Main preprocessing pipeline
- `_handle_missing_values(df)`: Missing value imputation
- `_remove_duplicates(df)`: Duplicate removal
- `_convert_data_types(df)`: Type conversions
- `_validate_data(df)`: Data validation
- `_engineer_features(df)`: Feature engineering (Line_Total, Hour, Day)
- `_create_transactions(df)`: Transaction format conversion

**Features**:
- Automatic missing value handling
- Data type conversion (dates, numeric)
- Feature engineering
- Transaction format creation for association rules

#### 5.3.2 AssociationRuleMiner (`association_rules.py`)
**Purpose**: Implements Apriori algorithm for association rule mining

**Key Methods**:
- `generate_rules(transactions, min_support, min_confidence, min_lift)`: Generate association rules
- `get_rules_for_item(item_name)`: Get rules for specific item
- `get_recommendations_from_rules(selected_items)`: Generate recommendations
- `get_frequent_itemsets_summary()`: Summary statistics

**Algorithm Parameters**:
- **Minimum Support**: 0.02 (2% of transactions)
- **Minimum Confidence**: 0.3 (30%)
- **Minimum Lift**: 1.2 (20% more likely than random)

#### 5.3.3 CustomerSegmentation (`clustering.py`)
**Purpose**: Implements K-Means clustering for customer segmentation

**Key Methods**:
- `perform_clustering(data, n_clusters)`: Perform K-Means clustering
- `_extract_customer_features(data)`: Extract customer-level features
- `get_cluster_characteristics(features, labels)`: Cluster analysis
- `get_cluster_profiles(features, labels)`: Detailed cluster profiles
- `find_optimal_clusters(data, max_clusters)`: Optimal cluster count using silhouette score

**Features Used**:
- Total Spending
- Order Frequency
- Average Basket Size
- Item Variety

**Default Clusters**: 4 segments

#### 5.3.4 RecommendationEngine (`recommendation.py`)
**Purpose**: Generates food recommendations

**Key Methods**:
- `get_recommendations(selected_items, rules_df, transactions)`: Rule-based recommendations
- `get_frequency_based_recommendations(selected_items, transactions)`: Frequency-based fallback
- `get_complementary_items(selected_items, transactions)`: Complementary item detection
- `get_popular_items(transactions)`: Popular item suggestions
- `get_bundle_recommendations(selected_items, rules_df)`: Bundle suggestions

**Recommendation Strategies**:
1. **Association Rule-Based**: Uses confidence scores from rules
2. **Frequency-Based**: Based on co-occurrence frequency
3. **Complementary Items**: Items frequently bought together
4. **Popular Items**: Most frequently ordered items

#### 5.3.5 DataVisualizer (`visualization.py`)
**Purpose**: Generates interactive and static visualizations

**Key Methods**:
- `plot_top_items(data, top_n)`: Top-selling items bar chart
- `plot_revenue_by_item(data, top_n)`: Revenue by item chart
- `plot_sales_trend(data)`: Time series sales analysis
- `plot_cooccurrence_heatmap(data, top_n)`: Item co-occurrence visualization
- `plot_hourly_heatmap(data)`: Peak hours analysis
- `plot_quantity_distribution(data)`: Quantity distribution histogram

**Visualization Features**:
- Theme-aware (Dark/Light mode)
- Interactive Plotly charts
- Statistical Seaborn heatmaps
- Responsive design with proper margins

---

## 6. Features and Functionality

### 6.1 Owner Dashboard Features

#### 6.1.1 Data Upload & Preprocessing
- **CSV File Upload**: Support for restaurant order data (up to 200MB)
- **Automatic Validation**: Column validation and data type checking
- **Preprocessing Summary**: Statistics on data quality and cleaning
- **Data Preview**: First few rows preview after preprocessing

**Required CSV Format**:
```csv
Order Number,Order Date,Item Name,Quantity,Product Price,Total products
16118,03/08/2019 20:25,Plain Papadum,2,0.8,6
```

#### 6.1.2 Data Insights Section
- **Summary Statistics**: Total Orders, Revenue, AOV, Unique Items
- **Top 10 Best-Selling Items**: Horizontal bar chart with quantities
- **Top 10 Items by Revenue**: Revenue-based ranking
- **Sales Trend Analysis**: Time series with quantity and revenue dual-axis
- **Item Co-occurrence Heatmap**: Most common item combinations
- **Peak Hours Heatmap**: Busiest hours analysis
- **Distribution Charts**: Quantity, order value, basket size distributions

#### 6.1.3 Association Rules Section
- **Rule Generation**: Configurable parameters (support, confidence, lift)
- **Rule Visualization**: Network graph showing item relationships
- **Rule Table**: Detailed table with metrics (antecedents, consequents, support, confidence, lift)
- **Item-Specific Rules**: Filter rules by specific items
- **Rule Export**: Export rules to CSV

#### 6.1.4 Customer Segmentation Section
- **Cluster Analysis**: K-Means clustering results with configurable k
- **Cluster Characteristics**: Statistics for each cluster
- **Cluster Visualization**: Feature comparison across clusters
- **Cluster Profiles**: Business interpretation of each segment
- **Optimal Cluster Detection**: Silhouette score analysis

**Customer Segments**:
1. **High-Value Customers**: High spending, frequent orders
2. **Big Spenders**: Very high spending per order
3. **Frequent Buyers**: High order frequency, moderate spending
4. **Occasional Customers**: Low frequency, low spending

#### 6.1.5 Decision Support Section
- **Key Performance Indicators (KPIs)**: Revenue, AOV, items per order, peak hours
- **Business Recommendations**: Data-driven suggestions
- **Performance Metrics**: Comparative analysis
- **Trend Analysis**: Time-based patterns

### 6.2 Customer Module Features

#### 6.2.1 Interactive Menu
- **Item Browsing**: Scrollable list of available items
- **Search Functionality**: Search items by name
- **Price Display**: Show prices for each item
- **Item Selection**: Multi-select checkboxes

#### 6.2.2 Smart Recommendations
- **Rule-Based Recommendations**: Based on association rules with confidence scores
- **Complementary Items**: Items frequently bought together
- **Popular Items**: Most ordered items
- **Bundle Suggestions**: Item bundles with high confidence
- **Recommendation Explanations**: Why items are recommended

---

## 7. Data Mining Algorithms

### 7.1 Apriori Algorithm

#### 7.1.1 Algorithm Overview
The Apriori algorithm discovers frequent itemsets and generates association rules.

**Algorithm Steps**:
1. Find all frequent 1-itemsets
2. Generate candidate k-itemsets from frequent (k-1)-itemsets
3. Count support for each candidate
4. Prune candidates below minimum support
5. Repeat until no more frequent itemsets
6. Generate association rules

#### 7.1.2 Metrics

**Support**: `Support(A → B) = P(A ∪ B) = |A ∪ B| / |Total Transactions|`
- Measures frequency of items appearing together
- Threshold: 0.02 (2%)

**Confidence**: `Confidence(A → B) = P(B | A) = Support(A ∪ B) / Support(A)`
- Measures likelihood of B given A
- Threshold: 0.3 (30%)

**Lift**: `Lift(A → B) = Confidence(A → B) / Support(B)`
- Measures strength of association
- Lift > 1: Positive association
- Threshold: 1.2

### 7.2 K-Means Clustering

#### 7.2.1 Algorithm Overview
Groups customers into segments based on ordering behavior.

**Algorithm Steps**:
1. Initialize k cluster centroids
2. Assign each data point to nearest centroid
3. Recalculate cluster centroids
4. Repeat until convergence
5. Output cluster labels and centroids

#### 7.2.2 Features Used
- Total Spending
- Order Frequency
- Average Basket Size
- Item Variety

#### 7.2.3 Optimal Cluster Selection
- **Method**: Silhouette Score
- **Range**: Test k from 2 to 10
- **Selection**: Maximum silhouette score

---

## 8. Key Performance Indicators

### 8.1 Business KPIs

| KPI | Description | Calculation |
|-----|-------------|-------------|
| **Total Revenue** | Sum of all order values | `SUM(Line_Total)` |
| **Average Order Value** | Average revenue per order | `Total Revenue / Total Orders` |
| **Items per Order** | Average items in each order | `Total Items / Total Orders` |
| **Peak Hours** | Busiest hours | Most frequent order hours |
| **Top Items** | Best-selling items | Items with highest quantity |

### 8.2 Model Performance Metrics

#### Association Rules
- Number of Rules generated
- Average Support, Confidence, Lift
- High-Lift Rules (lift > 2.0)

#### Clustering
- Silhouette Score (cluster quality)
- Within-Cluster Sum of Squares
- Cluster separation distances

---

## 9. Testing and Quality Assurance

### 9.1 Data Validation Testing
- ✅ CSV format validation
- ✅ Required column checking
- ✅ Data type validation
- ✅ Missing value handling
- ✅ Duplicate detection

### 9.2 Algorithm Testing
- ✅ Association rules generation with various thresholds
- ✅ Clustering with different k values
- ✅ Recommendation accuracy testing
- ✅ Edge cases handling

### 9.3 UI/UX Testing
- ✅ Theme switching (Dark/Light mode)
- ✅ Chart visibility in both themes
- ✅ File upload functionality
- ✅ Responsive design
- ✅ Cross-browser compatibility

### 9.4 Known Issues and Fixes

**Issue 1: PyArrow Warnings**
- **Solution**: Suppressed warnings and used standard pandas dtypes

**Issue 2: Chart Visibility in Light Mode**
- **Solution**: Explicit color settings for all chart elements

**Issue 3: Plotly titlefont Deprecation**
- **Solution**: Replaced with `title=dict(font=dict(...))`

**Issue 4: File Uploader Visibility**
- **Solution**: Theme-specific background colors

---

## 10. Project Challenges and Solutions

### 10.1 Challenge 1: Data Quality Issues
**Problem**: Inconsistent date formats, missing values, invalid data  
**Solution**: Comprehensive preprocessing pipeline with validation

### 10.2 Challenge 2: Association Rule Performance
**Problem**: Too many rules with low thresholds  
**Solution**: Configurable thresholds and rule filtering

### 10.3 Challenge 3: Customer Segmentation Without Customer ID
**Problem**: No customer ID, only order numbers  
**Solution**: Used order-level features grouped by order number

### 10.4 Challenge 4: Theme Compatibility
**Problem**: Charts not visible in both themes  
**Solution**: Theme-aware visualization with explicit color settings

### 10.5 Challenge 5: Large Dataset Performance
**Problem**: Slow processing with large datasets  
**Solution**: Optimized pandas operations and efficient data structures

---

## 11. Future Enhancements

### 11.1 Short-Term
1. Real-time Data Integration (POS systems)
2. User Authentication (multiple restaurants)
3. Export Functionality (PDF/Excel reports)
4. Email Notifications
5. Mobile Responsive Design

### 11.2 Medium-Term
1. Collaborative Filtering
2. Sentiment Analysis (customer reviews)
3. Seasonal Trends Analysis
4. Price Optimization (ML-based)
5. Inventory Management Recommendations

### 11.3 Long-Term
1. Deep Learning Models
2. Multi-restaurant Comparison
3. Predictive Analytics (demand forecasting)
4. A/B Testing Framework
5. RESTful API Integration

---

## 12. Conclusion

### 12.1 Project Summary
The Food Recommendation System successfully implements advanced data mining techniques to provide intelligent recommendations and business analytics for restaurants.

**Key Achievements**:
- ✅ Effective data mining and pattern extraction
- ✅ Practical real-world applications
- ✅ User-friendly interface
- ✅ Scalable modular architecture

### 12.2 Business Impact
- **Increased Revenue**: Through cross-selling recommendations
- **Better Customer Experience**: Personalized suggestions
- **Data-Driven Decisions**: Evidence-based menu optimization
- **Targeted Marketing**: Customer segmentation for campaigns
- **Operational Efficiency**: Peak hours and inventory insights

### 12.3 Learning Outcomes
- Data preprocessing and validation techniques
- Association rule mining and interpretation
- Customer segmentation methodologies
- Web application development with Streamlit
- Data visualization best practices
- Machine learning model evaluation

---

## 📚 Appendix

### A. References
1. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. VLDB.
2. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
3. Streamlit Documentation: https://docs.streamlit.io
4. MLxtend Documentation: http://rasbt.github.io/mlxtend/
5. Scikit-learn Documentation: https://scikit-learn.org/

### B. Glossary
- **Association Rules**: If-then relationships between items
- **Support**: Frequency of itemset occurrence
- **Confidence**: Conditional probability of consequent given antecedent
- **Lift**: Measure of association strength
- **K-Means**: Partitioning clustering algorithm
- **Silhouette Score**: Cluster quality metric
- **Transactions**: Sets of items purchased together

---

**Report Generated**: December 2024  
**Project Version**: 1.0  
**Status**: Completed ✅
