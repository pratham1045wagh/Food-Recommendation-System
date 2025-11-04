# 🍽️ Food Recommendation System

A comprehensive data mining-based web application built with Streamlit that provides intelligent food recommendations and business analytics for restaurants.

## 📋 Overview

The Food Recommendation System leverages advanced data mining techniques including **Apriori Algorithm** for association rule mining and **K-Means Clustering** for customer segmentation to deliver actionable insights and personalized recommendations.

### Key Features

#### 👨‍💼 Owner Dashboard
- **Data Upload & Preprocessing**: Upload CSV files with automatic data cleaning and validation
- **Comprehensive Analytics**: View detailed statistics, KPIs, and performance metrics
- **Interactive Visualizations**: 
  - Top-selling items charts
  - Revenue analysis
  - Sales trends over time
  - Item co-occurrence heatmaps
  - Peak hours analysis
- **Association Rule Mining**: Discover patterns using the Apriori algorithm
- **Customer Segmentation**: Segment customers using K-Means clustering
- **Decision Support**: Data-driven insights for business decisions

#### 👤 Customer Module
- **Interactive Menu**: Browse available food items with prices
- **Smart Recommendations**: Get personalized suggestions based on:
  - Association rules (items frequently bought together)
  - Customer behavior patterns
  - Order history analysis
- **Real-time Suggestions**: Dynamic recommendations as you build your order

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

3. **Prepare your data**:
   - Ensure your CSV file has the following columns:
     - `Order Number`: Unique identifier for each order
     - `Order Date`: Date and time of order (format: DD/MM/YYYY HH:MM)
     - `Item Name`: Name of the food item
     - `Quantity`: Quantity ordered
     - `Product Price`: Price per item
     - `Total products`: Total items in the order

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Data Format

Your CSV file should follow this format:

```csv
Order Number,Order Date,Item Name,Quantity,Product Price,Total products
16118,03/08/2019 20:25,Plain Papadum,2,0.8,6
16118,03/08/2019 20:25,King Prawn Balti,1,12.95,6
16117,03/08/2019 20:17,Plain Naan,1,2.6,7
```

## 🔧 Technical Architecture

### Modules

#### 1. **data_preprocessing.py**
- Data cleaning and validation
- Missing value handling
- Feature engineering
- Transaction format conversion

#### 2. **association_rules.py**
- Apriori algorithm implementation
- Association rule generation
- Rule-based recommendations
- Support, confidence, and lift calculations

#### 3. **clustering.py**
- K-Means clustering implementation
- Customer segmentation
- Cluster profiling and characterization
- Feature extraction and scaling

#### 4. **visualization.py**
- Interactive charts using Plotly
- Heatmaps and statistical plots
- Time series analysis
- Distribution visualizations

#### 5. **recommendation.py**
- Rule-based recommendations
- Frequency-based suggestions
- Complementary item detection
- Bundle recommendations

## 📈 Data Mining Techniques

### Apriori Algorithm
- **Purpose**: Discover frequent itemsets and association rules
- **Metrics**:
  - **Support**: How frequently items appear together
  - **Confidence**: Likelihood of consequent given antecedent
  - **Lift**: Strength of association between items

### K-Means Clustering
- **Purpose**: Segment customers based on ordering behavior
- **Features Used**:
  - Total spending
  - Order frequency
  - Average basket size
  - Item variety
- **Segments Identified**:
  - High-Value Customers
  - Big Spenders
  - Frequent Buyers
  - Occasional Customers

## 💡 Usage Guide

### For Restaurant Owners

1. **Navigate to Owner Dashboard**
2. **Upload your order data CSV**
3. **Explore the four main sections**:
   - **Data Insights**: View summary statistics and visualizations
   - **Association Rules**: Discover item relationships and patterns
   - **Customer Segmentation**: Understand customer groups
   - **Decision Support**: Access KPIs and business recommendations

### For Customers

1. **Navigate to Customer Recommendations**
2. **Browse the menu** or search for specific items
3. **Select items** you want to order
4. **Click "Get Recommendations"** to see personalized suggestions
5. **View popular items** and complementary products

## 📊 Key Performance Indicators (KPIs)

The system tracks and displays:
- Total Orders
- Total Revenue
- Average Order Value
- Unique Items
- Average Items per Order
- Peak Hours
- Best Performing Days
- Top/Bottom Performers

## 🎯 Business Insights

The system provides actionable insights such as:
- Items frequently purchased together (for bundling)
- Customer segments for targeted marketing
- Peak hours for staff optimization
- Underperforming items for menu optimization
- Revenue trends for forecasting

## 🔮 Future Enhancements

- Integration with live POS systems
- Sentiment analysis from customer reviews
- Collaborative filtering for enhanced personalization
- Mobile app integration
- Inventory management recommendations
- Price optimization suggestions
- Seasonal trend analysis
- Multi-restaurant comparison

## 📝 Project Structure

```
project-dmw/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── restaurant-1-orders.csv         # Sample data
└── modules/
    ├── __init__.py
    ├── data_preprocessing.py       # Data cleaning and preprocessing
    ├── association_rules.py        # Apriori algorithm implementation
    ├── clustering.py               # K-Means clustering
    ├── visualization.py            # Data visualization functions
    └── recommendation.py           # Recommendation engine
```

## 🛠️ Technologies Used

- **Frontend/UI**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: scikit-learn, mlxtend
- **Algorithms**: Apriori, K-Means

## 📄 License

This project is developed for educational and research purposes.

## 👥 Contributors

Data Mining Project - Food Recommendation System

## 🤝 Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

## 📚 References

- Apriori Algorithm: Agrawal, R., & Srikant, R. (1994)
- K-Means Clustering: MacQueen, J. (1967)
- Association Rule Mining: Frequent Pattern Mining techniques
- Streamlit Documentation: https://docs.streamlit.io

---

**Note**: This system is designed as a decision support tool. Always validate recommendations with domain expertise and business requirements.

