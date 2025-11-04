import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os
import sys

# Suppress all warnings including pyarrow warnings
warnings.filterwarnings('ignore')
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# Suppress stderr for pyarrow warnings
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# Page configuration
st.set_page_config(
    page_title="Food Recommendation System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from modules.data_preprocessing import DataPreprocessor
from modules.association_rules import AssociationRuleMiner
from modules.clustering import CustomerSegmentation
from modules.visualization import DataVisualizer
from modules.recommendation import RecommendationEngine

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'rules' not in st.session_state:
    st.session_state.rules = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def apply_theme():
    """Apply custom CSS based on selected theme"""
    theme = st.session_state.theme
    
    if theme == 'dark':
        # Dark theme colors
        bg_color = "#0E1117"
        secondary_bg = "#262730"
        text_color = "#FAFAFA"
        border_color = "#444654"
        accent_color = "#FF4B4B"
        success_color = "#21C354"
        warning_color = "#FFA500"
        info_color = "#00A9E0"
        chart_bg = "#262730"
        metric_bg = "#1E1E1E"
        file_uploader_bg = "#262730"
        file_uploader_border = "#444654"
    else:
        # Light theme colors
        bg_color = "#FFFFFF"
        secondary_bg = "#F8F9FA"
        text_color = "#1F2937"
        border_color = "#D1D5DB"
        accent_color = "#DC2626"
        success_color = "#059669"
        warning_color = "#D97706"
        info_color = "#0284C7"
        chart_bg = "#FFFFFF"
        metric_bg = "#F3F4F6"
        file_uploader_bg = "#FAFAFA"
        file_uploader_border = "#6B7280"
    
    css = f"""
    <style>
        /* Main background */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {secondary_bg};
        }}
        
        /* Sidebar text */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6 {{
            color: {text_color} !important;
        }}
        
        /* Sidebar radio buttons */
        [data-testid="stSidebar"] .stRadio > div > label {{
            color: {text_color} !important;
        }}
        
        /* Sidebar selectbox */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            color: {text_color} !important;
        }}
        
        /* Sidebar markdown links */
        [data-testid="stSidebar"] a {{
            color: {info_color} !important;
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {text_color};
        }}
        
        [data-testid="stMetricDelta"] svg {{
            color: {text_color};
        }}
        
        div[data-testid="metric-container"] {{
            background-color: {metric_bg};
            border: 1px solid {border_color};
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        /* Cards and containers */
        .element-container {{
            color: {text_color};
        }}
        
        .element-container p,
        .element-container span,
        .element-container div,
        .element-container label,
        .element-container small {{
            color: {text_color} !important;
        }}
        
        /* Expander */
        [data-testid="stExpander"] {{
            background-color: {secondary_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        }}
        
        [data-testid="stExpander"] summary,
        .streamlit-expanderHeader {{
            background-color: {secondary_bg} !important;
            color: {text_color} !important;
            border-radius: 5px;
            padding: 10px !important;
        }}
        
        .streamlit-expanderHeader p,
        .streamlit-expanderHeader span,
        .streamlit-expanderHeader div {{
            color: {text_color} !important;
        }}
        
        .streamlit-expanderContent {{
            background-color: {secondary_bg};
            color: {text_color};
            padding: 10px !important;
        }}
        
        .streamlit-expanderContent p,
        .streamlit-expanderContent span,
        .streamlit-expanderContent div,
        .streamlit-expanderContent label {{
            color: {text_color} !important;
        }}
        
        /* Buttons */
        .stButton>button {{
            border-radius: 8px;
            border: 1px solid {border_color};
            transition: all 0.3s ease;
            color: {text_color};
            background-color: {secondary_bg};
        }}
        
        /* Primary buttons */
        .stButton>button[data-testid="baseButton-secondary"] {{
            color: {text_color};
            border-color: {border_color};
        }}
        
        .stButton>button[data-testid="baseButton-secondary"]:hover {{
            border-color: {accent_color};
            box-shadow: 0 0 10px {accent_color}50;
        }}
        
        /* Primary button colors */
        .stButton>button[kind="primary"],
        .stButton>button[type="primary"] {{
            background-color: {accent_color};
            color: white;
            border-color: {accent_color};
        }}
        
        .stButton>button[kind="primary"]:hover,
        .stButton>button[type="primary"]:hover {{
            background-color: {accent_color};
            opacity: 0.9;
            border-color: {accent_color};
        }}
        
        .stButton>button:hover {{
            border-color: {accent_color};
            box-shadow: 0 0 10px {accent_color}50;
        }}
        
        /* Info boxes */
        .stAlert {{
            border-radius: 10px;
        }}
        
        /* Dataframes */
        .dataframe {{
            border: 1px solid {border_color} !important;
        }}
        
        .dataframe th {{
            background-color: {metric_bg};
            color: {text_color} !important;
            border-color: {border_color} !important;
        }}
        
        .dataframe td {{
            color: {text_color} !important;
            border-color: {border_color} !important;
        }}
        
        .dataframe tr:nth-child(even) {{
            background-color: {secondary_bg};
        }}
        
        .dataframe tr:hover {{
            background-color: {metric_bg};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {secondary_bg};
            border-radius: 10px;
            padding: 5px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            color: {text_color} !important;
        }}
        
        .stTabs [data-baseweb="tab"] span {{
            color: {text_color} !important;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_color} !important;
        }}
        
        /* Text */
        p, span, label {{
            color: {text_color} !important;
        }}
        
        /* Markdown */
        .stMarkdown {{
            color: {text_color} !important;
        }}
        
        /* Input fields */
        .stTextInput>div>div>input {{
            color: {text_color};
            background-color: {bg_color};
            border-color: {border_color};
        }}
        
        /* Selectbox */
        [data-baseweb="select"] {{
            color: {text_color};
            background-color: {bg_color};
        }}
        
        [data-baseweb="select"]>div {{
            color: {text_color} !important;
        }}
        
        /* Slider */
        .stSlider {{
            color: {text_color};
        }}
        
        /* Multiselect */
        .stMultiSelect {{
            color: {text_color};
        }}
        
        .stMultiSelect>div>div>div {{
            color: {text_color} !important;
        }}
        
        /* File uploader */
        [data-testid="stFileUploader"] {{
            background-color: {file_uploader_bg} !important;
            border-radius: 10px !important;
            border: 2px dashed {file_uploader_border} !important;
            padding: 20px !important;
        }}
        
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] label {{
            color: {text_color} !important;
        }}
        
        [data-testid="stFileUploader"] [data-baseweb="base-button"] {{
            background-color: {accent_color} !important;
            color: white !important;
            border: none !important;
        }}
        
        [data-testid="stFileUploader"] [data-baseweb="file-uploader"] {{
            background-color: {file_uploader_bg} !important;
            border: 2px dashed {file_uploader_border} !important;
        }}
        
        .stFileUploaderSmall {{
            background-color: {file_uploader_bg} !important;
        }}
        
        .stFileUploaderSmall label {{
            color: {text_color} !important;
        }}
        
        /* File uploader internal elements */
        [data-testid="stFileUploader"] * {{
            color: {text_color} !important;
        }}
        
        [data-testid="stFileUploader"] [data-baseweb="text"] {{
            color: {text_color} !important;
        }}
        
        [data-testid="stFileUploader"] [data-baseweb="button"] {{
            background-color: {accent_color} !important;
            color: white !important;
        }}
        
        [data-testid="stFileUploader"] [data-baseweb="button"]:hover {{
            background-color: {accent_color} !important;
            opacity: 0.9 !important;
        }}
        
        /* Radio buttons */
        .stRadio>div {{
            color: {text_color};
        }}
        
        .stRadio>div>label {{
            color: {text_color} !important;
        }}
        
        /* Checkbox */
        .stCheckbox>label {{
            color: {text_color} !important;
        }}
        
        /* Code blocks */
        .stCodeBlock {{
            border: 1px solid {border_color};
        }}
        
        code {{
            color: {text_color} !important;
        }}
        
        /* Theme toggle button */
        .theme-toggle {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
            background: linear-gradient(135deg, {accent_color}20, {info_color}20);
            border-radius: 10px;
            margin: 10px 0;
        }}
        
        /* Improve visibility for info/warning/success boxes */
        .stAlert [data-testid="stMarkdownContainer"] {{
            color: {text_color};
        }}
        
        /* Sidebar info box */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: {text_color} !important;
        }}
        
        /* Dividers */
        hr {{
            border-color: {border_color};
        }}
        
        /* Streamlit divs */
        .element-container {{
            color: {text_color};
        }}
        
        /* Styling for all Streamlit text */
        div[data-baseweb="base-input"] input {{
            color: {text_color};
        }}
        
        /* Metric labels */
        [data-testid="stMetricLabel"] {{
            color: {text_color};
        }}
        
        /* Additional file uploader styling */
        div[data-baseweb="file-uploader"] {{
            border: 2px dashed {file_uploader_border} !important;
            background-color: {file_uploader_bg} !important;
        }}
        
        div[data-baseweb="file-uploader"] span,
        div[data-baseweb="file-uploader"] p,
        div[data-baseweb="file-uploader"] div,
        div[data-baseweb="file-uploader"] label {{
            color: {text_color} !important;
        }}
        
        /* Upload text visibility */
        .uploadedFile {{
            color: {text_color} !important;
        }}
        
        .uploadedFileData {{
            color: {text_color} !important;
        }}
        
        .uploadedFileActions {{
            color: {text_color} !important;
        }}
        
        /* File uploader helper text below */
        .stFileUploader label + div,
        .stFileUploader > div > div {{
            color: {text_color} !important;
        }}
        
        /* Streamlit file uploader all text */
        .element-container .uploadedFileMessage,
        .element-container small,
        .element-container [class*="Uploader"] {{
            color: {text_color} !important;
        }}
        
        /* All elements after file uploader */
        div[data-testid="stFileUploader"] ~ div,
        div[data-testid="stFileUploader"] ~ span,
        div[data-testid="stFileUploader"] ~ p {{
            color: {text_color} !important;
        }}
        
        /* Widget label styling */
        .stWidgetLabel {{
            color: {text_color} !important;
        }}
        
        /* Caption text */
        .caption {{
            color: {text_color} !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def get_chart_theme():
    """Get color scheme for charts based on current theme"""
    if st.session_state.theme == 'dark':
        return {
            'bg_color': '#262730',
            'text_color': '#FAFAFA',
            'grid_color': '#444654',
            'primary_color': '#FF4B4B',
            'secondary_color': '#00A9E0',
            'colorscale': 'viridis',
            'template': 'plotly_dark'
        }
    else:
        return {
            'bg_color': '#FFFFFF',
            'text_color': '#1F2937',
            'grid_color': '#E5E7EB',
            'primary_color': '#DC2626',
            'secondary_color': '#0284C7',
            'colorscale': 'Blues',
            'template': 'none'
        }

def main():
    # Apply theme
    apply_theme()
    
    # Sidebar navigation
    st.sidebar.title("🍽️ Food Recommendation System")
    st.sidebar.markdown("---")
    
    # Theme Toggle
    st.sidebar.markdown("### 🎨 Theme Settings")
    theme_col1, theme_col2 = st.sidebar.columns([1, 2])
    
    with theme_col1:
        current_theme_emoji = "🌙" if st.session_state.theme == 'dark' else "☀️"
        st.markdown(f"<div style='font-size: 2em; text-align: center;'>{current_theme_emoji}</div>", unsafe_allow_html=True)
    
    with theme_col2:
        new_theme = st.selectbox(
            "Mode",
            options=['dark', 'light'],
            index=0 if st.session_state.theme == 'dark' else 1,
            key='theme_selector',
            label_visibility="collapsed"
        )
        
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()
    
    st.sidebar.markdown("---")
    
    module = st.sidebar.radio(
        "Select Module",
        ["Home", "Owner Dashboard", "Customer Recommendations"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About:**\n\n"
        "This system uses data mining techniques including:\n"
        "- Apriori Algorithm\n"
        "- K-Means Clustering\n"
        "- Association Rule Mining\n\n"
        "to provide insights and recommendations."
    )
    
    if module == "Home":
        show_home()
    elif module == "Owner Dashboard":
        show_owner_dashboard()
    elif module == "Customer Recommendations":
        show_customer_recommendations()

def show_home():
    st.title("🍽️ Food Recommendation System")
    st.markdown("### Welcome to the Intelligent Food Recommendation Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 👨‍💼 Owner Module
        - Upload and analyze restaurant order data
        - View comprehensive data insights and visualizations
        - Discover patterns using association rule mining
        - Segment customers using clustering algorithms
        - Make data-driven decisions with KPI dashboards
        """)
        
    with col2:
        st.markdown("""
        #### 👤 Customer Module
        - Browse available menu items
        - Get personalized food recommendations
        - Discover items frequently bought together
        - Receive suggestions based on order patterns
        - Enhanced ordering experience
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Key Features
    
    **Data Mining Techniques:**
    - **Apriori Algorithm**: Discovers frequent itemsets and association rules
    - **K-Means Clustering**: Segments customers based on ordering behavior
    - **Pattern Analysis**: Identifies trends and insights from order data
    
    **Visualization:**
    - Interactive charts and graphs
    - Real-time data analytics
    - Comprehensive statistical summaries
    
    **Decision Support:**
    - KPI tracking and monitoring
    - Peak hours analysis
    - Sales trend forecasting
    """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between Owner Dashboard and Customer Recommendations")

def show_owner_dashboard():
    st.title("👨‍💼 Owner Dashboard")
    st.markdown("### Data Analysis & Decision Support System")
    
    # Data Upload Section
    st.markdown("## 📤 Data Upload & Preprocessing")
    
    # Option 1: Load default dataset
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Restaurant Order Data (CSV)",
            type=['csv'],
            help="Upload your restaurant order data in CSV format"
        )
    
    with col2:
        st.markdown("**Or use default dataset:**")
        if st.button("📁 Load Default Dataset", type="primary", use_container_width=True):
            # Load from dataset folder
            default_file_path = os.path.join("dataset", "restaurant-1-orders.csv")
            if os.path.exists(default_file_path):
                try:
                    with st.spinner("Loading and preprocessing default dataset..."):
                        preprocessor = DataPreprocessor()
                        raw_data = pd.read_csv(default_file_path)
                        
                        # Preprocess
                        cleaned_data, transactions, item_list = preprocessor.preprocess(raw_data)
                        
                        # Store in session state
                        st.session_state.data_loaded = True
                        st.session_state.preprocessed_data = {
                            'cleaned_data': cleaned_data,
                            'transactions': transactions,
                            'item_list': item_list,
                            'raw_data': raw_data
                        }
                    st.success("✅ Default dataset loaded and preprocessed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading default dataset: {str(e)}")
            else:
                st.error(f"❌ Default dataset not found at: {default_file_path}")
    
    # Check if data already loaded
    if st.session_state.data_loaded and uploaded_file is None:
        # Data was loaded via button, display it
        cleaned_data = st.session_state.preprocessed_data['cleaned_data']
        transactions = st.session_state.preprocessed_data['transactions']
        item_list = st.session_state.preprocessed_data['item_list']
        raw_data = st.session_state.preprocessed_data['raw_data']
        
        st.success("✅ Data loaded successfully!")
        
        # Show preprocessing summary
        with st.expander("📊 Preprocessing Summary", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", len(cleaned_data['Order Number'].unique()))
            with col2:
                st.metric("Total Items", len(item_list))
            with col3:
                st.metric("Total Transactions", len(transactions))
            with col4:
                st.metric("Date Range", f"{len(cleaned_data['Order Date'].dt.date.unique())} days")
            
            st.markdown("**Data Quality:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                missing_before = raw_data.isnull().sum().sum()
                st.metric("Missing Values Handled", missing_before)
            with col2:
                duplicates = raw_data.duplicated().sum()
                st.metric("Duplicates Removed", duplicates)
            with col3:
                st.metric("Data Completeness", "100%")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Data Insights",
            "🔍 Association Rules",
            "👥 Customer Segmentation",
            "📊 Decision Support"
        ])
        
        with tab1:
            show_data_insights(cleaned_data, item_list)
        
        with tab2:
            show_association_rules(transactions, item_list)
        
        with tab3:
            show_customer_segmentation(cleaned_data)
        
        with tab4:
            show_decision_support(cleaned_data)
    
    elif uploaded_file is not None:
        try:
            # Load and preprocess data
            with st.spinner("Loading and preprocessing data..."):
                preprocessor = DataPreprocessor()
                raw_data = pd.read_csv(uploaded_file)
                
                # Preprocess
                cleaned_data, transactions, item_list = preprocessor.preprocess(raw_data)
                
                # Store in session state
                st.session_state.data_loaded = True
                st.session_state.preprocessed_data = {
                    'cleaned_data': cleaned_data,
                    'transactions': transactions,
                    'item_list': item_list,
                    'raw_data': raw_data
                }
                
            st.success("✅ Data loaded and preprocessed successfully!")
            
            # Show preprocessing summary
            with st.expander("📊 Preprocessing Summary", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Orders", len(cleaned_data['Order Number'].unique()))
                with col2:
                    st.metric("Total Items", len(item_list))
                with col3:
                    st.metric("Total Transactions", len(transactions))
                with col4:
                    st.metric("Date Range", f"{len(cleaned_data['Order Date'].dt.date.unique())} days")
                
                st.markdown("**Data Quality:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    missing_before = raw_data.isnull().sum().sum()
                    st.metric("Missing Values Handled", missing_before)
                with col2:
                    duplicates = raw_data.duplicated().sum()
                    st.metric("Duplicates Removed", duplicates)
                with col3:
                    st.metric("Data Completeness", "100%")
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Data Insights",
                "🔍 Association Rules",
                "👥 Customer Segmentation",
                "📊 Decision Support"
            ])
            
            with tab1:
                show_data_insights(cleaned_data, item_list)
            
            with tab2:
                show_association_rules(transactions, item_list)
            
            with tab3:
                show_customer_segmentation(cleaned_data)
            
            with tab4:
                show_decision_support(cleaned_data)
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please ensure your CSV file has the correct format with columns: Order Number, Order Date, Item Name, Quantity, Product Price, Total products")
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        
        # Show sample data format
        with st.expander("📋 Expected CSV Format"):
            sample_data = pd.DataFrame({
                'Order Number': [16118, 16118, 16117],
                'Order Date': ['03/08/2019 20:25', '03/08/2019 20:25', '03/08/2019 20:17'],
                'Item Name': ['Plain Papadum', 'King Prawn Balti', 'Plain Naan'],
                'Quantity': [2, 1, 1],
                'Product Price': [0.8, 12.95, 2.6],
                'Total products': [6, 6, 7]
            })
            st.dataframe(sample_data)

def show_data_insights(data, item_list):
    st.markdown("## 📊 Data Insights & Visualization")
    
    theme = get_chart_theme()
    visualizer = DataVisualizer(theme=st.session_state.theme)
    
    # Summary Statistics
    st.markdown("### 📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = (data['Quantity'] * data['Product Price']).sum()
        st.metric("Total Revenue", f"£{total_revenue:,.2f}")
    
    with col2:
        avg_order_value = data.groupby('Order Number').apply(
            lambda x: (x['Quantity'] * x['Product Price']).sum()
        ).mean()
        st.metric("Avg Order Value", f"£{avg_order_value:.2f}")
    
    with col3:
        total_items_sold = data['Quantity'].sum()
        st.metric("Total Items Sold", f"{total_items_sold:,}")
    
    with col4:
        avg_basket_size = data.groupby('Order Number')['Item Name'].count().mean()
        st.metric("Avg Basket Size", f"{avg_basket_size:.1f} items")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top 10 Best-Selling Items")
        fig = visualizer.plot_top_items(data, top_n=10)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Revenue by Item (Top 10)")
        fig = visualizer.plot_revenue_by_item(data, top_n=10)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.markdown("### 📅 Sales Trend Over Time")
    fig = visualizer.plot_sales_trend(data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Item co-occurrence heatmap
    st.markdown("### 🔥 Item Co-occurrence Heatmap")
    st.info("This heatmap shows how frequently items are purchased together")
    
    fig = visualizer.plot_cooccurrence_heatmap(data, top_n=15)
    st.pyplot(fig)
    
    # Category analysis (if applicable)
    st.markdown("### 📦 Item Quantity Distribution")
    fig = visualizer.plot_quantity_distribution(data)
    st.plotly_chart(fig, use_container_width=True)

def show_association_rules(transactions, item_list):
    st.markdown("## 🔍 Association Rule Mining")
    st.markdown("Discover patterns and relationships between items using the Apriori algorithm")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider(
            "Minimum Support",
            min_value=0.01,
            max_value=0.2,
            value=0.02,
            step=0.01,
            help="Minimum frequency of itemset occurrence"
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum confidence for association rules"
        )
    
    with col3:
        min_lift = st.slider(
            "Minimum Lift",
            min_value=1.0,
            max_value=5.0,
            value=1.2,
            step=0.1,
            help="Minimum lift value for rules"
        )
    
    if st.button("🔄 Generate Association Rules", type="primary"):
        with st.spinner("Mining association rules..."):
            miner = AssociationRuleMiner()
            
            try:
                # Generate rules
                rules_df, frequent_itemsets = miner.generate_rules(
                    transactions,
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift
                )
                
                # Store in session state
                st.session_state.rules = rules_df
                
                if len(rules_df) > 0:
                    st.success(f"✅ Found {len(rules_df)} association rules!")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rules", len(rules_df))
                    with col2:
                        st.metric("Frequent Itemsets", len(frequent_itemsets))
                    with col3:
                        avg_confidence = rules_df['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    
                    st.markdown("---")
                    
                    # Display rules
                    st.markdown("### 📋 Association Rules")
                    
                    # Format the dataframe for display
                    display_df = rules_df.copy()
                    display_df['support'] = display_df['support'].apply(lambda x: f"{x:.3f}")
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
                    display_df['lift'] = display_df['lift'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Visualizations
                    st.markdown("### 📊 Rule Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Support vs Confidence scatter
                        theme = get_chart_theme()
                        fig = px.scatter(
                            rules_df,
                            x='support',
                            y='confidence',
                            size='lift',
                            hover_data=['antecedents', 'consequents'],
                            title="Support vs Confidence (size = lift)",
                            labels={'support': 'Support', 'confidence': 'Confidence'},
                            template=theme['template']
                        )
                        fig.update_layout(
                            plot_bgcolor=theme['bg_color'],
                            paper_bgcolor=theme['bg_color'],
                            font_color=theme['text_color'],
                            xaxis=dict(
                                gridcolor=theme['grid_color'],
                                title=dict(font=dict(color=theme['text_color'])),
                                tickfont=dict(color=theme['text_color'])
                            ),
                            yaxis=dict(
                                gridcolor=theme['grid_color'],
                                title=dict(font=dict(color=theme['text_color'])),
                                tickfont=dict(color=theme['text_color'])
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Lift distribution
                        theme = get_chart_theme()
                        fig = px.histogram(
                            rules_df,
                            x='lift',
                            nbins=30,
                            title="Lift Distribution",
                            labels={'lift': 'Lift', 'count': 'Frequency'},
                            template=theme['template']
                        )
                        fig.update_layout(
                            plot_bgcolor=theme['bg_color'],
                            paper_bgcolor=theme['bg_color'],
                            font_color=theme['text_color'],
                            xaxis=dict(
                                gridcolor=theme['grid_color'],
                                title=dict(font=dict(color=theme['text_color'])),
                                tickfont=dict(color=theme['text_color'])
                            ),
                            yaxis=dict(
                                gridcolor=theme['grid_color'],
                                title=dict(font=dict(color=theme['text_color'])),
                                tickfont=dict(color=theme['text_color'])
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Top rules by lift
                    st.markdown("### 🏆 Top 10 Rules by Lift")
                    top_rules = rules_df.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                    
                    for idx, row in top_rules.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 2])
                        with col1:
                            st.markdown(f"**{row['antecedents']}** → **{row['consequents']}**")
                        with col2:
                            st.markdown(f"Lift: **{row['lift']:.2f}**")
                        with col3:
                            st.markdown(f"Confidence: **{row['confidence']:.2%}** | Support: **{row['support']:.3f}**")
                        st.markdown("---")
                    
                else:
                    st.warning("⚠️ No rules found with the current parameters. Try lowering the minimum support or confidence.")
                    
            except Exception as e:
                st.error(f"❌ Error generating rules: {str(e)}")
                st.info("Try adjusting the parameters or check your data format.")

def show_customer_segmentation(data):
    st.markdown("## 👥 Customer Segmentation")
    st.markdown("Segment customers based on their ordering behavior using K-Means clustering")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=4,
            help="Number of customer segments to create"
        )
    
    with col2:
        st.info("Customers are segmented based on: Total Spending, Order Frequency, Average Basket Size, and Item Variety")
    
    if st.button("🔄 Perform Clustering", type="primary"):
        with st.spinner("Clustering customers..."):
            segmentation = CustomerSegmentation()
            
            try:
                # Perform clustering with error handling
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    customer_features, cluster_labels, cluster_centers = segmentation.perform_clustering(
                        data,
                        n_clusters=n_clusters
                    )
                
                # Store in session state
                st.session_state.clusters = {
                    'features': customer_features,
                    'labels': cluster_labels,
                    'centers': cluster_centers
                }
                
                st.success(f"✅ Successfully segmented customers into {n_clusters} clusters!")
                
                # Display cluster summary
                st.markdown("### 📊 Cluster Summary")
                
                customer_features['Cluster'] = cluster_labels
                
                cluster_summary = customer_features.groupby('Cluster').agg({
                    'Total_Spending': ['mean', 'count'],
                    'Order_Frequency': 'mean',
                    'Avg_Basket_Size': 'mean',
                    'Item_Variety': 'mean'
                }).round(2)
                
                cluster_summary.columns = ['Avg Spending (£)', 'Customer Count', 'Avg Orders', 'Avg Basket Size', 'Avg Item Variety']
                st.dataframe(cluster_summary, use_container_width=True)
                
                # Visualizations
                st.markdown("### 📈 Cluster Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 2D scatter plot
                    theme = get_chart_theme()
                    fig = px.scatter(
                        customer_features,
                        x='Total_Spending',
                        y='Order_Frequency',
                        color='Cluster',
                        size='Avg_Basket_Size',
                        hover_data=['Item_Variety'],
                        title="Customer Segments (Spending vs Frequency)",
                        labels={
                            'Total_Spending': 'Total Spending (£)',
                            'Order_Frequency': 'Order Frequency',
                            'Cluster': 'Segment'
                        },
                        color_continuous_scale=theme['colorscale'],
                        template=theme['template']
                    )
                    fig.update_layout(
                        plot_bgcolor=theme['bg_color'],
                        paper_bgcolor=theme['bg_color'],
                        font_color=theme['text_color'],
                        legend=dict(font_color=theme['text_color']),
                        xaxis=dict(
                            gridcolor=theme['grid_color'],
                            title=dict(font=dict(color=theme['text_color'])),
                            tickfont=dict(color=theme['text_color'])
                        ),
                        yaxis=dict(
                            gridcolor=theme['grid_color'],
                            title=dict(font=dict(color=theme['text_color'])),
                            tickfont=dict(color=theme['text_color'])
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Cluster size distribution
                    theme = get_chart_theme()
                    cluster_counts = customer_features['Cluster'].value_counts().sort_index()
                    fig = px.pie(
                        values=cluster_counts.values,
                        names=[f'Cluster {i}' for i in cluster_counts.index],
                        title="Customer Distribution Across Clusters",
                        template=theme['template']
                    )
                    fig.update_layout(
                        plot_bgcolor=theme['bg_color'],
                        paper_bgcolor=theme['bg_color'],
                        font_color=theme['text_color'],
                        legend=dict(font_color=theme['text_color'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster characteristics
                st.markdown("### 🎯 Cluster Characteristics")
                
                for cluster_id in range(n_clusters):
                    cluster_data = customer_features[customer_features['Cluster'] == cluster_id]
                    
                    with st.expander(f"📌 Cluster {cluster_id} - {len(cluster_data)} customers"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_spending = cluster_data['Total_Spending'].mean()
                            st.metric("Avg Spending", f"£{avg_spending:.2f}")
                        
                        with col2:
                            avg_frequency = cluster_data['Order_Frequency'].mean()
                            st.metric("Avg Orders", f"{avg_frequency:.1f}")
                        
                        with col3:
                            avg_basket = cluster_data['Avg_Basket_Size'].mean()
                            st.metric("Avg Basket Size", f"{avg_basket:.1f}")
                        
                        with col4:
                            avg_variety = cluster_data['Item_Variety'].mean()
                            st.metric("Avg Item Variety", f"{avg_variety:.1f}")
                        
                        # Cluster interpretation
                        if avg_spending > customer_features['Total_Spending'].median() and avg_frequency > customer_features['Order_Frequency'].median():
                            st.success("💎 **High-Value Customers**: Frequent orders with high spending")
                        elif avg_spending > customer_features['Total_Spending'].median():
                            st.info("💰 **Big Spenders**: High spending but less frequent")
                        elif avg_frequency > customer_features['Order_Frequency'].median():
                            st.info("🔄 **Frequent Buyers**: Regular customers with moderate spending")
                        else:
                            st.warning("🌱 **Occasional Customers**: Lower frequency and spending")
                
            except Exception as e:
                st.error(f"❌ Error performing clustering: {str(e)}")
                st.info("💡 **Tip**: Try reducing the number of clusters or check if your data has enough records for clustering.")
                with st.expander("🔍 Show Error Details"):
                    st.code(str(e))
                    import traceback
                    st.code(traceback.format_exc())

def show_decision_support(data):
    st.markdown("## 📊 Decision Support Dashboard")
    st.markdown("Key Performance Indicators and Analytics for Business Decisions")
    
    # KPIs
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_orders = data['Order Number'].nunique()
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col2:
        total_revenue = (data['Quantity'] * data['Product Price']).sum()
        st.metric("Total Revenue", f"£{total_revenue:,.2f}")
    
    with col3:
        avg_order_value = data.groupby('Order Number').apply(
            lambda x: (x['Quantity'] * x['Product Price']).sum()
        ).mean()
        st.metric("Avg Order Value", f"£{avg_order_value:.2f}")
    
    with col4:
        unique_items = data['Item Name'].nunique()
        st.metric("Unique Items", unique_items)
    
    with col5:
        avg_items_per_order = data.groupby('Order Number')['Item Name'].count().mean()
        st.metric("Avg Items/Order", f"{avg_items_per_order:.1f}")
    
    st.markdown("---")
    
    # Peak hours analysis
    st.markdown("### ⏰ Peak Hours Analysis")
    
    theme = get_chart_theme()
    data['Hour'] = data['Order Date'].dt.hour
    hourly_orders = data.groupby('Hour')['Order Number'].nunique().reset_index()
    hourly_orders.columns = ['Hour', 'Orders']
    
    fig = px.bar(
        hourly_orders,
        x='Hour',
        y='Orders',
        title="Orders by Hour of Day",
        labels={'Hour': 'Hour of Day', 'Orders': 'Number of Orders'},
        color='Orders',
        color_continuous_scale=theme['colorscale'],
        template=theme['template']
    )
    fig.update_layout(
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['bg_color'],
        font_color=theme['text_color'],
        xaxis=dict(
            gridcolor=theme['grid_color'],
            title=dict(font=dict(color=theme['text_color'])),
            tickfont=dict(color=theme['text_color'])
        ),
        yaxis=dict(
            gridcolor=theme['grid_color'],
            title=dict(font=dict(color=theme['text_color'])),
            tickfont=dict(color=theme['text_color'])
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    st.markdown("### 📅 Day of Week Analysis")
    
    theme = get_chart_theme()
    data['DayOfWeek'] = data['Order Date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    daily_revenue = data.groupby('DayOfWeek').apply(
        lambda x: (x['Quantity'] * x['Product Price']).sum()
    ).reindex(day_order)
    
    fig = px.line(
        x=daily_revenue.index,
        y=daily_revenue.values,
        title="Revenue by Day of Week",
        labels={'x': 'Day', 'y': 'Revenue (£)'},
        markers=True,
        template=theme['template']
    )
    fig.update_traces(line_color=theme['primary_color'], marker_size=8)
    fig.update_layout(
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['bg_color'],
        font_color=theme['text_color'],
        xaxis=dict(
            gridcolor=theme['grid_color'],
            title=dict(font=dict(color=theme['text_color'])),
            tickfont=dict(color=theme['text_color'])
        ),
        yaxis=dict(
            gridcolor=theme['grid_color'],
            title=dict(font=dict(color=theme['text_color'])),
            tickfont=dict(color=theme['text_color'])
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Item performance matrix
    st.markdown("### 📈 Item Performance Matrix")
    
    theme = get_chart_theme()
    item_performance = data.groupby('Item Name').agg({
        'Quantity': 'sum',
        'Order Number': 'nunique',
        'Product Price': 'first'
    }).reset_index()
    
    item_performance['Revenue'] = item_performance['Quantity'] * item_performance['Product Price']
    item_performance = item_performance.nlargest(20, 'Revenue')
    
    fig = px.scatter(
        item_performance,
        x='Order Number',
        y='Revenue',
        size='Quantity',
        hover_data=['Item Name'],
        title="Item Performance (Top 20 by Revenue)",
        labels={'Order Number': 'Number of Orders', 'Revenue': 'Total Revenue (£)'},
        color='Revenue',
        color_continuous_scale=theme['colorscale'],
        template=theme['template']
    )
    fig.update_layout(
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['bg_color'],
        font_color=theme['text_color'],
        xaxis=dict(
            gridcolor=theme['grid_color'],
            title=dict(font=dict(color=theme['text_color'])),
            tickfont=dict(color=theme['text_color'])
        ),
        yaxis=dict(
            gridcolor=theme['grid_color'],
            title=dict(font=dict(color=theme['text_color'])),
            tickfont=dict(color=theme['text_color'])
        ),
        coloraxis_colorbar=dict(title_font_color=theme['text_color'], tickfont_color=theme['text_color'])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations for owner
    st.markdown("### 💡 Business Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Top Performers")
        top_items = data.groupby('Item Name')['Quantity'].sum().nlargest(5)
        for idx, (item, qty) in enumerate(top_items.items(), 1):
            st.markdown(f"{idx}. **{item}** - {qty} units sold")
    
    with col2:
        st.markdown("#### 📉 Underperformers")
        bottom_items = data.groupby('Item Name')['Quantity'].sum().nsmallest(5)
        for idx, (item, qty) in enumerate(bottom_items.items(), 1):
            st.markdown(f"{idx}. **{item}** - {qty} units sold")
    
    # Peak time recommendation
    peak_hour = hourly_orders.loc[hourly_orders['Orders'].idxmax(), 'Hour']
    st.info(f"🕐 **Peak Hour**: {peak_hour}:00 - Consider increasing staff during this time")
    
    # Revenue insights
    best_day = daily_revenue.idxmax()
    st.success(f"📅 **Best Day**: {best_day} - Focus marketing efforts on this day")

def show_customer_recommendations():
    st.title("👤 Customer Recommendations")
    st.markdown("### Get Personalized Food Recommendations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data in the Owner Dashboard first to enable recommendations")
        return
    
    data = st.session_state.preprocessed_data['cleaned_data']
    item_list = st.session_state.preprocessed_data['item_list']
    transactions = st.session_state.preprocessed_data['transactions']
    
    # Menu Display
    st.markdown("## 🍽️ Menu")
    
    # Get unique items with prices
    menu_items = data.groupby('Item Name').agg({
        'Product Price': 'first'
    }).reset_index()
    menu_items = menu_items.sort_values('Item Name')
    
    # Search and filter
    search_term = st.text_input("🔍 Search menu items", "")
    
    if search_term:
        menu_items = menu_items[menu_items['Item Name'].str.contains(search_term, case=False)]
    
    # Display menu in columns
    st.markdown("### Available Items")
    
    cols = st.columns(3)
    for idx, row in menu_items.iterrows():
        col_idx = idx % 3
        with cols[col_idx]:
            st.markdown(f"**{row['Item Name']}**")
            st.markdown(f"£{row['Product Price']:.2f}")
            st.markdown("---")
    
    st.markdown("---")
    
    # Order Input
    st.markdown("## 🛒 Your Current Order")
    
    selected_items = st.multiselect(
        "Select items you want to order",
        options=sorted(item_list),
        help="Select one or more items to get recommendations"
    )
    
    if selected_items:
        # Display selected items
        st.markdown("### Selected Items:")
        
        selected_df = menu_items[menu_items['Item Name'].isin(selected_items)]
        total_price = selected_df['Product Price'].sum()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            for item in selected_items:
                price = menu_items[menu_items['Item Name'] == item]['Product Price'].values[0]
                st.markdown(f"- **{item}** - £{price:.2f}")
        
        with col2:
            st.metric("Total", f"£{total_price:.2f}")
        
        st.markdown("---")
        
        # Generate Recommendations
        st.markdown("## ✨ Recommended For You")
        
        if st.button("🎯 Get Recommendations", type="primary"):
            with st.spinner("Generating personalized recommendations..."):
                recommender = RecommendationEngine()
                
                try:
                    # Generate recommendations using association rules
                    if st.session_state.rules is not None and len(st.session_state.rules) > 0:
                        recommendations = recommender.get_recommendations(
                            selected_items,
                            st.session_state.rules,
                            transactions,
                            top_n=5
                        )
                    else:
                        # Fallback to frequency-based recommendations
                        recommendations = recommender.get_frequency_based_recommendations(
                            selected_items,
                            transactions,
                            top_n=5
                        )
                    
                    if recommendations:
                        st.success(f"✅ Found {len(recommendations)} recommendations for you!")
                        
                        st.markdown("### 🎯 You might also like:")
                        
                        for idx, (item, score) in enumerate(recommendations, 1):
                            if item not in selected_items:
                                item_price = menu_items[menu_items['Item Name'] == item]['Product Price'].values
                                
                                if len(item_price) > 0:
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        st.markdown(f"**{idx}. {item}**")
                                    
                                    with col2:
                                        st.markdown(f"£{item_price[0]:.2f}")
                                    
                                    with col3:
                                        confidence = score * 100
                                        st.markdown(f"⭐ {confidence:.0f}% match")
                                    
                                    st.markdown("---")
                        
                        # Why these recommendations
                        st.info(
                            "💡 **How we recommend:**\n\n"
                            "These suggestions are based on what other customers "
                            "frequently ordered together with your selected items. "
                            "The match percentage indicates how often these items "
                            "appear together in orders."
                        )
                        
                    else:
                        st.warning("No specific recommendations found. Try selecting different items!")
                        
                        # Show popular items as fallback
                        st.markdown("### 🔥 Popular Items")
                        popular_items = data.groupby('Item Name')['Quantity'].sum().nlargest(5)
                        
                        for idx, (item, qty) in enumerate(popular_items.items(), 1):
                            if item not in selected_items:
                                item_price = menu_items[menu_items['Item Name'] == item]['Product Price'].values[0]
                                st.markdown(f"{idx}. **{item}** - £{item_price:.2f} (Sold {qty} times)")
                
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")
                    
    else:
        st.info("👆 Select items from the menu to get personalized recommendations")
        
        # Show popular items
        st.markdown("### 🔥 Most Popular Items")
        
        popular_items = data.groupby('Item Name').agg({
            'Quantity': 'sum',
            'Product Price': 'first'
        }).nlargest(10, 'Quantity')
        
        for idx, (item, row) in enumerate(popular_items.iterrows(), 1):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{idx}. {item}**")
            
            with col2:
                st.markdown(f"£{row['Product Price']:.2f}")
            
            with col3:
                st.markdown(f"🔥 {row['Quantity']} sold")

if __name__ == "__main__":
    main()

