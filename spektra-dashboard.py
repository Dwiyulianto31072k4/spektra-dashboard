import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import random
import string
import os
from PIL import Image
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="SPEKTRA Customer Segmentation & Promo App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        color: #003366;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 2px solid #003366;
    }
    .section-title {
        color: #003366;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid #ddd;
    }
    .sidebar-title {
        color: #003366;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #003366;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #004080;
    }
    .box-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .metric-box {
        background-color: #003366;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .card-title {
        font-weight: bold;
        color: #003366;
        margin-bottom: 5px;
    }
    .card-text {
        color: #333;
    }
    .highlight {
        background-color: #FFF9C4;
        padding: 3px;
        border-radius: 3px;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Create a directory for temporary files if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Add FIFGROUP logo
fifgroup_logo = "https://www.fifgroup.co.id/uploads/images/logo/20221028_142514_logo-FIFGROUPPUTIH.png"
st.sidebar.image(fifgroup_logo, width=200)

# Sidebar title
st.sidebar.markdown('<p class="sidebar-title">SPEKTRA Customer Segmentation & Promo Mapping</p>', unsafe_allow_html=True)

# Main application title
st.markdown('<p class="main-title">SPEKTRA Customer Segmentation & Promo Mapping</p>', unsafe_allow_html=True)

# Add navigation
st.sidebar.markdown("### Navigation")
pages = ["Upload & Preprocessing", "Exploratory Data Analysis", "Segmentation Analysis", "Promo Mapping", "Dashboard", "Export & Documentation"]
selected_page = st.sidebar.radio("Go to", pages)

# Initialize session state if not already initialized
if 'data' not in st.session_state:
    st.session_state.data = None
if 'segmented_data' not in st.session_state:
    st.session_state.segmented_data = None
if 'promo_mapped_data' not in st.session_state:
    st.session_state.promo_mapped_data = None
if 'rfm_data' not in st.session_state:
    st.session_state.rfm_data = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'eda_completed' not in st.session_state:
    st.session_state.eda_completed = False
if 'segmentation_completed' not in st.session_state:
    st.session_state.segmentation_completed = False
if 'promo_mapping_completed' not in st.session_state:
    st.session_state.promo_mapping_completed = False

# Function to handle data upload and preprocessing
def upload_and_preprocess():
    st.markdown('<p class="section-title">Upload Customer Data</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Excel file with customer data", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name
        
        try:
            data = pd.read_excel(uploaded_file)
            st.success(f"File '{uploaded_file.name}' successfully loaded with {data.shape[0]} rows and {data.shape[1]} columns!")
            
            # Display data preview
            st.markdown('<p class="section-title">Data Preview</p>', unsafe_allow_html=True)
            st.dataframe(data.head())
            
            # Data preprocessing options
            st.markdown('<p class="section-title">Data Preprocessing</p>', unsafe_allow_html=True)
            with st.expander("Data Preprocessing Options", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Date Columns")
                    date_cols = st.multiselect(
                        "Select date columns to convert",
                        options=data.columns.tolist(),
                        default=[col for col in data.columns if 'DATE' in col]
                    )
                
                with col2:
                    st.subheader("Missing Values Strategy")
                    handle_missing = st.checkbox("Handle missing values", value=True)
                    missing_strategy = st.radio(
                        "Strategy for numeric columns",
                        options=["Mean", "Median", "Zero", "None"],
                        index=1,
                        disabled=not handle_missing
                    )
                    
                    categorical_strategy = st.radio(
                        "Strategy for categorical columns",
                        options=["Mode", "Fill with 'Unknown'", "None"],
                        index=0,
                        disabled=not handle_missing
                    )
            
            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    # Make a copy of the data
                    processed_data = data.copy()
                    
                    # Convert date columns
                    for col in date_cols:
                        processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                    
                    # Calculate age if birth date column exists
                    if 'BIRTH_DATE' in processed_data.columns:
                        current_year = datetime.datetime.now().year
                        processed_data['Usia'] = current_year - processed_data['BIRTH_DATE'].dt.year
                    
                    # Handle missing values if enabled
                    if handle_missing:
                        numeric_cols = processed_data.select_dtypes(include=['int64', 'float64']).columns
                        categorical_cols = processed_data.select_dtypes(include=['object']).columns
                        
                        for col in numeric_cols:
                            if processed_data[col].isnull().sum() > 0:
                                if missing_strategy == "Mean":
                                    processed_data[col].fillna(processed_data[col].mean(), inplace=True)
                                elif missing_strategy == "Median":
                                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
                                elif missing_strategy == "Zero":
                                    processed_data[col].fillna(0, inplace=True)
                        
                        for col in categorical_cols:
                            if processed_data[col].isnull().sum() > 0:
                                if categorical_strategy == "Mode":
                                    processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
                                elif categorical_strategy == "Fill with 'Unknown'":
                                    processed_data[col].fillna("Unknown", inplace=True)
                                    
                    # Add data processing timestamp
                    processed_data['PROCESSING_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Success message and preview of results
                    st.success("Data preprocessing completed!")
                    st.markdown('<p class="section-title">Preprocessing Results</p>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Missing values before preprocessing:")
                        st.dataframe(data.isnull().sum())
                    
                    with col2:
                        st.write("Missing values after preprocessing:")
                        st.dataframe(processed_data.isnull().sum())
                    
                    st.write("Processed data preview:")
                    st.dataframe(processed_data.head())
                    
                    # Store processed data in session state and save to temp file
                    st.session_state.data = processed_data
                    processed_data.to_excel("temp/processed_data.xlsx", index=False)
                    
                    st.markdown("### Next Steps")
                    st.info("You can now proceed to the Exploratory Data Analysis section to visualize and understand your data patterns.")
                    
                    st.session_state.eda_completed = True
        
        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please check your file and try again.")
    else:
        if st.button("Use Example Data"):
            example_data = create_example_data()
            st.success("Example data loaded successfully!")
            
            st.session_state.data = example_data
            st.session_state.uploaded_file_name = "example_data.xlsx"
            example_data.to_excel("temp/processed_data.xlsx", index=False)
            
            st.markdown('<p class="section-title">Example Data Preview</p>', unsafe_allow_html=True)
            st.dataframe(example_data.head())
            
            st.session_state.eda_completed = True
            st.markdown("### Next Steps")
            st.info("You can now proceed to the Exploratory Data Analysis section to visualize and understand your data patterns.")

# Function to create example data
def create_example_data(n=500):
    np.random.seed(42)
    cust_ids = [f"1010000{i:05d}" for i in range(1, n+1)]
    product_categories = ['GADGET', 'ELECTRONIC', 'FURNITURE', 'OTHER']
    product_weights = [0.4, 0.3, 0.2, 0.1]
    ppc_types = ['MPF', 'REFI', 'NMC']
    genders = ['M', 'F']
    education = ['SD', 'SMP', 'SMA', 'S1', 'S2', 'S3']
    house_status = ['H01', 'H02', 'H03', 'H04', 'H05']
    marital_status = ['M', 'S', 'D']
    areas = ['JATA 1', 'JATA 2', 'JATA 3', 'SULSEL', 'KALSEL', 'SUMSEL']
    
    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2023-12-31')
    date_range = (end_date - start_date).days
    
    df = pd.DataFrame({
        'CUST_NO': cust_ids,
        'FIRST_PPC': np.random.choice(ppc_types, size=n, p=[0.6, 0.3, 0.1]),
        'FIRST_PPC_DATE': [start_date + pd.Timedelta(days=np.random.randint(0, date_range)) for _ in range(n)],
        'FIRST_MPF_DATE': [start_date + pd.Timedelta(days=np.random.randint(0, date_range)) for _ in range(n)],
        'LAST_MPF_DATE': [start_date + pd.Timedelta(days=np.random.randint(0, date_range)) for _ in range(n)],
        'JMH_CON_SBLM_MPF': np.random.randint(0, 5, size=n),
        'MAX_MPF_AMOUNT': np.random.randint(1000000, 20000000, size=n),
        'MIN_MPF_AMOUNT': np.random.randint(1000000, 10000000, size=n),
        'AVG_MPF_INST': np.random.randint(100000, 2000000, size=n),
        'MPF_CATEGORIES_TAKEN': [np.random.choice(product_categories, p=product_weights) for _ in range(n)],
        'LAST_MPF_PURPOSE': [np.random.choice(product_categories, p=product_weights) for _ in range(n)],
        'LAST_MPF_AMOUNT': np.random.randint(1000000, 15000000, size=n),
        'LAST_MPF_TOP': np.random.choice([6, 9, 12, 18, 24, 36], size=n),
        'LAST_MPF_INST': np.random.randint(100000, 1500000, size=n),
        'JMH_PPC': np.random.randint(1, 6, size=n),
        'PRINCIPAL': np.random.randint(2000000, 20000000, size=n),
        'GRS_DP': np.random.randint(0, 5000000, size=n),
        'BIRTH_DATE': [pd.Timestamp('1970-01-01') + pd.Timedelta(days=np.random.randint(0, 365*40)) for _ in range(n)],
        'CUST_SEX': np.random.choice(genders, size=n),
        'EDU_TYPE': np.random.choice(education, size=n, p=[0.05, 0.1, 0.4, 0.35, 0.08, 0.02]),
        'OCPT_CODE': np.random.randint(1, 25, size=n),
        'HOUSE_STAT': np.random.choice(house_status, size=n),
        'MARITAL_STAT': np.random.choice(marital_status, size=n, p=[0.7, 0.2, 0.1]),
        'NO_OF_DEPEND': np.random.randint(0, 5, size=n),
        'BRANCH_ID': np.random.randint(10000, 99999, size=n),
        'AREA': np.random.choice(areas, size=n),
        'TOTAL_AMOUNT_MPF': np.random.randint(1000000, 50000000, size=n),
        'TOTAL_PRODUCT_MPF': np.random.randint(1, 5, size=n)
    })
    
    # Adjust values for realism
    for i in range(len(df)):
        if df.loc[i, 'MIN_MPF_AMOUNT'] > df.loc[i, 'MAX_MPF_AMOUNT']:
            df.loc[i, 'MIN_MPF_AMOUNT'], df.loc[i, 'MAX_MPF_AMOUNT'] = df.loc[i, 'MAX_MPF_AMOUNT'], df.loc[i, 'MIN_MPF_AMOUNT']
    
    for i in range(len(df)):
        if df.loc[i, 'FIRST_MPF_DATE'] > df.loc[i, 'LAST_MPF_DATE']:
            df.loc[i, 'FIRST_MPF_DATE'], df.loc[i, 'LAST_MPF_DATE'] = df.loc[i, 'LAST_MPF_DATE'], df.loc[i, 'FIRST_MPF_DATE']
        if df.loc[i, 'FIRST_PPC_DATE'] > df.loc[i, 'FIRST_MPF_DATE']:
            df.loc[i, 'FIRST_PPC_DATE'] = df.loc[i, 'FIRST_MPF_DATE'] - pd.Timedelta(days=np.random.randint(1, 100))
    
    df['Usia'] = 2024 - df['BIRTH_DATE'].dt.year
    return df

# Function for Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload and preprocess your data first.")
        return
    
    data = st.session_state.data
    st.markdown("### Select Analysis Options")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Customer Demographics", "Transaction Patterns"])
    
    with tab1:
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            selected_num_col = st.selectbox(
                "Select numeric column for distribution analysis", 
                options=numeric_cols,
                index=numeric_cols.index('TOTAL_AMOUNT_MPF') if 'TOTAL_AMOUNT_MPF' in numeric_cols else 0
            )
            log_transform = st.checkbox("Apply log transformation", value=True)
            fig = px.histogram(
                data, 
                x=selected_num_col, 
                title=f"Distribution of {selected_num_col}",
                nbins=50,
                color_discrete_sequence=['#003366']
            )
            if log_transform and data[selected_num_col].min() > 0:
                fig = px.histogram(
                    data, 
                    x=np.log1p(data[selected_num_col]), 
                    title=f"Log Distribution of {selected_num_col}",
                    nbins=50,
                    color_discrete_sequence=['#003366']
                )
                fig.update_layout(xaxis_title=f"Log({selected_num_col})")
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="white",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist() + (['Usia_Kategori'] if 'Usia_Kategori' in data.columns else [])
            if len(categorical_cols) > 0:
                selected_cat_col = st.selectbox(
                    "Select categorical column for distribution analysis", 
                    options=categorical_cols,
                    index=categorical_cols.index('MPF_CATEGORIES_TAKEN') if 'MPF_CATEGORIES_TAKEN' in categorical_cols else 0
                )
                value_counts = data[selected_cat_col].value_counts().reset_index()
                value_counts.columns = [selected_cat_col, 'Count']
                top_n = st.slider("Show top N categories", min_value=5, max_value=30, value=10)
                fig = px.bar(
                    value_counts.head(top_n), 
                    x=selected_cat_col, 
                    y='Count',
                    title=f"Top {top_n} values of {selected_cat_col}",
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available.")
    
    with tab2:
        st.subheader("Customer Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Usia' in data.columns:
                if 'Usia_Kategori' not in data.columns:
                    bins = [0, 25, 35, 45, 55, 100]
                    labels = ['<25', '25-35', '35-45', '45-55', '55+']
                    data['Usia_Kategori'] = pd.cut(data['Usia'], bins=bins, labels=labels, right=False)
                age_counts = data['Usia_Kategori'].value_counts().reset_index()
                age_counts.columns = ['Age Group', 'Count']
                fig = px.pie(
                    age_counts, 
                    values='Count', 
                    names='Age Group',
                    title="Customer Age Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age data not available.")
        
        with col2:
            if 'CUST_SEX' in data.columns:
                gender_counts = data['CUST_SEX'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']
                fig = px.pie(
                    gender_counts, 
                    values='Count', 
                    names='Gender',
                    title="Customer Gender Distribution",
                    hole=0.4,
                    color_discrete_sequence=['#003366', '#66b3ff']
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Gender data not available.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'EDU_TYPE' in data.columns:
                edu_counts = data['EDU_TYPE'].value_counts().reset_index()
                edu_counts.columns = ['Education', 'Count']
                fig = px.bar(
                    edu_counts, 
                    x='Education', 
                    y='Count',
                    title="Customer Education Level",
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Education data not available.")
        
        with col2:
            if 'MARITAL_STAT' in data.columns:
                marital_counts = data['MARITAL_STAT'].value_counts().reset_index()
                marital_counts.columns = ['Marital Status', 'Count']
                fig = px.pie(
                    marital_counts, 
                    values='Count', 
                    names='Marital Status',
                    title="Customer Marital Status",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Marital status data not available.")
    
    with tab3:
        st.subheader("Transaction Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'MPF_CATEGORIES_TAKEN' in data.columns:
                product_counts = data['MPF_CATEGORIES_TAKEN'].value_counts().reset_index()
                product_counts.columns = ['Product Category', 'Count']
                fig = px.pie(
                    product_counts, 
                    values='Count', 
                    names='Product Category',
                    title="Product Category Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Product category data not available.")
        
        with col2:
            if 'TOTAL_AMOUNT_MPF' in data.columns and 'Usia' in data.columns:
                fig = px.scatter(
                    data, 
                    x='Usia', 
                    y='TOTAL_AMOUNT_MPF',
                    title="Total Transaction Amount vs Age",
                    color='TOTAL_AMOUNT_MPF',
                    color_continuous_scale=px.colors.sequential.Blues,
                    opacity=0.7
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Transaction amount or age data not available.")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'TOTAL_PRODUCT_MPF' in data.columns:
                product_counts = data['TOTAL_PRODUCT_MPF'].value_counts().reset_index()
                product_counts.columns = ['Products Purchased', 'Count']
                product_counts = product_counts.sort_values('Products Purchased')
                fig = px.bar(
                    product_counts, 
                    x='Products Purchased', 
                    y='Count',
                    title="Number of Products Purchased Distribution",
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Number of products data not available.")
        
        # Additional visualization: Average Transaction Amount by Product Category
        with st.container():
            if 'TOTAL_AMOUNT_MPF' in data.columns and 'MPF_CATEGORIES_TAKEN' in data.columns:
                product_amount = data.groupby('MPF_CATEGORIES_TAKEN')['TOTAL_AMOUNT_MPF'].mean().reset_index()
                product_amount.columns = ['Product Category', 'Average Amount']
                fig = px.bar(
                    product_amount, 
                    x='Product Category', 
                    y='Average Amount',
                    title="Average Transaction Amount by Product Category",
                    color='Average Amount',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Required columns not available for this visualization.")

# Navigation routing
if selected_page == "Upload & Preprocessing":
    upload_and_preprocess()
elif selected_page == "Exploratory Data Analysis":
    exploratory_data_analysis()
# Additional pages like "Segmentation Analysis", "Promo Mapping", "Dashboard", and "Export & Documentation"
# should be implemented similarly.

# Footer
st.markdown("""
<div class="footer">
    <p>Developed with ❤️ by the Data Science Team at FIFGROUP · Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
