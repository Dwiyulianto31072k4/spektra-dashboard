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
fifgroup_logo = "data:image/png;base64,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"
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
                    processed_data = data.copy()  # mempertahankan semua kolo
                    
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
                    # Add Multi-Transaction_Customer column
                    processed_data["Multi-Transaction_Customer"] = processed_data["TOTAL_PRODUCT_MPF"].apply(lambda x: 1 if x > 1 else 0)

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


def segmentation_analysis():
    st.markdown('<p class="section-title">Segmentation Analysis (RFM + K-Means)</p>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please upload and preprocess your data first.")
        return

    df = st.session_state.data.copy()

    # Pastikan kolom 'Multi-Transaction_Customer' ada
    if 'Multi-Transaction_Customer' not in df.columns and 'TOTAL_PRODUCT_MPF' in df.columns:
        df["Multi-Transaction_Customer"] = df["TOTAL_PRODUCT_MPF"].apply(lambda x: 1 if x > 1 else 0)


    # Pilih kolom untuk RFM
    st.markdown("### Pilih Kolom RFM dan Parameter Clustering")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency_col = st.selectbox("Recency (tanggal transaksi terakhir)", df.columns, index=df.columns.get_loc("LAST_MPF_DATE") if "LAST_MPF_DATE" in df.columns else 0)
    with col2:
        freq_col = st.selectbox("Frequency (jumlah produk)", df.columns, index=df.columns.get_loc("TOTAL_PRODUCT_MPF") if "TOTAL_PRODUCT_MPF" in df.columns else 0)
    with col3:
        mon_col = st.selectbox("Monetary (total amount)", df.columns, index=df.columns.get_loc("TOTAL_AMOUNT_MPF") if "TOTAL_AMOUNT_MPF" in df.columns else 0)

    cluster_k = st.slider("Pilih jumlah cluster (k)", min_value=2, max_value=10, value=4)

    if st.button("Lakukan Segmentasi"):
        with st.spinner("Memproses segmentasi..."):
            now = df[recency_col].max() + pd.Timedelta(days=1)
            st.write("Kolom tersedia di df saat segmentasi:", df.columns.tolist())
            rfm = df.groupby('CUST_NO').agg({
                recency_col: 'max',
                freq_col: 'sum',
                mon_col: 'sum',
                'BIRTH_DATE': 'max',
                'Multi-Transaction_Customer': 'max' if 'Multi-Transaction_Customer' in df.columns else lambda x: 0
            }).reset_index()

            rfm['Recency'] = (now - rfm[recency_col]).dt.days
            rfm['Frequency'] = rfm[freq_col]
            rfm['Monetary'] = rfm[mon_col]
            rfm['Usia'] = 2024 - rfm['BIRTH_DATE'].dt.year
            rfm['Usia_Segment'] = rfm['Usia'].apply(lambda x: 1 if 25 <= x <= 50 else 0)
            rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
            rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

            features = ['Recency', 'Frequency_log', 'Monetary_log', 'Multi-Transaction_Customer', 'Usia_Segment']
            rfm_scaled = rfm[features].apply(zscore)

            # Tangani missing value di rfm_scaled
            if rfm_scaled.isnull().values.any():
                st.warning("Terdapat data kosong (NaN) dalam data clustering. Baris-baris tersebut akan dihapus.")
                rows_before = rfm_scaled.shape[0]
                rfm_scaled = rfm_scaled.dropna()
                rfm = rfm.loc[rfm_scaled.index]
                rows_after = rfm_scaled.shape[0]
                st.info(f"{rows_before - rows_after} baris dihapus karena mengandung NaN.")

            # Lakukan clustering
            kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init='auto')
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)


            cluster_score = rfm.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency_log': 'mean',
                'Monetary_log': 'mean',
                'Multi-Transaction_Customer': 'mean',
                'Usia_Segment': 'mean'
            }).reset_index()
            cluster_score['Recency_Score'] = 1 / (cluster_score['Recency'] + 1)
            cluster_score['Total_Score'] = (
                cluster_score['Recency_Score'] +
                cluster_score['Frequency_log'] +
                cluster_score['Monetary_log'] +
                cluster_score['Multi-Transaction_Customer'] +
                cluster_score['Usia_Segment']
            )
            n_invited = int(0.5 * cluster_k)
            top_clusters = cluster_score.sort_values('Total_Score', ascending=False).head(n_invited)['Cluster'].tolist()
            rfm['Layak_Diundang'] = rfm['Cluster'].apply(lambda x: '✅ Diundang' if x in top_clusters else '❌ Tidak')

            st.session_state.segmented_data = rfm
            st.session_state.segmentation_completed = True

            st.success("Segmentasi selesai!")
            st.dataframe(rfm[['CUST_NO', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Layak_Diundang']].head(10))

            fig = px.scatter(
                rfm, x='Recency', y='Monetary_log',
                color='Cluster',
                title="Visualisasi Cluster Berdasarkan Recency & Monetary",
                hover_data=['CUST_NO', 'Frequency', 'Monetary', 'Layak_Diundang']
            )
            st.plotly_chart(fig, use_container_width=True)


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
elif selected_page == "Segmentation Analysis":
    segmentation_analysis()




# Footer
st.markdown("""
<div class="footer">
    <p>Developed with ❤️ by the Data Science Team at FIFGROUP · Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
