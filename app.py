import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="CarAdvisor AI", layout="wide")

# Style
st.markdown("""
<style>
body { background-color: #0f172a; }

.hero {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 180px;
}
.hero h1 { color: white; font-size: 42px; margin-bottom: 5px; }
.hero p  { color: #9ca3af; font-size: 16px; }

.car-card {
    background: #1f2937;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 8px;
}
.car-card h4 { color: #00E5FF; margin: 0 0 8px 0; }
.car-card p  { color: white; margin: 2px 0; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <h1>🚗 CarAdvisor AI</h1>
    <p>Find Your Perfect Car</p>
</div>
""", unsafe_allow_html=True)


# Helper: clean numeric strings
def clean_data(value):
    """Strips $ and commas, averages ranges like '200-300', returns np.nan if no number found."""
    value   = str(value).replace(',', '').replace('$', '')
    numbers = re.findall(r'\d+\.?\d*', value)
    if len(numbers) == 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    return float(numbers[0]) if numbers else np.nan


# Helper: brand image lookup
def get_image(name):
    name = name.lower().replace(' ', '')
    mapping = {
        'jaguarlandrover': 'jaguar',
        'tatamotors'     : 'tata',
        'marutisuzuki'   : 'marutisuzuki',
        'suzuki'         : 'marutisuzuki',
        'maruti'         : 'marutisuzuki',
    }
    name     = mapping.get(name, name)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ['png', 'jpg', 'jpeg']:
        path = os.path.join(base_dir, 'carimage', f'{name}.{ext}')
        if os.path.exists(path):
            return path
    return None


# Load data and train model — cached so it runs only once per session
@st.cache_resource(show_spinner='Loading data and training model...')
def load_and_train():
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'Cars Datasets 2025.csv')

    df = pd.read_csv(dataset_path, encoding='latin1')
    df.columns = df.columns.str.strip()
    df['Company Names'] = df['Company Names'].astype(str).str.strip().str.upper()
    df['Fuel Types']    = df['Fuel Types'].astype(str).str.strip().str.upper()

    # Consolidate brand name variants
    df['Company Names'] = df['Company Names'].replace({
        'JAGUAR LAND ROVER': 'JAGUAR',
        'TATA MOTORS'      : 'TATA',
    })

    # Clean numeric columns
    for col in ['Cars Prices', 'Seats', 'HorsePower', 'Total Speed', 'CC/Battery Capacity']:
        df[col] = df[col].apply(clean_data)

    df.dropna(inplace=True)

    # Remove extreme price outliers (1st-99th percentile)
    Q1 = df['Cars Prices'].quantile(0.01)
    Q3 = df['Cars Prices'].quantile(0.99)
    df = df[(df['Cars Prices'] >= Q1) & (df['Cars Prices'] <= Q3)].copy()

    # Log-transform price to reduce skew
    df['Log Price'] = np.log1p(df['Cars Prices'])

    # Encode company as integer for model input
    df['Company Encoded'] = df['Company Names'].astype('category').cat.codes

    feature_cols = ['Seats', 'HorsePower', 'Total Speed', 'CC/Battery Capacity', 'Company Encoded']
    X = df[feature_cols]
    y = df['Log Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # SVM is the best-performing model (matches notebook conclusion)
    model = SVR(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train_scaled, y_train)

    r2_global = r2_score(y_test, model.predict(X_test_scaled))

    return df, scaler, model, r2_global


df, scaler, model, r2_global = load_and_train()


# Sidebar Inputs
st.sidebar.header('🔧 Car Preferences')

seats  = st.sidebar.selectbox('Seats', [2, 4, 5, 7])
hp     = st.sidebar.slider('HorsePower', 50, 1500, 300)
speed  = st.sidebar.slider('Top Speed (km/h)', 100, 400, 200)
engine = st.sidebar.slider('Engine CC / Battery kWh', 800, 8000, 2000)
brand  = st.sidebar.selectbox('Brand', sorted(df['Company Names'].unique()))


# Input Validation
brand_df = df[df['Company Names'] == brand].copy()

if brand_df.empty:
    st.error(f'No data found for brand: {brand}. Please select a different brand.')
    st.stop()


# Prediction — predict in log-price space, convert back to USD
brand_code = brand_df['Company Encoded'].iloc[0]
input_data = np.array([[seats, hp, speed, engine, brand_code]])
log_pred   = model.predict(scaler.transform(input_data))[0]
pred_price = np.expm1(log_pred)

st.subheader('💰 Predicted Price')
st.success(f'${int(pred_price):,}')


# Similar Cars — closest to predicted price within selected brand
brand_df['diff'] = abs(brand_df['Cars Prices'] - pred_price)
top3 = brand_df.sort_values('diff').head(3)

avg_diff   = top3['diff'].mean()
confidence = max(0.0, 100 - (avg_diff / pred_price) * 100) if pred_price > 0 else 0.0


# Top Recommended Cards
st.markdown('<h3 style="text-align:center;color:gold;">🏆 Top Recommended Cars</h3>',
            unsafe_allow_html=True)

if top3.empty:
    st.warning(f'No cars found for brand: {brand}. Try a different brand.')
else:
    cols = st.columns(len(top3))

    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            img = get_image(row['Company Names'])
            if img:
                st.image(img, use_container_width=True)

            st.markdown(f"""
            <div class="car-card">
                <h4>{row['Cars Names']}</h4>
                <p>💰 ${int(row['Cars Prices']):,}</p>
                <p>⚡ {int(row['HorsePower'])} HP</p>
                <p>🚗 {int(row['Total Speed'])} km/h</p>
            </div>
            """, unsafe_allow_html=True)


# Price Comparison Chart
if not top3.empty:
    st.subheader('📊 Price Comparison')
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(top3['Cars Names'], top3['Cars Prices'], color=['#00E5FF', '#00BCD4', '#0097A7'])
    ax.set_ylabel('Price (USD)')
    ax.axhline(pred_price, color='gold', linestyle='--', label=f'Predicted: ${int(pred_price):,}')
    ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Feature Correlation Heatmap
st.subheader('🔥 Feature Correlation')
numeric_df = df.select_dtypes(include=np.number)
fig2, ax2  = plt.subplots(figsize=(8, 5))
sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)


# Model Performance
st.subheader('📈 Model Performance')
st.info(f'Overall Model R² Score: **{r2_global:.3f}**')

MIN_SAMPLES_FOR_R2 = 5

if len(brand_df) >= MIN_SAMPLES_FOR_R2:
    X_brand        = brand_df[['Seats', 'HorsePower', 'Total Speed',
                                'CC/Battery Capacity', 'Company Encoded']]
    y_brand        = brand_df['Log Price']
    X_brand_scaled = scaler.transform(X_brand)
    brand_r2       = r2_score(y_brand, model.predict(X_brand_scaled))
    st.info(f'{brand} Brand R² Score: **{brand_r2:.3f}**')
else:
    st.warning(
        f'Only {len(brand_df)} car(s) available for {brand} — '
        'brand-level R² is not reliable with such a small sample.'
    )

st.success(f'Prediction Confidence: **{confidence:.1f}%**')


# Footer
st.markdown(
    "<hr><p style='text-align:center;color:gray;'>Made by Dhruvik Parmar</p>",
    unsafe_allow_html=True
)
