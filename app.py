import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px

# -------------------------
# 1. Page Config & CSS
# -------------------------
st.set_page_config(
    page_title="MotoPrice AI Predictor",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling for a sizzling hot, modern aesthetic
st.markdown("""
    <style>
    /* Styling for the prediction output card */
    .prediction-card {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(255, 75, 43, 0.4);
        color: white;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
        transition: transform 0.3s;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: -1px;
    }
    .prediction-subtitle {
        font-size: 1.2rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ff4b2b;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255, 75, 43, 0.2);
    }
    
    /* Button enhancements */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #FF416C, #FF4B2B);
        color: white;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.3);
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.5);
    }
    
    /* Input grouping backgrounds */
    .stSelectbox, .stSlider {
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# 2. Performance Caching
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("bike_price_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("Used_Bikes.csv")
    TOP_N_CITIES = 50
    top_cities = df["city"].value_counts().nlargest(TOP_N_CITIES).index
    
    brands = sorted(df["brand"].unique())
    cities = sorted(top_cities.tolist() + ["Other"])
    
    df["model_series"] = df["bike_name"].str.split().str[1].fillna("unknown")
    brand_to_series = {}
    for b in brands:
        vals = sorted(df.loc[df["brand"] == b, "model_series"].dropna().unique().tolist())
        brand_to_series[b] = vals if vals else sorted(df["model_series"].unique().tolist())
    
    return df, top_cities, brands, cities, brand_to_series

try:
    model = load_model()
    df, top_cities, brands, cities, brand_to_series = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

numeric_cols = ["kms_driven", "age", "power", "owner", "engine_from_name", "km_per_year"]
categorical_cols = ["brand", "city", "model_series"]

# -------------------------
# 3. UI Header
# -------------------------
col_head1, col_head2 = st.columns([1, 8])
with col_head1:
    st.image("https://cdn-icons-png.flaticon.com/512/1041/1041888.png", width=80) # Subtle icon
with col_head2:
    st.title("MotoPrice AI Engine 🏍️💨")
    st.markdown("*Leverage advanced market-driven machine learning to evaluate your used bike's true worth instantly.*")

st.write("---")

# -------------------------
# 4. Inputs Area
# -------------------------
st.markdown('<div class="section-header">🛠️ Bike Specifications</div>', unsafe_allow_html=True)

# Group inputs logically into two responsive columns
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🏷️ Identity & Origin")
    st.caption("Categorical defining traits of the motorcycle.")
    
    brand = st.selectbox("Manufacturer Brand", brands, help="Select the brand constructing the motorcycle.")
    series_options = brand_to_series.get(brand, sorted(df["model_series"].unique().tolist()))
    model_series = st.selectbox("Model Series", series_options, help="Specific model sub-series depending on the brand.")
    city = st.selectbox("Registration City", cities, help="City where the evaluation takes place.")
    
    # Format owners nicely
    owner = st.selectbox("Owner Sequence", [1, 2, 3, 4], 
                         format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd' if x==3 else 'th'} Owner", 
                         help="Number of previous owners.")

with col2:
    st.subheader("⚙️ Usage & Condition")
    st.caption("Technical parameters and wear history.")
    
    age = st.slider("Bike Age (Years)", min_value=0, max_value=30, value=5, step=1,
                    help="Years since the vehicle was manufactured.")
    kms_driven = st.slider("Kilometers Driven", min_value=0, max_value=200000, value=20000, step=500,
                           help="Total ground covered by the vehicle.")
    power = st.slider("Engine Power (CC)", min_value=50, max_value=2000, value=150, step=10,
                      help="Engine displacement volume.")
    
    # Live mini-calculation
    current_kmy = int(kms_driven / max(1, age))
    st.info(f"💡 **Live Insight:** This bike averages **{current_kmy:,} km** per year.")

st.write("---")

# -------------------------
# 5. Prediction Execution
# -------------------------
st.markdown('<div class="section-header">🔮 Price Prediction</div>', unsafe_allow_html=True)

action_col, result_col = st.columns([1, 1], gap="large")

with action_col:
    st.write("Analyze depreciation, brand prestige, and city demand to formulate a prediction using our trained XGBoost models.")
    predict_btn = st.button("🚀 Calculate Market Value", use_container_width=True)
    
    with st.expander("💡 How does this work?"):
        st.write("Our XGBoost model processes your 9 parameters securely to benchmark against local historical sale figures. Preprocessing steps match exactly the logic used during training.")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.form_submitted = False

if predict_btn:
    st.session_state.form_submitted = True
    with result_col:
        # User Experience Animations
        with st.spinner("Crunching metrics with XGBoost... ⚡"):
            time.sleep(1) # Minor delay for UI effect
            
            # --- Exact same original ML logic ---
            city_grouped = city if city in top_cities else "Other"
            engine_from_name = power
            km_per_year = kms_driven / (age + 1)
            
            input_data = pd.DataFrame([{
                "brand": brand,
                "city": city_grouped,
                "model_series": model_series,
                "kms_driven": kms_driven,
                "age": age,
                "power": power,
                "owner": owner,
                "engine_from_name": engine_from_name,
                "km_per_year": km_per_year
            }])
            
            prediction = model.predict(input_data)[0]
            st.session_state.prediction = prediction
            
            # Show celebratory balloons
            st.balloons()

if st.session_state.form_submitted and st.session_state.prediction is not None:
    pred = st.session_state.prediction
    with result_col:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-subtitle">Estimated Resale Value</div>
            <div class="prediction-value">₹ {int(pred):,}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # 6. Feature Importance DataViz
    # -------------------------
    st.write("---")
    st.markdown('<div class="section-header">📊 AI Insights: What Drives This Price?</div>', unsafe_allow_html=True)
    
    try:
        fitted_pipeline = model.regressor_
        pre = fitted_pipeline.named_steps["preprocessor"]
        xgb_model = fitted_pipeline.named_steps["model"]
        
        ohe = pre.named_transformers_["cat"]
        try:
            ohe_names = ohe.get_feature_names_out(categorical_cols)
        except Exception:
            cats = []
            for i, col in enumerate(categorical_cols):
                for v in ohe.categories_[i]:
                    cats.append(f"{col}_{v}")
            ohe_names = np.array(cats)
            
        num_names = np.array(numeric_cols)
        feature_names = np.concatenate([ohe_names, num_names])
        
        importances = xgb_model.feature_importances_
        if len(importances) == len(feature_names):
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(15)
            
            # Make names clean
            fi_df["Feature"] = fi_df["Feature"].str.replace('_', ' ').str.title()
            
            # Plotly Express visual with gradient theme
            fig = px.bar(
                fi_df, 
                x="Importance", 
                y="Feature", 
                orientation='h',
                color="Importance",
                color_continuous_scale="Inferno",  # Sizzling hot gradient!
                hover_data={"Feature": True, "Importance": ':.4f'},
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="Relative Impact on Price",
                yaxis_title="",
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("🤔 How to interpret these insights?"):
                st.write("• **Longer Bars:** Features that heavily swayed the final calculated price.")
                st.write("• **Shorter Bars:** Features acting as minor adjustments.")
        else:
            st.info("Feature importance dimension mismatch.")
            
    except Exception as e:
        st.info("Could not render feature importance engine context. Setup may differ slightly.")
        
elif not st.session_state.form_submitted:
    with result_col:
        st.info("👈 Fill out your bike specifics on the left, then hit **Calculate Market Value** to see the magic! ✨")