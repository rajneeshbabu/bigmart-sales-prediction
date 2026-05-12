import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Big Mart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    min-height: 100vh;
}
.block-container { padding: 2rem 3rem; max-width: 1200px; }
.main-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #56CCF2, #2F80ED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title { text-align: center; color: #aaa; font-size: 1.05rem; margin-bottom: 2rem; }
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-value { font-size: 2rem; font-weight: 800; color: #56CCF2; }
.metric-label { font-size: 0.85rem; color: #aaa; margin-top: 0.2rem; }
.result-box {
    background: linear-gradient(135deg, rgba(86,204,242,0.15), rgba(47,128,237,0.15));
    border: 2px solid rgba(86,204,242,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-value { font-size: 3rem; font-weight: 900; color: #56CCF2; }
.result-label { color: #ccc; font-size: 1.1rem; margin-top: 0.5rem; }
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #fff;
    margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
div[data-testid='stSelectbox'] label { color: #ddd !important; }
div[data-testid='stNumberInput'] label { color: #ddd !important; }
div[data-testid='stSlider'] label { color: #ddd !important; }
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
    color: white; font-size: 1.1rem; font-weight: 700;
    padding: 0.7rem 2rem; border-radius: 50px; border: none;
    box-shadow: 0 4px 20px rgba(86,204,242,0.4); letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & metadata ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('models/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/metadata.json', 'r') as f:
        meta = json.load(f)
    return model, encoders, meta

model, encoders, meta = load_model()
FEATURES     = meta['features']
CAT_FEATURES = ['Item_Fat_Content', 'Item_Type', 'Item_Category', 'MRP_Band',
                'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']


# ── Feature engineering (must match training) ──────────────────────────────────
def engineer_single(row: dict) -> dict:
    fat_map = {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
    fc = fat_map.get(row['Item_Fat_Content'], row['Item_Fat_Content'])
    non_consumable = ['Health and Hygiene', 'Household', 'Others']
    if row['Item_Type'] in non_consumable:
        fc = 'Non-Consumable'

    prefix = row['Item_Identifier'][:2]
    item_cat = {'FD': 'Food', 'DR': 'Drinks', 'NC': 'Non-Consumable'}.get(prefix, 'Other')

    outlet_age = 2013 - row['Outlet_Establishment_Year']
    mrp = row['Item_MRP']
    mrp_log = np.log1p(mrp)

    if mrp <= 70:
        mrp_band = 'Budget'
    elif mrp <= 130:
        mrp_band = 'Mid'
    elif mrp <= 200:
        mrp_band = 'Premium'
    else:
        mrp_band = 'Luxury'

    ot_enc = {'Grocery Store': 0, 'Supermarket Type1': 1,
              'Supermarket Type2': 2, 'Supermarket Type3': 3}
    outlet_type_enc = ot_enc.get(row['Outlet_Type'], 1)

    vis = row['Item_Visibility'] if row['Item_Visibility'] > 0 else 0.06
    vis_ratio = vis / 0.06  # approximate mean outlet visibility

    outlet_size = row.get('Outlet_Size', 'Small') or 'Small'

    return {
        'Item_Weight':          row['Item_Weight'],
        'Item_Fat_Content':     fc,
        'Item_Visibility':      vis,
        'Item_Type':            row['Item_Type'],
        'Item_MRP':             mrp,
        'Outlet_Size':          outlet_size,
        'Outlet_Location_Type': row['Outlet_Location_Type'],
        'Outlet_Type':          row['Outlet_Type'],
        'Item_Category':        item_cat,
        'Outlet_Age':           outlet_age,
        'Vis_Ratio':            vis_ratio,
        'MRP_Band':             mrp_band,
        'Item_MRP_Log':         mrp_log,
        'Item_MRP_Sq':          mrp ** 2,
        'Outlet_Type_Encoded':  outlet_type_enc,
        'MRP_x_Outlet':         mrp * outlet_type_enc,
    }


def predict_sales(row: dict) -> float:
    feats = engineer_single(row)
    X_row = []
    for feat in FEATURES:
        val = feats[feat]
        if feat in CAT_FEATURES and feat in encoders:
            le = encoders[feat]
            val_str = str(val)
            if val_str in le.classes_:
                val = le.transform([val_str])[0]
            else:
                val = 0
        X_row.append(val)
    pred = model.predict([X_row])[0]
    return max(pred, 0)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🛒 Big Mart Sales Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict outlet sales for any item using a trained Random Forest model</div>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Model metrics row ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics_data = [
    ("CV RMSE", f"₹{meta['cv_rmse']:,.0f}", c1),
    ("CV R²",   f"{meta['cv_r2']:.4f}", c2),
    ("Model",   "Random Forest", c3),
    ("Training Samples", "8,523", c4),
]
for label, value, col in metrics_data:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>📦 Item Details</div>", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    item_id = st.selectbox("Item Identifier (prefix matters)",
        ['FD001', 'FD002', 'DR001', 'DR002', 'NC001', 'NC002'],
        help="FD = Food, DR = Drinks, NC = Non-Consumable")
    item_weight = st.number_input("Item Weight (kg)", min_value=1.0, max_value=25.0, value=12.0, step=0.1)
    item_fat    = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])

with col_b:
    item_type = st.selectbox("Item Type", [
        'Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods',
        'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene',
        'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks',
        'Others', 'Starchy Foods', 'Breakfast', 'Seafood'
    ])
    item_mrp  = st.slider("Item MRP (₹)", min_value=10.0, max_value=270.0, value=130.0, step=0.5)
    item_vis  = st.slider("Item Visibility (%)", min_value=0.0, max_value=0.35, value=0.05, step=0.001,
                          format="%.3f")

with col_c:
    outlet_type = st.selectbox("Outlet Type", [
        'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'
    ])
    outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
    outlet_loc  = st.selectbox("Outlet Location Tier", ['Tier 1', 'Tier 2', 'Tier 3'])

st.markdown("<div class='section-title'>🏪 Outlet Details</div>", unsafe_allow_html=True)
col_d, col_e, _ = st.columns([1, 1, 1])
with col_d:
    outlet_year = st.slider("Outlet Establishment Year", 1985, 2010, 1999)
with col_e:
    outlet_id = st.selectbox("Outlet Identifier", [
        'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT017',
        'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046'
    ])

st.markdown("<br>", unsafe_allow_html=True)

col_btn, _, _ = st.columns([1, 2, 1])
with col_btn:
    predict_btn = st.button("🔮  Predict Sales")

# ── Prediction output ──────────────────────────────────────────────────────────
if predict_btn:
    row = {
        'Item_Identifier':        item_id,
        'Item_Weight':            item_weight,
        'Item_Fat_Content':       item_fat,
        'Item_Visibility':        item_vis,
        'Item_Type':              item_type,
        'Item_MRP':               item_mrp,
        'Outlet_Identifier':      outlet_id,
        'Outlet_Establishment_Year': outlet_year,
        'Outlet_Size':            outlet_size,
        'Outlet_Location_Type':   outlet_loc,
        'Outlet_Type':            outlet_type,
    }
    pred = predict_sales(row)

    # Sales band classification
    if pred < 1000:
        band, color = "Low", "#F2994A"
    elif pred < 3000:
        band, color = "Medium", "#56CCF2"
    elif pred < 6000:
        band, color = "High", "#27AE60"
    else:
        band, color = "Very High", "#9B59B6"

    st.markdown(f"""
    <div class='result-box'>
        <div class='result-value' style='color:{color}'>₹ {pred:,.2f}</div>
        <div class='result-label'>Predicted Annual Item-Outlet Sales — <strong>{band}</strong> sales range</div>
    </div>""", unsafe_allow_html=True)

    # Show feature breakdown
    feats = engineer_single(row)
    mrp_band = feats['MRP_Band']
    item_cat = feats['Item_Category']
    outlet_age = feats['Outlet_Age']

    st.markdown("**Key factors used in prediction:**")
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("MRP Band",    mrp_band)
    fc2.metric("Item Category", item_cat)
    fc3.metric("Outlet Age",  f"{outlet_age} yrs")
    fc4.metric("MRP Log",     f"{feats['Item_MRP_Log']:.2f}")

# ── Model comparison ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>📊 Model Benchmark (5-fold CV)</div>", unsafe_allow_html=True)

comp = meta['model_comparison']
comp_df = pd.DataFrame([
    {"Model": k, "RMSE": f"₹{v['RMSE']:,.0f}", "R²": f"{v['R2']:.3f}",
     "Winner": "✅" if k == "Blend (final)" else ""}
    for k, v in comp.items()
])
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.8rem'>"
    "Built with Random Forest · scikit-learn · Streamlit · Big Mart Dataset (Kaggle)"
    "</p>",
    unsafe_allow_html=True
)
