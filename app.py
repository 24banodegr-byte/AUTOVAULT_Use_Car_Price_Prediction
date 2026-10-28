import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AUTOVAULT | Used Car Price AI", page_icon="🚘", layout="wide")

st.markdown("""
<style>
.stApp {background:#101312;color:#f4f7f4}
[data-testid="stSidebar"] {background:#171d19}
h1,h2,h3 {color:#d5ff3f}
div[data-testid="stMetric"] {background:#1b241e;border:1px solid #394b30;padding:16px;border-radius:14px}
.stButton>button {background:#d5ff3f;color:#101312;border:0;border-radius:9px;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚘 AUTOVAULT</h1><p>USED CAR PRICE ANALYSIS & PREDICTION</p>", unsafe_allow_html=True)
st.caption("Upload your CarDekho CSV to explore the market and estimate a car's resale price.")

uploaded = st.sidebar.file_uploader("Upload CarDekho CSV", type=["csv"])
if uploaded is None:
    st.info("Upload your dataset from the left sidebar to open the dashboard.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    data = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    data.columns = data.columns.str.strip().str.lower()
    return data

df = load_data(uploaded.getvalue())
required = ["name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", "owner"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.write("Available columns:", df.columns.tolist())
    st.stop()

for col in ["year", "selling_price", "km_driven"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["name", "year", "selling_price", "km_driven"])
df = df[(df["selling_price"] > 0) & (df["km_driven"] >= 0)].copy()
if len(df) < 10:
    st.error("Please upload a dataset with at least 10 valid car records.")
    st.stop()

features = ["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
X = df[features].copy()
y = df["selling_price"].copy()
numeric = ["year", "km_driven"]
categorical = ["name", "fuel", "seller_type", "transmission", "owner"]
preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_options = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}
scores, fitted = [], {}
with st.spinner("Training four simple regression models..."):
    for name, regressor in model_options.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        scores.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        })
        fitted[name] = pipe
scores_df = pd.DataFrame(scores).sort_values("RMSE").reset_index(drop=True)
best_name = scores_df.loc[0, "Model"]
model = fitted[best_name]

c1, c2, c3 = st.columns(3)
c1.metric("Cars in dataset", f"{len(df):,}")
c2.metric("Average listed price", f"₹{df['selling_price'].mean():,.0f}")
c3.metric("Best model", best_name)
st.divider()

tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "📊 Market Dashboard", "🧪 Model Comparison"])

with tab1:
    st.subheader("Estimate a car's resale price")
    left, right = st.columns(2)
    with left:
        car_name = st.selectbox("Car model", sorted(df["name"].astype(str).unique()))
        year = st.slider("Manufacturing year", int(df["year"].min()), max(2026, int(df["year"].max())), int(df["year"].median()))
        km = st.number_input("Kilometres driven", min_value=0, value=30000, step=1000)
        fuel = st.selectbox("Fuel", sorted(df["fuel"].dropna().astype(str).unique()))
    with right:
        seller = st.selectbox("Seller type", sorted(df["seller_type"].dropna().astype(str).unique()))
        transmission = st.selectbox("Transmission", sorted(df["transmission"].dropna().astype(str).unique()))
        owner = st.selectbox("Previous owner", sorted(df["owner"].dropna().astype(str).unique()))
        st.write("")
        predict = st.button("PREDICT PRICE", use_container_width=True)
    if predict:
        item = pd.DataFrame([{
            "name": car_name, "year": year, "km_driven": km, "fuel": fuel,
            "seller_type": seller, "transmission": transmission, "owner": owner
        }])[features]
        estimate = max(0, float(model.predict(item)[0]))
        st.success(f"Estimated selling price: ₹{estimate:,.0f}  (₹{estimate/100000:.2f} lakh)")
        st.caption("This is a machine-learning estimate, not a guaranteed market price.")

with tab2:
    st.subheader("Explore the used-car market")
    a, b, c = st.columns(3)
    search = a.text_input("Search car name")
    fuel_filter = b.selectbox("Fuel filter", ["All"] + sorted(df["fuel"].dropna().astype(str).unique()))
    trans_filter = c.selectbox("Transmission filter", ["All"] + sorted(df["transmission"].dropna().astype(str).unique()))
    min_price, max_price = int(df["selling_price"].min()), int(df["selling_price"].max())
    price_range = st.slider("Selling price range (₹)", min_price, max_price, (min_price, max_price))
    view = df[df["selling_price"].between(price_range[0], price_range[1])]
    if search:
        view = view[view["name"].astype(str).str.contains(search, case=False, na=False)]
    if fuel_filter != "All":
        view = view[view["fuel"].astype(str) == fuel_filter]
    if trans_filter != "All":
        view = view[view["transmission"].astype(str) == trans_filter]
    st.write(f"Matching records: {len(view):,}")
    st.dataframe(view.head(500), use_container_width=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"), "filtered_cars.csv", "text/csv")
    fig, ax = plt.subplots()
    ax.hist(view["selling_price"], bins=25)
    ax.set_title("Selling price distribution")
    ax.set_xlabel("Selling price (₹)")
    ax.set_ylabel("Number of cars")
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Compare model performance")
    st.caption("Lower MAE/RMSE is better; higher R² is generally better.")
    st.dataframe(scores_df.round(3), use_container_width=True)
    fig, ax = plt.subplots()
    ax.bar(scores_df["Model"], scores_df["RMSE"])
    ax.set_ylabel("RMSE (₹)")
    ax.set_title("RMSE comparison (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
    plt.close(fig)
    st.success(f"Selected model: {best_name} — lowest test RMSE.")
