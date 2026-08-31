import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os


# Page config

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

MODEL_PATH = "best_model_random_forest.pkl"
REFERENCE_YEAR = 2026  

FALLBACK_BRANDS = [
    "Maruti", "Hyundai", "Mahindra", "Tata", "Honda", "Toyota", "Ford",
    "Chevrolet", "Renault", "Volkswagen", "Skoda", "Nissan", "Datsun",
    "BMW", "Mercedes-Benz", "Audi", "Fiat", "Jaguar", "Land", "Jeep",
    "Mitsubishi", "Kia", "MG", "Volvo", "Isuzu", "Force", "Ambassador",
    "Daewoo", "Opel", "Peugeot", "Other",
]
FALLBACK_FUEL = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
FALLBACK_SELLER_TYPE = ["Individual", "Dealer", "Trustmark Dealer"]
FALLBACK_TRANSMISSION = ["Manual", "Automatic"]
FALLBACK_OWNER = [
    "First Owner", "Second Owner", "Third Owner",
    "Fourth & Above Owner", "Test Drive Car",
]


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def get_categories(_model):
    """Pull the exact categories the model was trained on from the
    fitted OneHotEncoder inside the pipeline. Falls back to hardcoded
    lists if anything about the pipeline structure doesn't match."""
    defaults = {
        "brand": FALLBACK_BRANDS,
        "fuel": FALLBACK_FUEL,
        "seller_type": FALLBACK_SELLER_TYPE,
        "transmission": FALLBACK_TRANSMISSION,
        "owner": FALLBACK_OWNER,
    }
    if _model is None:
        return defaults, False

    try:
        ohe = _model.named_steps["prep"].named_transformers_["cat"]
        cat_features = ["fuel", "seller_type", "transmission", "owner", "brand"]
        cats = dict(zip(cat_features, ohe.categories_))
        result = {
            "fuel": list(cats["fuel"]),
            "seller_type": list(cats["seller_type"]),
            "transmission": list(cats["transmission"]),
            "owner": list(cats["owner"]),
            "brand": sorted(cats["brand"]),
        }
        return result, True
    except Exception:
        return defaults, False


model = load_model(MODEL_PATH)
categories, loaded_from_model = get_categories(model)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🚗 Used Car Price Predictor")
st.caption(
    "Estimate a fair resale price from vehicle specs, trained on the "
    "CarDekho used-car dataset with a Random Forest model."
)

if model is None:
    st.warning(
        f"⚠️ Couldn't find **{MODEL_PATH}** in the app folder. "
        "The form below still works, but predictions are disabled until "
        "you place the trained pipeline file (produced by the notebook's "
        "`joblib.dump(rf_pipe, \"best_model_random_forest.pkl\")` step) "
        "next to `app.py`."
    )
elif not loaded_from_model:
    st.info(
        "Model loaded, but dropdown options fell back to defaults — "
        "double check the pipeline's step names match the notebook."
    )

st.divider()

# --------------------------------------------------------------------------
# Input form — laid out in columns
# --------------------------------------------------------------------------
st.subheader("Vehicle Details")

with st.form("car_details_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", options=categories["brand"])
        year = st.number_input(
            "Year of Manufacture",
            min_value=1990,
            max_value=2026,
            value=2017,
            step=1,
            help="Used to compute the car's age, same way as in training.",
        )
        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1_000_000,
            value=40000,
            step=1000,
        )
        fuel = st.selectbox("Fuel Type", options=categories["fuel"])

    with col2:
        seller_type = st.selectbox("Seller Type", options=categories["seller_type"])
        transmission = st.selectbox("Transmission", options=categories["transmission"])
        owner = st.selectbox("Ownership", options=categories["owner"])
        st.write("")  # spacing to visually balance the two columns
        st.write("")

    submitted = st.form_submit_button("🔮 Predict Selling Price", use_container_width=True)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if submitted:
    if model is None:
        st.error("No model loaded — add best_model_random_forest.pkl next to app.py to get a prediction.")
    else:
        car_age = REFERENCE_YEAR - year

        input_df = pd.DataFrame([{
            "km_driven": km_driven,
            "car_age": car_age,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "brand": brand,
        }])

        log_pred = model.predict(input_df)[0]
        price_pred = float(np.expm1(log_pred))

        st.divider()
        st.subheader("Estimated Selling Price")

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Price", f"₹ {price_pred:,.0f}")
        r2.metric("Car Age", f"{car_age} yrs")
        r3.metric("Km Driven", f"{km_driven:,}")

        with st.expander("Show model input"):
            st.dataframe(input_df, use_container_width=True)

        st.caption(
            "Estimate only — based on brand, age, mileage, fuel, seller type, "
            "transmission and ownership history. Condition, accident history, "
            "trim level and service records aren't captured by this model."
        )

st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown(
        """
- **Algorithm:** Random Forest Regressor (best performer vs. Linear Regression and Ridge in the notebook's comparison)
- **Target:** Selling price (log-transformed during training, converted back for display)
- **Features:** Brand, car age, kilometers driven, fuel type, seller type, transmission, ownership history
- **Dataset:** CarDekho used-car listings (4,340 records)
- **Known limitations:** No condition/accident history, trim, or service records — the model prices off usage & spec data only, so treat this as a starting estimate, not a final valuation.
        """
    )
