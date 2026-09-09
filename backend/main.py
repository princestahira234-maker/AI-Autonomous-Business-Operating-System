from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import json


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Autonomous Business Operating System API",
    version="1.0.0",
    description="Machine Learning APIs for business operations."
)


# =========================================================
# CORS - LOVABLE INTEGRATION
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# =========================================================
# HELPER
# =========================================================

def load_joblib(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# =========================================================
# =========================================================
# CUSTOMER CHURN
# =========================================================
# =========================================================

CHURN_DIR = MODELS_DIR / "customer_churn"

CHURN_MODEL_PATH = (
    CHURN_DIR / "customer_churn_logistic_regression.joblib"
)

CHURN_PREPROCESSOR_PATH = (
    CHURN_DIR / "customer_churn_preprocessor.joblib"
)

CHURN_THRESHOLD_PATH = (
    CHURN_DIR / "customer_churn_threshold.joblib"
)

CHURN_METADATA_PATH = (
    CHURN_DIR / "customer_churn_metadata.joblib"
)


try:
    churn_model = load_joblib(CHURN_MODEL_PATH)
    churn_preprocessor = load_joblib(CHURN_PREPROCESSOR_PATH)
    churn_threshold = load_joblib(CHURN_THRESHOLD_PATH)

    if isinstance(churn_threshold, dict):
        churn_threshold = churn_threshold.get(
            "threshold",
            churn_threshold.get("best_threshold", 0.60)
        )

    churn_threshold = float(churn_threshold)

    try:
        churn_metadata = load_joblib(CHURN_METADATA_PATH)
    except Exception:
        churn_metadata = {}

except Exception as e:
    churn_model = None
    churn_preprocessor = None
    churn_threshold = 0.60
    churn_metadata = {}
    print(f"Churn model loading warning: {e}")


class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/churn/health")
def churn_health():
    return {
        "module": "Customer Churn",
        "status": (
            "healthy"
            if churn_model is not None
            else "model_error"
        )
    }


@app.get("/churn/model-info")
def churn_model_info():
    return {
        "module": "Customer Churn",
        "model": "Logistic Regression",
        "threshold": churn_threshold,
        "model_loaded": churn_model is not None
    }


@app.post("/predict/churn")
def predict_churn(data: ChurnInput):

    if churn_model is None or churn_preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Churn model is not loaded."
        )

    input_data = pd.DataFrame([data.model_dump()])

    # Feature engineering
    input_data["TenureYears"] = (
        input_data["tenure"] / 12
    )

    input_data["IsNewCustomer"] = (
        input_data["tenure"] <= 6
    ).astype(int)

    input_data["IsLongTermCustomer"] = (
        input_data["tenure"] >= 24
    ).astype(int)

    input_data["MonthlyChargesLog"] = np.log1p(
        input_data["MonthlyCharges"]
    )

    input_data["TotalChargesLog"] = np.log1p(
        input_data["TotalCharges"]
    )

    input_data["AvgMonthlyValue"] = (
        input_data["TotalCharges"] /
        (input_data["tenure"] + 1)
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    input_data["ServiceCount"] = sum(
        (input_data[col] == "Yes").astype(int)
        for col in service_columns
    )

    input_data["HighMonthlyCharge"] = (
        input_data["MonthlyCharges"] > 70.35
    ).astype(int)

    input_data["MonthToMonth"] = (
        input_data["Contract"] == "Month-to-month"
    ).astype(int)

    input_data["ElectronicPayment"] = (
        input_data["PaymentMethod"] == "Electronic check"
    ).astype(int)

    try:
        transformed = churn_preprocessor.transform(
            input_data
        )

        probability = float(
            churn_model.predict_proba(transformed)[0][1]
        )

        prediction = int(
            probability >= churn_threshold
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Churn prediction error: {str(e)}"
        )

    if probability >= 0.75:
        risk_level = "HIGH"
    elif probability >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if prediction == 1:
        business_action = (
            "Customer is at high risk of churn. "
            "Consider retention action."
        )
    else:
        business_action = (
            "Customer is currently at lower risk of churn."
        )

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "threshold": churn_threshold,
        "risk_level": risk_level,
        "business_action": business_action
    }


# =========================================================
# =========================================================
# INVENTORY PREDICTION
# =========================================================
# =========================================================

INVENTORY_DIR = MODELS_DIR / "inventory"

INVENTORY_MODEL_PATH = (
    INVENTORY_DIR / "inventory_xgboost_model.pkl"
)

INVENTORY_ENCODER_PATH = (
    INVENTORY_DIR / "inventory_onehot_encoder.pkl"
)


try:
    inventory_model = load_joblib(
        INVENTORY_MODEL_PATH
    )

    inventory_encoder = load_joblib(
        INVENTORY_ENCODER_PATH
    )

except Exception as e:
    inventory_model = None
    inventory_encoder = None
    print(f"Inventory model loading warning: {e}")


class InventoryInput(BaseModel):
    store_id: str
    product_id: str
    category: str
    region: str
    weather_condition: str
    seasonality: str

    inventory_level: float
    price: float = 0.0
    discount: float = 0.0
    promotion: float = 0.0
    competitor_pricing: float = 0.0
    epidemic: float = 0.0

    demand_lag_1: float = 0.0
    demand_lag_7: float = 0.0
    demand_lag_14: float = 0.0
    demand_lag_28: float = 0.0

    demand_rolling_mean_7: float = 0.0
    demand_rolling_mean_14: float = 0.0
    demand_rolling_mean_28: float = 0.0


@app.get("/inventory/health")
def inventory_health():
    return {
        "module": "Inventory Prediction",
        "status": (
            "healthy"
            if inventory_model is not None
            else "model_error"
        )
    }


@app.get("/inventory/model-info")
def inventory_model_info():
    return {
        "module": "Inventory Prediction",
        "model": "XGBoost",
        "model_features": 55,
        "model_loaded": inventory_model is not None
    }


@app.post("/predict/inventory")
def predict_inventory(data: InventoryInput):

    if inventory_model is None or inventory_encoder is None:
        raise HTTPException(
            status_code=500,
            detail="Inventory model is not loaded."
        )

    categorical_columns = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    numerical_columns = [
        "Inventory Level",
        "Price",
        "Discount",
        "Promotion",
        "Competitor Pricing",
        "Epidemic",
        "demand_lag_1",
        "demand_lag_7",
        "demand_lag_14",
        "demand_lag_28",
        "demand_rolling_mean_7",
        "demand_rolling_mean_14",
        "demand_rolling_mean_28"
    ]

    input_data = pd.DataFrame([{
        "Store ID": data.store_id,
        "Product ID": data.product_id,
        "Category": data.category,
        "Region": data.region,
        "Weather Condition": data.weather_condition,
        "Seasonality": data.seasonality,

        "Inventory Level": data.inventory_level,
        "Price": data.price,
        "Discount": data.discount,
        "Promotion": data.promotion,
        "Competitor Pricing": data.competitor_pricing,
        "Epidemic": data.epidemic,

        "demand_lag_1": data.demand_lag_1,
        "demand_lag_7": data.demand_lag_7,
        "demand_lag_14": data.demand_lag_14,
        "demand_lag_28": data.demand_lag_28,

        "demand_rolling_mean_7":
            data.demand_rolling_mean_7,

        "demand_rolling_mean_14":
            data.demand_rolling_mean_14,

        "demand_rolling_mean_28":
            data.demand_rolling_mean_28
    }])

    try:
        categorical_encoded = inventory_encoder.transform(
            input_data[categorical_columns]
        )

        numerical_values = input_data[
            numerical_columns
        ].to_numpy(dtype=np.float32)

        final_features = np.hstack([
            numerical_values,
            categorical_encoded
        ])

        if final_features.shape[1] != 55:
            raise ValueError(
                "Inventory feature count mismatch: "
                f"expected 55, "
                f"received {final_features.shape[1]}"
            )

        prediction = float(
            inventory_model.predict(
                final_features
            )[0]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inventory prediction error: {str(e)}"
        )

    return {
        "predicted_inventory": round(
            prediction,
            2
        ),
        "store_id": data.store_id,
        "product_id": data.product_id,
        "category": data.category,
        "region": data.region,
        "weather_condition": data.weather_condition,
        "seasonality": data.seasonality
    }


# =========================================================
# =========================================================
# REVENUE PREDICTION
# =========================================================
# =========================================================
#
# Dataset is NOT required.
#
# Exact model structure:
#
# 18 numerical features
# + 75 encoded categorical features
# = 93 total features
#
# =========================================================

REVENUE_DIR = MODELS_DIR / "revenue"

REVENUE_MODEL_PATH = (
    REVENUE_DIR / "xgboost_revenue_model_v2.pkl"
)

REVENUE_ENCODER_PATH = (
    REVENUE_DIR / "xgboost_onehot_encoder.pkl"
)


try:
    revenue_model = load_joblib(
        REVENUE_MODEL_PATH
    )

    revenue_encoder = load_joblib(
        REVENUE_ENCODER_PATH
    )

except Exception as e:
    revenue_model = None
    revenue_encoder = None
    print(f"Revenue model loading warning: {e}")


REVENUE_CATEGORICAL_FEATURES = [
    "family",
    "city",
    "state",
    "type"
]


REVENUE_NUMERICAL_FEATURES = [
    "store_nbr",
    "onpromotion",
    "year",
    "month",
    "day",
    "week_of_year",
    "quarter",
    "day_of_week",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "cluster",
    "dcoilwtico",
    "is_holiday",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28"
]


REVENUE_EXPECTED_FEATURES = 93


class RevenueInput(BaseModel):
    date: str

    store_nbr: int

    family: str
    city: str
    state: str
    type: str

    onpromotion: float

    cluster: int

    dcoilwtico: float

    is_holiday: int

    sales_lag_1: float
    sales_lag_7: float
    sales_lag_14: float
    sales_lag_28: float


@app.get("/revenue/health")
def revenue_health():
    return {
        "module": "Revenue Prediction",
        "status": (
            "healthy"
            if revenue_model is not None
            else "model_error"
        )
    }


@app.get("/revenue/model-info")
def revenue_model_info():

    encoder_output_features = None

    if revenue_encoder is not None:
        try:
            sample = pd.DataFrame([{
                "family": "BEVERAGES",
                "city": "Quito",
                "state": "Pichincha",
                "type": "A"
            }])

            encoder_output_features = int(
                revenue_encoder.transform(
                    sample[
                        REVENUE_CATEGORICAL_FEATURES
                    ]
                ).shape[1]
            )

        except Exception:
            encoder_output_features = None

    return {
        "module": "Revenue Prediction",
        "model": "XGBoost Regressor",
        "target": "sales",
        "expected_total_features":
            REVENUE_EXPECTED_FEATURES,
        "numerical_features":
            REVENUE_NUMERICAL_FEATURES,
        "categorical_features":
            REVENUE_CATEGORICAL_FEATURES,
        "encoder_output_features":
            encoder_output_features,
        "model_loaded":
            revenue_model is not None,
        "dataset_required": False
    }


@app.post("/predict/revenue")
def predict_revenue(data: RevenueInput):

    if revenue_model is None or revenue_encoder is None:
        raise HTTPException(
            status_code=500,
            detail="Revenue model is not loaded."
        )

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    parsed_date = pd.to_datetime(
        data.date,
        errors="coerce"
    )

    if pd.isna(parsed_date):
        raise HTTPException(
            status_code=400,
            detail="Invalid date. Use YYYY-MM-DD format."
        )

    # -----------------------------------------------------
    # CALENDAR FEATURES
    # EXACTLY MATCH TRAINING
    # -----------------------------------------------------

    year = int(parsed_date.year)

    month = int(parsed_date.month)

    day = int(parsed_date.day)

    week_of_year = int(
        parsed_date.isocalendar().week
    )

    quarter = int(
        parsed_date.quarter
    )

    day_of_week = int(
        parsed_date.dayofweek
    )

    is_weekend = int(
        day_of_week >= 5
    )

    is_month_start = int(
        parsed_date.is_month_start
    )

    is_month_end = int(
        parsed_date.is_month_end
    )

    # -----------------------------------------------------
    # NUMERICAL FEATURES
    # -----------------------------------------------------

    numerical_data = pd.DataFrame([{
        "store_nbr": data.store_nbr,
        "onpromotion": data.onpromotion,

        "year": year,
        "month": month,
        "day": day,
        "week_of_year": week_of_year,
        "quarter": quarter,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_month_start": is_month_start,
        "is_month_end": is_month_end,

        "cluster": data.cluster,
        "dcoilwtico": data.dcoilwtico,
        "is_holiday": data.is_holiday,

        "sales_lag_1": data.sales_lag_1,
        "sales_lag_7": data.sales_lag_7,
        "sales_lag_14": data.sales_lag_14,
        "sales_lag_28": data.sales_lag_28
    }])

    numerical_values = numerical_data[
        REVENUE_NUMERICAL_FEATURES
    ].to_numpy(dtype=np.float32)

    # -----------------------------------------------------
    # CATEGORICAL FEATURES
    # -----------------------------------------------------

    categorical_data = pd.DataFrame([{
        "family": data.family,
        "city": data.city,
        "state": data.state,
        "type": data.type
    }])

    try:
        categorical_values = revenue_encoder.transform(
            categorical_data[
                REVENUE_CATEGORICAL_FEATURES
            ]
        )

        final_features = np.hstack([
            numerical_values,
            categorical_values
        ])

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Revenue feature transformation "
                f"error: {str(e)}"
            )
        )

    # -----------------------------------------------------
    # VERIFY 93 FEATURES
    # -----------------------------------------------------

    received_features = final_features.shape[1]

    if received_features != REVENUE_EXPECTED_FEATURES:
        raise HTTPException(
            status_code=500,
            detail=(
                "Revenue feature count mismatch: "
                f"expected "
                f"{REVENUE_EXPECTED_FEATURES}, "
                f"received {received_features}"
            )
        )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:
        prediction = float(
            revenue_model.predict(
                final_features
            )[0]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Revenue prediction error: "
                f"{str(e)}"
            )
        )

    prediction = max(
        0.0,
        prediction
    )

    return {
        "predicted_revenue": round(
            prediction,
            2
        ),
        "prediction_date":
            parsed_date.strftime("%Y-%m-%d"),
        "store_nbr": data.store_nbr,
        "family": data.family,
        "features_used": received_features
    }


# =========================================================
# =========================================================
# SUPPLIER LATE DELIVERY
# =========================================================
# =========================================================

SUPPLIER_DIR = MODELS_DIR / "supplier"

SUPPLIER_MODEL_PATH = (
    SUPPLIER_DIR /
    "supplier_xgboost_model.joblib"
)

SUPPLIER_ENCODER_PATH = (
    SUPPLIER_DIR /
    "supplier_onehot_encoder.joblib"
)

SUPPLIER_THRESHOLD_PATH = (
    SUPPLIER_DIR /
    "supplier_xgboost_threshold.joblib"
)


try:
    supplier_model = load_joblib(
        SUPPLIER_MODEL_PATH
    )

    supplier_encoder = load_joblib(
        SUPPLIER_ENCODER_PATH
    )

    supplier_threshold = load_joblib(
        SUPPLIER_THRESHOLD_PATH
    )

    if isinstance(supplier_threshold, dict):
        supplier_threshold = supplier_threshold.get(
            "threshold",
            0.32
        )

    supplier_threshold = float(
        supplier_threshold
    )

except Exception as e:
    supplier_model = None
    supplier_encoder = None
    supplier_threshold = 0.32
    print(f"Supplier model loading warning: {e}")


SUPPLIER_CATEGORICAL_FEATURES = [
    "Type",
    "Category Name",
    "Customer Country",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Country",
    "Order Region",
    "Product Name",
    "Shipping Mode"
]


SUPPLIER_NUMERICAL_FEATURES = [
    "Days for shipment (scheduled)",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Quantity",
    "order_year",
    "order_month",
    "order_day",
    "order_day_of_week",
    "order_week_of_year",
    "order_quarter",
    "order_is_weekend"
]


class SupplierInput(BaseModel):
    prediction_date: str

    type: str
    category_name: str
    customer_country: str
    customer_segment: str
    department_name: str
    market: str
    order_country: str
    order_region: str
    product_name: str
    shipping_mode: str

    days_for_shipment_scheduled: float
    order_item_discount: float
    order_item_discount_rate: float
    order_item_product_price: float
    order_item_quantity: float


@app.get("/supplier/health")
def supplier_health():
    return {
        "module": "Supplier Late Delivery Risk",
        "status": (
            "healthy"
            if supplier_model is not None
            else "model_error"
        )
    }


@app.get("/supplier/model-info")
def supplier_model_info():
    return {
        "module": "Supplier Late Delivery Risk",
        "model": "XGBoost",
        "threshold": supplier_threshold,
        "encoded_features": 301,
        "model_loaded": supplier_model is not None
    }


@app.post("/predict/supplier")
def predict_supplier(data: SupplierInput):

    if supplier_model is None or supplier_encoder is None:
        raise HTTPException(
            status_code=500,
            detail="Supplier model is not loaded."
        )

    parsed_date = pd.to_datetime(
        data.prediction_date,
        errors="coerce"
    )

    if pd.isna(parsed_date):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid prediction_date. "
                "Use YYYY-MM-DD."
            )
        )

    input_data = pd.DataFrame([{
        "Type": data.type,
        "Category Name": data.category_name,
        "Customer Country":
            data.customer_country,
        "Customer Segment":
            data.customer_segment,
        "Department Name":
            data.department_name,
        "Market": data.market,
        "Order Country":
            data.order_country,
        "Order Region":
            data.order_region,
        "Product Name":
            data.product_name,
        "Shipping Mode":
            data.shipping_mode,

        "Days for shipment (scheduled)":
            data.days_for_shipment_scheduled,

        "Order Item Discount":
            data.order_item_discount,

        "Order Item Discount Rate":
            data.order_item_discount_rate,

        "Order Item Product Price":
            data.order_item_product_price,

        "Order Item Quantity":
            data.order_item_quantity,

        "order_year":
            int(parsed_date.year),

        "order_month":
            int(parsed_date.month),

        "order_day":
            int(parsed_date.day),

        "order_day_of_week":
            int(parsed_date.dayofweek),

        "order_week_of_year":
            int(parsed_date.isocalendar().week),

        "order_quarter":
            int(parsed_date.quarter),

        "order_is_weekend":
            int(parsed_date.dayofweek >= 5)
    }])

    try:
        encoded = supplier_encoder.transform(
            input_data[
                SUPPLIER_CATEGORICAL_FEATURES
            ]
        )

        numerical = input_data[
            SUPPLIER_NUMERICAL_FEATURES
        ].to_numpy(dtype=np.float32)

        final_features = np.hstack([
            numerical,
            encoded
        ])

        probability = float(
            supplier_model.predict_proba(
                final_features
            )[0][1]
        )

        prediction = int(
            probability >= supplier_threshold
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Supplier prediction error: "
                f"{str(e)}"
            )
        )

    if probability >= 0.60:
        risk_level = "HIGH"
    elif probability >= supplier_threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if prediction == 1:
        business_action = (
            "Flag supplier/order for "
            "late delivery risk review"
        )
    else:
        business_action = (
            "Supplier/order appears to have "
            "lower late-delivery risk."
        )

    return {
        "late_delivery_risk": prediction,
        "late_delivery_probability":
            round(probability, 4),
        "threshold": supplier_threshold,
        "risk_level": risk_level,
        "business_action": business_action,
        "prediction_date":
            parsed_date.strftime("%Y-%m-%d")
    }


# =========================================================
# =========================================================
# TRANSACTION ANOMALY DETECTION
# =========================================================
# =========================================================

ANOMALY_DIR = MODELS_DIR / "anomaly"

ANOMALY_MODEL_PATH = (
    ANOMALY_DIR /
    "anomaly_xgboost_model.json"
)

ANOMALY_CONFIG_PATH = (
    ANOMALY_DIR /
    "anomaly_xgboost_config.json"
)

ANOMALY_FEATURE_CONFIG_PATH = (
    ANOMALY_DIR /
    "anomaly_feature_config.json"
)


try:
    anomaly_model = xgb.Booster()

    anomaly_model.load_model(
        str(ANOMALY_MODEL_PATH)
    )

    with open(
        ANOMALY_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        anomaly_config = json.load(f)

    with open(
        ANOMALY_FEATURE_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        anomaly_feature_config = json.load(f)

except Exception as e:
    anomaly_model = None
    anomaly_config = {}
    anomaly_feature_config = {}
    print(f"Anomaly model loading warning: {e}")


ANOMALY_THRESHOLD = 0.74

ANOMALY_AMOUNT_MEAN = 88.34961925093133

ANOMALY_AMOUNT_STD = 250.1201092401885


ANOMALY_FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "time_hour",
    "time_day",
    "time_hour_sin",
    "time_hour_cos",
    "amount_log",
    "amount_zscore"
]


class AnomalyInput(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


@app.get("/anomaly/health")
def anomaly_health():
    return {
        "module": "Transaction Anomaly Detection",
        "status": (
            "healthy"
            if anomaly_model is not None
            else "model_error"
        )
    }


@app.get("/anomaly/model-info")
def anomaly_model_info():
    return {
        "module": "Transaction Anomaly Detection",
        "model": "XGBoost",
        "threshold": ANOMALY_THRESHOLD,
        "features": len(ANOMALY_FEATURES),
        "model_loaded": anomaly_model is not None
    }


@app.post("/predict/anomaly")
def predict_anomaly(data: AnomalyInput):

    if anomaly_model is None:
        raise HTTPException(
            status_code=500,
            detail="Anomaly model is not loaded."
        )

    values = data.model_dump()

    time_value = float(
        values["Time"]
    )

    amount_value = float(
        values["Amount"]
    )

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    time_hour = (
        time_value / 3600.0
    ) % 24

    time_day = (
        time_value / 86400.0
    )

    time_hour_sin = np.sin(
        2 * np.pi * time_hour / 24
    )

    time_hour_cos = np.cos(
        2 * np.pi * time_hour / 24
    )

    amount_log = np.log1p(
        max(amount_value, 0)
    )

    amount_zscore = (
        amount_value -
        ANOMALY_AMOUNT_MEAN
    ) / ANOMALY_AMOUNT_STD

    final_data = {
        "Time": time_value,

        "V1": values["V1"],
        "V2": values["V2"],
        "V3": values["V3"],
        "V4": values["V4"],
        "V5": values["V5"],
        "V6": values["V6"],
        "V7": values["V7"],
        "V8": values["V8"],
        "V9": values["V9"],
        "V10": values["V10"],
        "V11": values["V11"],
        "V12": values["V12"],
        "V13": values["V13"],
        "V14": values["V14"],
        "V15": values["V15"],
        "V16": values["V16"],
        "V17": values["V17"],
        "V18": values["V18"],
        "V19": values["V19"],
        "V20": values["V20"],
        "V21": values["V21"],
        "V22": values["V22"],
        "V23": values["V23"],
        "V24": values["V24"],
        "V25": values["V25"],
        "V26": values["V26"],
        "V27": values["V27"],
        "V28": values["V28"],

        "Amount": amount_value,

        "time_hour": time_hour,
        "time_day": time_day,
        "time_hour_sin": time_hour_sin,
        "time_hour_cos": time_hour_cos,
        "amount_log": amount_log,
        "amount_zscore": amount_zscore
    }

    feature_df = pd.DataFrame(
        [final_data],
        columns=ANOMALY_FEATURES
    )

    try:
        matrix = feature_df.to_numpy(
            dtype=np.float32
        )

        dmatrix = xgb.DMatrix(
            matrix,
            feature_names=ANOMALY_FEATURES
        )

        prediction_values = anomaly_model.predict(
            dmatrix
        )

        probability = float(
            np.asarray(
                prediction_values
            ).reshape(-1)[0]
        )

        prediction = int(
            probability >= ANOMALY_THRESHOLD
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Anomaly prediction error: "
                f"{str(e)}"
            )
        )

    if probability >= 0.85:
        risk_level = "HIGH"
    elif probability >= ANOMALY_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if prediction == 1:
        business_action = (
            "Transaction flagged as potentially "
            "anomalous. Review transaction immediately."
        )
    else:
        business_action = (
            "Transaction appears normal."
        )

    return {
        "anomaly_prediction": prediction,
        "anomaly_probability":
            round(probability, 4),
        "threshold": ANOMALY_THRESHOLD,
        "risk_level": risk_level,
        "business_action": business_action
    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message":
            "AI Autonomous Business Operating System API",

        "status": "running",

        "version": "1.0.0",

        "modules": [
            "Customer Churn",
            "Inventory Prediction",
            "Revenue Prediction",
            "Supplier Late Delivery Risk",
            "Transaction Anomaly Detection"
        ],

        "endpoints": [
            "/health",

            "/churn/health",
            "/churn/model-info",
            "/predict/churn",

            "/inventory/health",
            "/inventory/model-info",
            "/predict/inventory",

            "/revenue/health",
            "/revenue/model-info",
            "/predict/revenue",

            "/supplier/health",
            "/supplier/model-info",
            "/predict/supplier",

            "/anomaly/health",
            "/anomaly/model-info",
            "/predict/anomaly",

            "/docs"
        ]
    }


# =========================================================
# GLOBAL HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message":
            "AI Autonomous Business Operating System API "
            "is running"
    }
