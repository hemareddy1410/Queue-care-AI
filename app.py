
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="QueueCare AI",
    page_icon="🏥",
    layout="wide"
)


# ==========================================
# 2. LOAD RANDOM FOREST MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "queuecare_random_forest.joblib"


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except Exception as error:

    st.error(f"Unable to load model: {error}")

    st.info(
        "Run care.py first to generate "
        "queuecare_random_forest.joblib."
    )

    st.stop()


# ==========================================
# 3. DASHBOARD CSS
# ==========================================

st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */

    .stApp {
        background-color: #F4F7FB;
        color: #172B4D;
    }

    /* HEADINGS */

    h1, h2, h3, h4, h5, h6 {
        color: #172B4D !important;
    }

    /* MAIN TEXT */

    [data-testid="stMarkdownContainer"] p {
        color: #172B4D;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background-color: #E8EFF8;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #1261A0 !important;
    }

    [data-testid="stSidebar"] p {
        color: #172B4D !important;
    }

    /* NUMBER INPUT */

    [data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #172B4D !important;
        -webkit-text-fill-color: #172B4D !important;
        border: 1px solid #B8C8DC !important;
        border-radius: 8px !important;
        opacity: 1 !important;
    }

    [data-testid="stNumberInput"] button {
        background-color: #1261A0 !important;
        color: #FFFFFF !important;
    }

    /* SELECTBOX */

    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #B8C8DC !important;
        border-radius: 8px !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: #172B4D !important;
        -webkit-text-fill-color: #172B4D !important;
    }

    /* DROPDOWN OPTIONS */

    [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }

    [data-baseweb="popover"] [role="option"] {
        background-color: #FFFFFF !important;
        color: #172B4D !important;
    }

    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background-color: #E8F1FC !important;
        color: #1261A0 !important;
    }

    /* SLIDER */

    [data-testid="stSlider"] label {
        color: #172B4D !important;
    }

    /* BUTTONS */

    [data-testid="stFormSubmitButton"] button {
        background-color: #1261A0 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }

    [data-testid="stFormSubmitButton"] button p,
    [data-testid="stFormSubmitButton"] button span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #084B83 !important;
    }

    /* FORM */

    [data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D5DEE9 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }

    /* METRIC CARDS */

    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D5DEE9 !important;
        border-radius: 12px !important;
        padding: 18px !important;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetricValue"] {
        color: #172B4D !important;
    }

    /* DATAFRAME */

    [data-testid="stDataFrame"] {
        border: 1px solid #D5DEE9;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 4. HEADER
# ==========================================

st.title("🏥 QueueCare AI")

st.subheader(
    "Hospital Queue Waiting-Time Prediction"
)

st.write(
    "Predict hospital waiting times using Machine Learning "
    "and plan your hospital visit more effectively."
)

st.divider()


# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About QueueCare AI")

    st.write(
        "QueueCare AI uses a Random Forest Regression "
        "model to estimate patient waiting times."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Regression")

    st.write("**Task:** Waiting-Time Prediction")

    st.write("**Output:** Estimated waiting time in minutes")

    st.divider()

    st.caption(
        "Educational project. Predictions are estimates "
        "and are not guaranteed real-time waiting times."
    )


# ==========================================
# 6. PATIENT INPUT SECTION
# ==========================================

st.header("📋 Patient Queue Details")

st.write(
    "Enter the current hospital queue information "
    "to estimate the waiting time."
)


days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


with st.form("queue_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        hour = st.slider(
            "Hospital Visit Hour",
            min_value=8,
            max_value=20,
            value=10,
            help="Select the hospital visit hour."
        )

        day_name = st.selectbox(
            "Day of the Week",
            options=days,
            index=0
        )

        queue_length = st.number_input(
            "Current Queue Length",
            min_value=0,
            max_value=500,
            value=20,
            step=1
        )

        doctors_available = st.number_input(
            "Number of Doctors Available",
            min_value=1,
            max_value=100,
            value=2,
            step=1
        )

    with col2:

        avg_service_minutes = st.number_input(
            "Average Service Time (Minutes)",
            min_value=1,
            max_value=300,
            value=20,
            step=1
        )

        arrivals_last_hour = st.number_input(
            "Patients Arrived in Last Hour",
            min_value=0,
            max_value=1000,
            value=50,
            step=1
        )

        emergency_cases = st.number_input(
            "Number of Emergency Cases",
            min_value=0,
            max_value=200,
            value=5,
            step=1
        )

    st.write("")

    predict_button = st.form_submit_button(
        "🔍 Predict Waiting Time",
        use_container_width=True
    )


# ==========================================
# 7. PREDICTION SECTION
# ==========================================

if predict_button:

    try:

        # Convert day name to numerical value

        day = days.index(day_name)

        # Model feature names

        feature_columns = [
            "hour",
            "day_of_week",
            "queue_length",
            "doctors_available",
            "avg_service_minutes",
            "arrivals_last_hour",
            "emergency_cases"
        ]

        # CURRENT WAITING-TIME PREDICTION

        input_data = pd.DataFrame(
            [[
                hour,
                day,
                queue_length,
                doctors_available,
                avg_service_minutes,
                arrivals_last_hour,
                emergency_cases
            ]],
            columns=feature_columns
        )

        prediction = max(
            0,
            float(model.predict(input_data)[0])
        )

        # CROWD CLASSIFICATION

        if prediction <= 30:

            crowd_level = "LOW"

            crowd_message = (
                "The estimated waiting time is low."
            )

        elif prediction <= 60:

            crowd_level = "MEDIUM"

            crowd_message = (
                "The estimated waiting time is moderate."
            )

        else:

            crowd_level = "HIGH"

            crowd_message = (
                "The estimated waiting time is high."
            )

        # ==================================
        # PREDICTION RESULTS
        # ==================================

        st.divider()

        st.header("📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="Estimated Waiting Time",
                value=f"{prediction:.1f} minutes"
            )

        with result_col2:

            st.metric(
                label="Estimated Crowd Level",
                value=crowd_level
            )

        if crowd_level == "LOW":

            st.success(crowd_message)

        elif crowd_level == "MEDIUM":

            st.warning(crowd_message)

        else:

            st.error(crowd_message)

        # ==================================
        # 8. HOURLY PREDICTIONS
        # ==================================

        st.divider()

        st.header("🕒 Estimated Waiting Time by Hour")

        hourly_data = []

        for visit_hour in range(8, 21):

            hourly_input = pd.DataFrame(
                [[
                    visit_hour,
                    day,
                    queue_length,
                    doctors_available,
                    avg_service_minutes,
                    arrivals_last_hour,
                    emergency_cases
                ]],
                columns=feature_columns
            )

            hourly_prediction = max(
                0,
                float(model.predict(hourly_input)[0])
            )

            hourly_data.append({
                "Hour": f"{visit_hour:02d}:00",
                "Estimated Waiting Time": hourly_prediction
            })

        hourly_df = pd.DataFrame(hourly_data)

        # HOURLY GRAPH

        st.subheader("Hourly Waiting-Time Graph")

        st.line_chart(
            hourly_df.set_index("Hour"),
            y="Estimated Waiting Time"
        )

        # HOURLY DETAILS

        st.subheader("Hourly Prediction Details")

        st.dataframe(
            hourly_df,
            use_container_width=True,
            hide_index=True
        )

        # ==================================
        # 9. RECOMMENDED VISITING SLOTS
        # ==================================

        st.divider()

        st.header("⭐ Recommended Hospital Visiting Slots")

        st.write(
            "These are the three hours with the lowest "
            "predicted waiting times based on your inputs."
        )

        recommended_slots = (
            hourly_df
            .nsmallest(3, "Estimated Waiting Time")
            .sort_values(by="Hour")
            .reset_index(drop=True)
        )

        # DISPLAY RECOMMENDED SLOT CARDS
        # Use st.html() to render HTML correctly.

        slot_columns = st.columns(3)

        for index, row in recommended_slots.iterrows():

            with slot_columns[index]:

                slot_html = f"""
<div style="
    background-color: #1261A0;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    min-height: 170px;
    margin-bottom: 10px;
    box-sizing: border-box;
">

    <h3 style="
        color: #FFFFFF;
        font-size: 19px;
        margin-bottom: 15px;
    ">
        Recommended Slot {index + 1}
    </h3>

    <h1 style="
        color: #FFFFFF;
        font-size: 30px;
        margin: 10px 0;
    ">
        {row['Hour']}
    </h1>

    <p style="
        color: #FFFFFF;
        font-size: 15px;
        margin-top: 12px;
    ">
        {row['Estimated Waiting Time']:.1f}
        minutes waiting time
    </p>

</div>
"""

                st.html(slot_html)

        # ==================================
        # BEST RECOMMENDED SLOT
        # ==================================

        best_slot = (
            recommended_slots
            .sort_values(
                by="Estimated Waiting Time"
            )
            .iloc[0]
        )

        st.success(
            f"Best predicted visiting time: {best_slot['Hour']} "
            f"with an estimated waiting time of "
            f"{best_slot['Estimated Waiting Time']:.1f} minutes."
        )

        # ==================================
        # RECOMMENDED SLOTS TABLE
        # ==================================

        st.subheader("Recommended Visiting Time Details")

        display_slots = recommended_slots.rename(
            columns={
                "Hour": "Recommended Visiting Time",
                "Estimated Waiting Time":
                    "Predicted Waiting Time (Minutes)"
            }
        )

        st.dataframe(
            display_slots,
            use_container_width=True,
            hide_index=True
        )

        # ==================================
        # DISCLAIMER
        # ==================================

        st.info(
            "These visiting slots are predictions from your "
            "Random Forest model, not live hospital availability. "
            "Actual waiting times may vary."
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

footer_html = """
<div style="
    background-color: #E8EFF8;
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #D5DEE9;
">

    <h3 style="color: #1261A0;">
        QueueCare AI
    </h3>

    <p style="color: #172B4D;">
        Smart Hospital Queue Waiting-Time Prediction
    </p>

    <p style="color: #172B4D;">
        Built using Python, Streamlit and Random Forest
    </p>

</div>
"""

st.html(footer_html)
