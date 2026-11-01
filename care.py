"""
QueueCare AI - Hospital Queue Waiting-Time Prediction
Educational project using Random Forest Regression.

IMPORTANT:
- The built-in dataset is simulated, not real hospital data.
- Replace hospital_queue_data.csv with consent-based real data when available.
- Predictions are estimates, not medical advice or guaranteed wait times.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hospital_queue_data.csv"
MODEL_PATH = BASE_DIR / "queuecare_random_forest.joblib"

FEATURES = [
    "hour",
    "day_of_week",
    "queue_length",
    "doctors_available",
    "avg_service_minutes",
    "arrivals_last_hour",
    "emergency_cases",
]
TARGET = "waiting_time_minutes"


def create_sample_dataset(rows=1500, seed=42):
    """Create a simulated dataset for testing the project workflow."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(8, 21, rows)
    day_of_week = rng.integers(0, 7, rows)  # Monday=0, Sunday=6
    doctors = rng.integers(1, 7, rows)
    queue = rng.integers(0, 61, rows)
    avg_service = rng.uniform(5, 25, rows).round(1)
    arrivals = rng.integers(0, 31, rows)
    emergencies = rng.integers(0, 6, rows)

    # Synthetic relationship to produce plausible training labels.
    peak_period = ((hour >= 10) & (hour <= 13)) | ((hour >= 17) & (hour <= 19))
    weekend_factor = np.where(day_of_week >= 5, 1.08, 1.0)
    wait = (
        2
        + queue * avg_service / (doctors * 2.4)
        + arrivals * 0.65
        + emergencies * 2.8
        + peak_period * 5
    ) * weekend_factor
    wait += rng.normal(0, 4, rows)
    wait = np.clip(wait, 0, None).round(1)

    df = pd.DataFrame({
        "hour": hour,
        "day_of_week": day_of_week,
        "queue_length": queue,
        "doctors_available": doctors,
        "avg_service_minutes": avg_service,
        "arrivals_last_hour": arrivals,
        "emergency_cases": emergencies,
        TARGET: wait,
    })
    return df


def load_dataset():
    """Load a user dataset, or create a sample CSV if none exists."""
    if not DATA_PATH.exists():
        df = create_sample_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Created sample dataset: {DATA_PATH.name}")
        print("NOTE: This is simulated data for demonstration only.\n")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset: {DATA_PATH.name}")

    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
            + "\nPlease check the column names in hospital_queue_data.csv."
        )

    df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])

    if len(df) < 20:
        raise ValueError("Please provide at least 20 valid records to train and test the model.")
    return df


def crowd_label(queue_length, predicted_wait):
    """Simple project labels based on queue size and predicted wait."""
    if queue_length >= 30 or predicted_wait >= 60:
        return "HIGH"
    if queue_length >= 15 or predicted_wait >= 30:
        return "MEDIUM"
    return "LOW"


def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("========== MODEL EVALUATION ==========")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")
    print(f"MAE             : {mae:.2f} minutes")
    print(f"RMSE            : {rmse:.2f} minutes")
    print(f"R² score       : {r2:.3f}")
    print("======================================")
    print("Evaluation reflects the supplied data; simulated-data scores do not show real-world performance.\n")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model: {MODEL_PATH.name}\n")
    return model


def get_number(prompt, minimum, maximum, integer=True):
    """Read and validate numeric input."""
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
            if not minimum <= value <= maximum:
                print(f"Enter a value from {minimum} to {maximum}.")
                continue
            return int(value) if integer else value
        except ValueError:
            print("Please enter a valid number.")


def collect_patient_inputs():
    print("\nEnter current hospital conditions")
    print("Hour uses 24-hour format: 8 means 8 AM, 14 means 2 PM.")
    values = {
        "hour": get_number("Current hour (8-20): ", 8, 20),
        "day_of_week": get_number("Day of week (0=Mon, 6=Sun): ", 0, 6),
        "queue_length": get_number("Patients currently waiting: ", 0, 500),
        "doctors_available": get_number("Doctors currently available: ", 1, 100),
        "avg_service_minutes": get_number("Average service time per patient (minutes): ", 1, 240, integer=False),
        "arrivals_last_hour": get_number("Patients arriving in the last hour: ", 0, 500),
        "emergency_cases": get_number("Emergency cases currently waiting: ", 0, 100),
    }
    return pd.DataFrame([values], columns=FEATURES)


def recommend_time_slots(model, current_day):
    """Compare illustrative future slots under assumed conditions."""
    slots = []
    for hour in range(8, 21):
        # These are assumptions, not a real forecast of future queues.
        # In a real deployment, replace these with forecasted queue/arrival inputs.
        row = {
            "hour": hour,
            "day_of_week": current_day,
            "queue_length": 10 if hour in (8, 15, 16) else (30 if hour in (11, 12, 18) else 18),
            "doctors_available": 4 if hour in (8, 15, 16) else 3,
            "avg_service_minutes": 12,
            "arrivals_last_hour": 8 if hour in (8, 15, 16) else (22 if hour in (11, 12, 18) else 14),
            "emergency_cases": 1,
        }
        X_slot = pd.DataFrame([row], columns=FEATURES)
        wait = float(model.predict(X_slot)[0])
        slots.append((hour, wait, row["queue_length"]))
    slots.sort(key=lambda item: item[1])
    return slots[:3]


def main():
    print("\n====================================")
    print("        QUEUECARE AI")
    print(" Hospital Queue Prediction System")
    print("====================================\n")

    try:
        df = load_dataset()
        model = train_model(df)

        while True:
            patient = collect_patient_inputs()
            predicted_wait = max(0.0, float(model.predict(patient)[0]))
            queue = int(patient.iloc[0]["queue_length"])
            crowd = crowd_label(queue, predicted_wait)

            print("\n========== QUEUECARE RESULT ==========")
            print(f"Estimated waiting time: {predicted_wait:.1f} minutes")
            print(f"Current crowd category: {crowd}")
            print("This is an estimate and can change as the queue changes.")
            print("======================================")

            day = int(patient.iloc[0]["day_of_week"])
            print("\nPotential less-busy slots (based on demo assumptions):")
            for hour, wait, q in recommend_time_slots(model, day):
                am_pm = "AM" if hour < 12 else "PM"
                display_hour = hour if hour <= 12 else hour - 12
                print(f"  {display_hour}:00 {am_pm} | estimated {wait:.1f} min | assumed queue: {q}")

            again = input("\nPredict for another situation? (y/n): ").strip().lower()
            if again != "y":
                print("\nThank you for using QueueCare AI!")
                break

    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
        print(f"\nCould not run QueueCare AI: {exc}")
        print("Check your dataset file and column names, then try again.")


if __name__ == "__main__":
    main()
