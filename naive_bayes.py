import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("RUNNING NAIVE BAYES FOR W-L% BUCKET CLASSIFICATION")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models_wlp_bucket")

REQUIRED_COLUMNS = [
    "Tm", "W", "L", "T", "PF", "PA", "OSRS", "DSRS"
]


def wl_bucket_label(wlp):
    """
    Convert W-L% into one of 4 bucket classes.
    """
    if wlp < 0.250:
        return 0   # Very Poor
    elif wlp < 0.500:
        return 1   # Below Average
    elif wlp < 0.750:
        return 2   # Good
    else:
        return 3   # Elite


def bucket_name(bucket_id):
    names = {
        0: "Very Poor",
        1: "Below Average",
        2: "Good",
        3: "Elite"
    }
    return names.get(bucket_id, "Unknown")


def bucket_midpoint(bucket_id):
    midpoints = {
        0: 0.125,
        1: 0.375,
        2: 0.625,
        3: 0.875,
    }
    return midpoints.get(bucket_id, 0.0)


def clean_team_summary(df: pd.DataFrame, season_name: str) -> pd.DataFrame:
    df = df.copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Tm"] = df["Tm"].astype(str).str.strip()

    # Remove any extra summary rows if present
    df = df[~df["Tm"].str.contains("Avg|Average|League", case=False, na=False)].copy()

    numeric_cols = ["W", "L", "T", "PF", "PA", "OSRS", "DSRS"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()

    if df.empty:
        raise ValueError("No usable rows after cleaning")

    df["Season"] = season_name
    df["total_games"] = df["W"] + df["L"] + df["T"]

    # Prevent divide-by-zero
    df = df[df["total_games"] > 0].copy()

    # Compute W-L%
    df["WL_percent"] = (df["W"] + 0.5 * df["T"]) / df["total_games"]

    # Extra engineered features
    df["point_diff_raw"] = df["PF"] - df["PA"]
    df["pf_pa_ratio"] = df["PF"] / (df["PA"] + 1)

    # Bucket target
    df["WL_bucket"] = df["WL_percent"].apply(wl_bucket_label)

    return df


def preprocess(df: pd.DataFrame, season_name: str):
    df = clean_team_summary(df, season_name)

    info_cols = [
        "Season", "Tm", "W", "L", "T", "PF", "PA",
        "WL_percent", "WL_bucket"
    ]
    info_df = df[info_cols].copy()

    # Features for Naive Bayes
    feature_cols = [
        "PF",
        "PA",
        "OSRS",
        "DSRS",
        "point_diff_raw",
        "pf_pa_ratio",
        "total_games",
    ]

    X = df[feature_cols].copy()
    y = df["WL_bucket"].copy()

    return X, y, info_df


def write_line(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    print("Looking in:", DATA_DIR)
    print("Saving outputs to:", MODEL_DIR)

    if not os.path.exists(DATA_DIR):
        print("ERROR: data folder not found")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    results_path = os.path.join(MODEL_DIR, "results.txt")
    predictions_txt_path = os.path.join(MODEL_DIR, "predictions.txt")
    predictions_csv_path = os.path.join(MODEL_DIR, "predictions.csv")
    model_path = os.path.join(MODEL_DIR, "nb_wlp_bucket.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler_wlp_bucket.pkl")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Naive Bayes Results for W-L% Bucket Classification\n")
        f.write("=" * 60 + "\n\n")

    with open(predictions_txt_path, "w", encoding="utf-8") as f:
        f.write("Predictions for W-L% Bucket Classification\n")
        f.write("=" * 60 + "\n\n")

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

    if not csv_files:
        write_line(results_path, "No CSV files found in data folder.")
        print("No CSV files found.")
        return

    all_X = []
    all_y = []
    all_info = []

    for file in csv_files:
        file_path = os.path.join(DATA_DIR, file)
        season_name = os.path.splitext(file)[0]

        print(f"\nProcessing {file}...")

        try:
            df = pd.read_csv(file_path)
            write_line(results_path, f"Season: {season_name}")
            write_line(results_path, f"Source file: {file}")
            write_line(results_path, f"Original rows: {len(df)}")
        except Exception as e:
            msg = f"Skipped: could not read file ({e})"
            print(msg)
            write_line(results_path, msg)
            write_line(results_path, "-" * 40)
            continue

        try:
            X, y, info_df = preprocess(df, season_name)
            write_line(results_path, f"Usable team rows: {len(X)}")
            print("Usable rows:", len(X))
        except Exception as e:
            msg = f"Skipped: preprocessing failed ({e})"
            print(msg)
            write_line(results_path, msg)
            write_line(results_path, "-" * 40)
            continue

        all_X.append(X)
        all_y.append(y)
        all_info.append(info_df)

        write_line(results_path, "-" * 40)

    if not all_X:
        write_line(results_path, "No valid data found after preprocessing.")
        print("No valid data found.")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)
    info_all = pd.concat(all_info, ignore_index=True)

    if len(X_all) < 10:
        print("Not enough total rows to train.")
        write_line(results_path, "Not enough total rows to train.")
        return

    if y_all.nunique() < 2:
        print("Target has only one class.")
        write_line(results_path, "Target has only one class.")
        return

    X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
        X_all,
        y_all,
        info_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print(f"\nAccuracy: {acc:.4f}")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    write_line(results_path, f"Total rows used: {len(X_all)}")
    write_line(results_path, f"Train rows: {len(X_train)}")
    write_line(results_path, f"Test rows: {len(X_test)}")
    write_line(results_path, f"Accuracy: {acc:.4f}")
    write_line(results_path, "")
    write_line(results_path, "Classification Report")
    write_line(results_path, "-" * 30)
    write_line(results_path, report)
    write_line(results_path, "")
    write_line(results_path, "Confusion Matrix")
    write_line(results_path, "-" * 30)
    write_line(results_path, str(cm))

    prediction_rows = []
    info_test = info_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    with open(predictions_txt_path, "a", encoding="utf-8") as f:
        for i, row in info_test.iterrows():
            actual_bucket = int(y_test.iloc[i])
            pred_bucket = int(preds[i])

            actual_name = bucket_name(actual_bucket)
            predicted_name = bucket_name(pred_bucket)
            actual_wl_percent = float(row["WL_percent"])
            predicted_wl_percent = bucket_midpoint(pred_bucket)

            line = (
                f"Team {i + 1}: {row['Tm']} ({row['Season']}) | "
                f"Actual Bucket={actual_name} | Predicted Bucket={predicted_name} | "
                f"Actual W-L%={actual_wl_percent:.3f} | "
                f"Predicted W-L%={predicted_wl_percent:.3f} | "
                f"Record={int(row['W'])}-{int(row['L'])}"
            )
            f.write(line + "\n")

            prediction_rows.append({
                "Season": row["Season"],
                "Team": row["Tm"],
                "Wins": int(row["W"]),
                "Losses": int(row["L"]),
                "Ties": int(row["T"]),
                "PF": int(row["PF"]),
                "PA": int(row["PA"]),
                "Actual_WL_Percent": round(actual_wl_percent, 3),
                "Predicted_WL_Percent": round(predicted_wl_percent, 3),
                "Actual_Bucket_ID": actual_bucket,
                "Actual_Bucket_Name": actual_name,
                "Predicted_Bucket_ID": pred_bucket,
                "Predicted_Bucket_Name": predicted_name,
                "Correct": int(actual_bucket == pred_bucket),
            })

    pd.DataFrame(prediction_rows).to_csv(predictions_csv_path, index=False)

    print("Saved model:", model_path)
    print("Saved scaler:", scaler_path)
    print("Saved results:", results_path)
    print("Saved predictions txt:", predictions_txt_path)
    print("Saved predictions csv:", predictions_csv_path)


if __name__ == "__main__":
    main()
