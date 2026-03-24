# 🏈 NFL Game Prediction Models

This project builds machine learning models to predict NFL game outcomes across multiple seasons (2015–2025). It includes data scraping, model training, serialization, and prediction generation using both **Random Forest** and **Logistic Regression** approaches.

---

## 📁 Project Structure
.
├── scrape_data.py
├── random_forest.py
├── logistic_regression.py
├── data/
│   ├── 2015/
│   ├── 2016/
│   ├── 2017/
│   ├── 2018/
│   ├── 2019/
│   ├── 2020/
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   └── 2025/
├── models/
│   ├── 2015/
│   ├── 2016/
│   ├── 2017/
│   ├── 2018/
│   ├── 2019/
│   ├── 2020/
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   └── 2025/
└── predictions/
    ├── 2015/
    ├── 2016/
    ├── 2017/
    ├── 2018/
    ├── 2019/
    ├── 2020/
    ├── 2021/
    ├── 2022/
    ├── 2023/
    ├── 2024/
    └── 2025/


---

## 📊 Data

- Located in the `data/` directory
- Each subdirectory (2015–2025) contains CSV files for that NFL season
- Data is used as input for model training and evaluation

---

## 🤖 Models

- Stored in the `models/` directory
- Each year contains serialized versions of trained models
- Models are generated using:
  - `random_forest.py`
  - `logistic_regression.py`

---

## 📈 Predictions

- Stored in the `predictions/` directory
- Each subdirectory contains an Excel file with predicted game winners for the full season
- Predictions are generated after training each model

---

## ⚙️ Scripts

### `scrape_data.py`
- Scrapes NFL data from a source website
- Processes and saves it as CSV files
- Outputs data into the appropriate `data/<year>/` directory

### `random_forest.py`
- Builds and trains a Random Forest model
- Generates predictions for a given season
- Serializes and saves the trained model to `models/<year>/`
- Outputs predictions to `predictions/<year>/`

### `logistic_regression.py`
- Builds and trains a Logistic Regression model
- Generates predictions for a given season
- Serializes and saves the trained model to `models/<year>/`
- Outputs predictions to `predictions/<year>/`

---

## 🚀 Workflow

1. **Scrape Data**
   ```bash
   python scrape_data.py
   python random_forest.py
   python logistic_regression.py

