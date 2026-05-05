# 🏈 NFL Game Prediction Models

This project builds machine learning models to predict NFL team Win-Loss rate across multiple seasons (2010–2025). It includes data scraping, model training, serialization, and prediction generation using **Random Forest**, **Linear Regression**, **Naive Bayes**, and **Neural Network** approaches.

---

## 📁 Project Structure

```text
COSC522_Group4_FinalProject/
├── Linear_regression.py
├── naive_bayes.py
├── neural_network.py
├── random_forest.py
├── scrape_data.py
├── data/
│   ├── 2010.csv
│   ├── 2011.csv
│   ├── 2012.csv
│   ├── 2013.csv
│   ├── 2014.csv
│   ├── 2015.csv
│   ├── 2016.csv
│   ├── 2017.csv
│   ├── 2018.csv
│   ├── 2019.csv
│   ├── 2020.csv
│   ├── 2021.csv
│   ├── 2022.csv
│   ├── 2023.csv
│   ├── 2024.csv
│   └── 2025.csv
├── models/
│   ├── Linear_Regression/
│   ├── Naive Bayes/
│   ├── Neural Network/
│   ├── Random Forest/
└── predictions/
│   ├── Linear_Regression/
│   ├── Naive Bayes/
│   ├── Neural Network/
│   ├── Random Forest/
├── LaTeX/
```

---

## 📊 Data

- Located in the `data/` directory
- Each subdirectory (2010–2025) contains CSV files for that NFL season
- Data is used as input for model training and evaluation

## 🤖 Models

- Stored in the `models/` directory
- Each year contains serialized versions of trained models using 'pickle' Python library
- Models are generated using:
  - `random_forest.py`
  - `Linear_regression.py`
  - `naive_bayes.py`
  - `neural_network.py`

## 📈 Predictions

- Stored in the `predictions/` directory
- Each subdirectory contains CSV files with predictions of win/loss percentages for each team for that season
- Predictions are generated after training each model

## 📁 LaTeX Directory
- Latex: Stores all LaTex Documents related to the report

---

## ⚙️ Scripts

### `scrape_data.py`
- Scrapes NFL data from a source website: https://www.pro-football-reference.com/years/<year>/ (where the <year> is the year we are collecting data for)
- Processes and saves it as CSV files
- Outputs data into the appropriate `data/<year>.csv/` directory

### `random_forest.py`
- Builds and trains a Random Forest model
- Generates predictions for a given season
- Serializes and saves the trained model to `Models/Random Forest/<year>/` using 'pickle' Python library
- Outputs predictions to `Predictions/Random Forest/<year>/`

### `Linear_regression.py`
- Builds and trains a Linear Regression model
- Generates predictions for a given season
- Serializes and saves the trained model to `Models/Linear Regression/<year>/` using 'pickle' Python library
- Outputs predictions to `Predictions/Linear Regression/<year>/` 

### `naive_bayes.py`
- Builds and trains a Naive Bayes model
- Generates predictions for a given season  
- Serializes and saves the trained model to `Models/Naive Bayes/<year>/` using 'pickle' Python library
- Outputs predictions to `Predictions/Naive Bayes/<year>/`

### `neural_network.py`
- Builds and trains a Neural Network model
- Generates predictions for a given season
- Serializes and saves the trained model to `Models/Neural Network/<year>/` using 'pickle' Python library
- Outputs predictions to `Predictions/Neural Network/<year>/`

---

## 🚀 Workflow

1. **Scrape Data**
   ```bash
   python scrape_data.py
   python random_forest.py
   python Linear_regression.py
   python naive_bayes.py
   python neural_network.py 
   ```

