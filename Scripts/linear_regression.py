
#              Greg Yonan                 #
#               COSC522                   #

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Below are column indicies
TEAM = 0 # Team Name
WINS = 1 # Wins
LOSSES = 2 # Losses
TIES = 3 # Ties
WIN_PERCENTAGE = 4 # Win Percentage
POINTS_FOR = 5 # Points For
POINTS_AGAINST = 6 # Points Against
POINTS_DIFFERENTIAL = 7 # Points Differential
MARGIN_OF_VICTORY = 8 # Margin of Victory
STRENGTH_OF_SCHEDULE = 9 # Strength of Schedule
SIMPLE_RATING_SYSTEM = 10 # Simple Rating System
OFFENSIVE_SRS = 11 # Offensive SRS
DEFENSIVE_SRS = 12 # Defensive SRS

# NFL constants
NFL_TEAMS = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders']
NFL_YEARS = [str(year) for year in range(2010, 2026)]

# Filepath constants
DATA_FILEPATH = "Data"
SCHEDULE_FILEPATH = "Schedule"
MODEL_FILEPATH = "Models\Linear Regression"
PREDICTIONS_FILEPATH = "Predictions\Linear Regression"

# this function reads data from each csv and retruns a dataframe 
def read_data(NFL_YEAR, team_name):
    team_data = []
    with open(os.path.join(DATA_FILEPATH, f"{NFL_YEAR}.csv"), "r") as file:
        data = pd.read_csv(file)
        for index, row in data.iterrows():
            if row.iloc[TEAM] == team_name:
                team_data.append([row.iloc[WINS], row.iloc[LOSSES], row.iloc[TIES], row.iloc[WIN_PERCENTAGE], row.iloc[POINTS_FOR], row.iloc[POINTS_AGAINST], row.iloc[POINTS_DIFFERENTIAL], row.iloc[MARGIN_OF_VICTORY], row.iloc[STRENGTH_OF_SCHEDULE], row.iloc[SIMPLE_RATING_SYSTEM], row.iloc[OFFENSIVE_SRS], row.iloc[DEFENSIVE_SRS]])
            elif row.iloc[TEAM] == team_name:
                team_data.append([row.iloc[WINS], row.iloc[LOSSES], row.iloc[TIES], row.iloc[WIN_PERCENTAGE], row.iloc[POINTS_FOR], row.iloc[POINTS_AGAINST], row.iloc[POINTS_DIFFERENTIAL], row.iloc[MARGIN_OF_VICTORY], row.iloc[STRENGTH_OF_SCHEDULE], row.iloc[SIMPLE_RATING_SYSTEM], row.iloc[OFFENSIVE_SRS], row.iloc[DEFENSIVE_SRS]])
    team_data = pd.DataFrame(team_data, columns=['W', 'L', 'T', 'W-L%', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS', 'OSRS', 'DSRS'])
    return team_data
  
# this function trains a linear regression model on all historical data and returns the model  
def train_model(NFL_YEAR):
    
    # read data for all teams and combine into a single dataframe
    all_data = []
    for team in NFL_TEAMS:
        team_data = read_data(NFL_YEAR, team)
        if team_data is not None and len(team_data) > 0:  # Ensure team_data is not None and not empty
            all_data.append(team_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # convert all columns to numeric values, if possible
    for column in combined_data.columns:
        combined_data[column] = pd.to_numeric(combined_data[column], errors='coerce')
    
    # Remove rows with NaN values
    combined_data = combined_data.dropna()
    
    # Separate features and target variable
    target = combined_data['W-L%']
    features = combined_data.drop(columns=['W', 'L', 'T', 'W-L%', 'PF', 'PA', 'MoV'])
    
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model
  
# this function saves the trained model as a serialized file in the Models directory  
def save_model(NFL_YEAR, model):
    if not os.path.exists(MODEL_FILEPATH):
        os.makedirs(MODEL_FILEPATH)
    with open(os.path.join(MODEL_FILEPATH, f"linear_regression_model_{NFL_YEAR}.pkl"), "wb") as file:
        pickle.dump(model, file)
        
# this function uses the trained model and the schedule for the season to make predictions for each game, then calculates the predicted W-L% for each team  at the end of the season and saves the predictions in a csv file in the Predictions directory
def make_predictions(NFL_YEAR):
    # Load the trained model
    model_path = os.path.join(MODEL_FILEPATH, f"linear_regression_model_{NFL_YEAR}.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    # Get the schedule for the season
    schedule_path = os.path.join(SCHEDULE_FILEPATH, f"{NFL_YEAR}.csv")
    schedule_df = pd.read_csv(schedule_path)
    if schedule_df.empty:
        print(f"No schedule found for {NFL_YEAR}")
        return
    
    # Initialize win/loss tracking for each team
    team_wins = {team: 0 for team in NFL_TEAMS}
    team_games = {team: 0 for team in NFL_TEAMS}
    
    # Read team stats for the season
    team_stats = {}
    try:
        season_data = pd.read_csv(os.path.join(DATA_FILEPATH, f"{NFL_YEAR}.csv"))
        for index, row in season_data.iterrows():
            team_name = row.iloc[TEAM]
            team_data = read_data(NFL_YEAR, team_name)
            if team_data is not None and len(team_data) > 0:
                team_stats[team_name] = team_data
    except FileNotFoundError:
        print(f"Data file for {NFL_YEAR} not found")
        return
    
    # Simulate each game in the schedule
    for index, row in schedule_df.iterrows():
        winner = row['Winner/tie']
        loser = row['Loser/tie']
            
        # Get team features
        winner_features = team_stats[winner].drop(columns=['W', 'L', 'T', 'W-L%', 'PF', 'PA', 'MoV'])
        loser_features = team_stats[loser].drop(columns=['W', 'L', 'T', 'W-L%', 'PF', 'PA', 'MoV'])
        
        # Predict point differential for each team
        winner_pd_pred = model.predict(winner_features)[0]
        loser_pd_pred = model.predict(loser_features)[0]
        
        # Predict the winner: team with higher predicted point differential wins
        if winner_pd_pred > loser_pd_pred:
            predicted_winner = winner
            predicted_loser = loser
        else:
            predicted_winner = loser
            predicted_loser = winner
        
        # Update win/loss records
        team_wins[predicted_winner] += 1
        team_games[predicted_winner] += 1
        team_games[predicted_loser] += 1
    
    # Calculate final predictions and actual results
    predictions = []
    for team in NFL_TEAMS:
        if team_games[team] > 0:
            predicted_wl = team_wins[team] / team_games[team]
        else:
            predicted_wl = 0.0
            
        if team in team_stats:
            actual_wl = team_stats[team]['W-L%'].iloc[0]
        else:
            actual_wl = 0.0
            
        predictions.append([team, actual_wl, predicted_wl])
    
    predictions_df = pd.DataFrame(predictions, columns=['Team', 'Actual W-L%', 'Predicted W-L%'])
    
    # see if the directory exists, if not create it
    if not os.path.exists(PREDICTIONS_FILEPATH):
        os.makedirs(PREDICTIONS_FILEPATH)
    predictions_df.to_csv(os.path.join(PREDICTIONS_FILEPATH, f"predictions_{NFL_YEAR}.csv"), index=False)

# === MAIN FUNCTION ===
if __name__ == "__main__":
    
    for year in NFL_YEARS:
        print(f"Training model for {year}...\n")
        model = train_model(year)
        save_model(year, model)
        print(f"Model for {year} saved successfully.\n")
        make_predictions(year)
        print(f"Predictions for {year} made successfully.\n")