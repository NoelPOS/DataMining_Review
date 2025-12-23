import pandas as pd
import numpy as np
import os
from collections import defaultdict
from operator import itemgetter
from sklearn.tree import DecisionTreeClassifier
# Using model_selection for modern scikit-learn compatibility
from sklearn.model_selection import cross_val_score, GridSearchCV 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --- 1. Setup and Data Loading (Pages 43-44) ---

data_folder = os.path.dirname(__file__)
data_filename = os.path.join(data_folder, "CH3_NBA_2013_2014_Games.csv")
# standings_filename = os.path.join(data_folder, "leagues_NBA_2013_standings_expanded-standings.csv")

dataset = pd.read_csv(data_filename, parse_dates=["Date"], skiprows=[0,])
dataset.columns = ["Date", "Start_ET", "Visitor_Team", "VisitorPts", "Home_Team", "HomePts", "Score_Type", "Empty", "Attend", "LOG", "Arena", "Notes"]


print(dataset)


dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
y_true = dataset["HomeWin"].values

print(dataset["HomeWin"])

y_true = dataset["HomeWin"].values

# print the home win rate
home_win_rate = np.mean(dataset["HomeWin"]) * 100
print(f"Home Win Rate: {home_win_rate:.1f}%")

from collections import defaultdict
won_last = defaultdict(int)

dataset["HomeLastWin"] = 0
dataset["VisitorLastWin"] = 0

# Sort the dataset by date to ensure proper chronological processing
for index, row in dataset.sort_values(by="Date").iterrows():
    home_team = row["Home_Team"]
    visitor_team = row["Visitor_Team"]
    dataset.loc[index, "HomeLastWin"] = int(won_last[home_team])
    dataset.loc[index, "VisitorLastWin"] = int(won_last[visitor_team])
    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]
    

print(dataset[["Date", "Home_Team", "Visitor_Team", "HomeLastWin", "VisitorLastWin"]])


clf = DecisionTreeClassifier(random_state=14)
X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print("\nAccuracy (LastWin features only, Decision Tree): {0:.1f}%".format(np.mean(scores) * 100))







# --- 2. Feature Engineering (Pages 45-53) ---

# 2.1. Create the target variable: HomeWin (Page 45)
# True if HomePts > VisitorPts (Home team won)
# dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
# y_true = dataset["HomeWin"].values
# print("Target variable 'HomeWin' created.")


# # 2.2. Create rolling window features: HomeLastWin and VisitorLastWin (Page 45)
# # Tracks whether the team won its immediate previous game.
# won_last = defaultdict(int)
# dataset["HomeLastWin"] = 0
# dataset["VisitorLastWin"] = 0

# for index, row in dataset.iterrows():
#     home_team = row["Home Team"]
#     visitor_team = row["Visitor Team"]

#     # Store the result of the last game for both teams
#     dataset.loc[index, "HomeLastWin"] = won_last[home_team]
#     dataset.loc[index, "VisitorLastWin"] = won_last[visitor_team]

#     # Update the dictionary for the next game: 1 (True) for win, 0 (False) for loss
#     won_last[home_team] = row["HomeWin"]
#     won_last[visitor_team] = not row["HomeWin"]

# print("Features 'HomeLastWin' and 'VisitorLastWin' created.")

# # 2.3. Load Standings for HomeTeamRanksHigher feature (Page 50)
# standings = pd.read_csv(standings_filename, skiprows=[0, 1])

# # 2.4. Create the HomeTeamRanksHigher feature (Pages 50-51)
# dataset["HomeTeamRanksHigher"] = 0

# for index, row in dataset.iterrows():
#     home_team = row["Home Team"]
#     visitor_team = row["Visitor Team"]

#     # Handle team name change from 2013 to 2014 season (Page 51)
#     if home_team == "New Orleans Pelicans":
#         home_team = "New Orleans Hornets"
#     elif visitor_team == "New Orleans Pelicans":
#         visitor_team = "New Orleans Hornets"

#     # Fetch ranks. 'Rk' (Rank) in the standings uses lower number for higher rank.
#     # The book's implementation uses '>' which is numerically correct for ranks 
#     # (higher rank number means lower actual rank) but logically questionable 
#     # in this context. We follow the book's code literally:
#     home_rank = standings[standings["Team"] == home_team]["Rk"].values[0]
#     visitor_rank = standings[standings["Team"] == visitor_team]["Rk"].values[0]

#     # HomeTeamRanksHigher = 1 if home team's rank is numerically > visitor's rank 
#     # (i.e., home team has a worse standing)
#     dataset.loc[index, "HomeTeamRanksHigher"] = int(home_rank > visitor_rank)

# print("Feature 'HomeTeamRanksHigher' created.")


# # 2.5. Create the HomeTeamWonLast feature (Past Encounters) (Page 52)
# # Tracks which team won the last time they met.
# last_match_winner = defaultdict(int)
# dataset["HomeTeamWonLast"] = 0

# for index, row in dataset.iterrows():
#     home_team = row["Home Team"]
#     visitor_team = row["Visitor Team"]
    
#     # Sort team names alphabetically for a consistent key for the matchup (Page 52)
#     teams = tuple(sorted([home_team, visitor_team]))
    
#     # Assign if the home team won the last encounter. 
#     # last_match_winner[teams] stores the *name* of the winner of the last matchup.
#     dataset.loc[index, "HomeTeamWonLast"] = 1 if last_match_winner[teams] == home_team else 0
    
#     # Update the dictionary with the winner of this game
#     winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
#     last_match_winner[teams] = winner

# print("Feature 'HomeTeamWonLast' created.")


# # --- 3. Model Training and Evaluation (Pages 49, 53-57) ---

# # 3.1. Evaluate Decision Tree with HomeLastWin/VisitorLastWin (Page 49)
# X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values
# clf_dt = DecisionTreeClassifier(random_state=14)
# scores = cross_val_score(clf_dt, X_previouswins, y_true, scoring='accuracy')
# print("\nAccuracy (LastWin features only, Decision Tree): {0:.1f}%".format(np.mean(scores) * 100))


# # 3.2. Evaluate Decision Tree with 3 features (Page 52)
# X_homehigher = dataset[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
# scores = cross_val_score(clf_dt, X_homehigher, y_true, scoring='accuracy')
# print("Accuracy (3 features, Decision Tree): {0:.1f}%".format(np.mean(scores) * 100))


# # 3.3. Evaluate Decision Tree with RanksHigher and HomeTeamWonLast (Page 53)
# X_lastwinner = dataset[["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
# scores = cross_val_score(clf_dt, X_lastwinner, y_true, scoring='accuracy')
# print("Accuracy (RanksHigher + HomeTeamWonLast, Decision Tree): {0:.1f}%".format(np.mean(scores) * 100))


# # 3.4. Decision Tree with One-Hot Encoded Teams (Pages 53-54)

# # a) Label Encode team names (Page 53)
# encoding = LabelEncoder()
# encoding.fit(dataset["Home Team"].values) 
# home_teams = encoding.transform(dataset["Home Team"].values)
# visitor_teams = encoding.transform(dataset["Visitor Team"].values)
# X_teams = np.vstack([home_teams, visitor_teams]).T

# # b) One-Hot Encode team integers (Page 54)
# onehot = OneHotEncoder(sparse_output=False)
# X_teams_expanded = onehot.fit_transform(X_teams)

# # c) Evaluate (Page 54)
# scores = cross_val_score(clf_dt, X_teams_expanded, y_true, scoring='accuracy')
# print("Accuracy (One-Hot Encoded Teams only, Decision Tree): {0:.1f}%".format(np.mean(scores) * 100))


# # 3.5. Random Forests with ALL Features (Pages 56-57)

# # Combine all meaningful numerical features (4 features) with the expanded team features
# X_home_4_features = dataset[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher", "HomeTeamWonLast"]].values
# X_all = np.hstack([X_home_4_features, X_teams_expanded])

# # a) Basic Random Forest Evaluation (Page 56, modified for all features, Page 57)
# clf_rf = RandomForestClassifier(random_state=14)
# scores = cross_val_score(clf_rf, X_all, y_true, scoring='accuracy')
# print("Accuracy (All Features, Random Forest): {0:.1f}%".format(np.mean(scores) * 100))
# # 

# # b) Optimization with GridSearchCV (Page 57)
# parameter_space = {
#     "max_features": [2, 10, 'auto'],
#     "n_estimators": [100,],
#     "criterion": ["gini", "entropy"],
#     "min_samples_leaf": [2, 4, 6],
# }

# grid = GridSearchCV(clf_rf, parameter_space, scoring='accuracy', cv=5)
# grid.fit(X_all, y_true)

# print("Accuracy (GridSearchCV Best Score): {0:.1f}%".format(grid.best_score_ * 100))
# print("Best Estimator Parameters: {}".format(grid.best_estimator_))