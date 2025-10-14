# %% [markdown]
"""
# MLB Moneyline Regression Analysis

This is a starting point for a more complex data analysis - but let's lay some groundwork.

**Mission:** Predict Moneylines from current standings and starting pitcher ERA

**Complexity:** Small

**Backtested:** No

**Chances of beating the books:** Low
"""

# %%
import mlbstatsapi, requests

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings, pprint, random
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette('husl')

# %% [markdown]
"""
## Data Collection

Collect current MLB team standings, historical game results, and starting pitcher ERA data to build our moneyline prediction features.
"""

# %%
def get_all_gamePks(season, start_month=1, end_month=12):
    season = requests.get('https://statsapi.mlb.com/api/v1/schedule', params={
        'startDate': f'{season}-{start_month}-01',
        'endDate': f'{season}-{end_month}-31',
        'sportId': 1,
        'gameType': 'R',  # Regular season games
        'language': 'en',
        'hydrate': 'game(boxscore)',
    }).json()
    return [game['gamePk'] for date in season['dates'] for game in date['games']]

twothree_gamepks = get_all_gamePks(2023)  # Example usage to fetch all game Pks for the 2023 season
twothree_gamepks[100]

# %%
def get_line_score(game_pk):
    """
    Fetches the line score for a given game using its gamePk.
    """
    url = f'https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching line score for game {game_pk}: {response.status_code}")
        return None

def get_box_score(game_pk):
    """
    Fetches the box score for a given game using its gamePk.
    """
    url = f'https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching box score for game {game_pk}: {response.status_code}")
        return None
    
box = get_box_score(twothree_gamepks[100]) # Example usage to fetch box score for the first gamePk
line = get_line_score(twothree_gamepks[100]) # Example usage to fetch line score for the first gamePk
pprint.pprint(box['teams']['home'])

# %%
def get_team_postgame_record(box, home=True):
    """
    Extracts the pregame record of the team from the box score.
    """
    if 'teams' in box:
        team = box['teams']['home'] if home else box['teams']['away']
        return {
            'wins': team['team']['record']['wins'],
            'losses': team['team']['record']['losses'],
        }
    return None

def get_starter_pregame_stats(box, home=True, stats=['earnedRuns', 'hits', 'baseOnBalls', 'homeRuns', 'outs']):
    """
    Extracts the starting pitcher's ERA before the game from the box score.

    possible stats to take: ['gamesPlayed', 'gamesStarted', 'flyOuts', 'groundOuts', 'airOuts', 'runs', 'doubles', 'triples',
     'homeRuns', 'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch', 'atBats', 'obp', 'caughtStealing', 'stolenBases',
      'stolenBasePercentage', 'numberOfPitches', 'era', 'inningsPitched', 'wins', 'losses', 'saves', 'saveOpportunities', 'holds',
       'blownSaves', 'earnedRuns', 'whip', 'battersFaced', 'outs', 'gamesPitched', 'completeGames', 'shutouts', 'pitchesThrown',
        'balls', 'strikes', 'strikePercentage', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffs', 'groundOutsToAirouts', 'rbi',
         'winPercentage', 'pitchesPerInning', 'gamesFinished', 'strikeoutWalkRatio', 'strikeoutsPer9Inn', 'walksPer9Inn', 'hitsPer9Inn',
          'runsScoredPer9', 'homeRunsPer9', 'inheritedRunners', 'inheritedRunnersScored', 'catchersInterference', 'sacBunts', 'sacFlies',
           'passedBall', 'popOuts', 'lineOuts']
    """
    if 'teams' in box and 'pitchers' in box['teams']['home']:
        pitchers = box['teams']['home']['pitchers'] if home else box['teams']['away']['pitchers']
        starter = box['teams']['home']['players'][f'ID{pitchers[0]}'] if home else box['teams']['away']['players'][f'ID{pitchers[0]}']
        postgame = starter['seasonStats']['pitching']
        game = starter['stats']['pitching']

        return {stat: postgame.get(stat, 0) - game.get(stat, 0) for stat in stats}
    return None

def get_batters_pregrame_stats(box, home=True, stats=['hits', 'totalBases', 'homeRuns', 'plateAppearances', 'baseOnBalls']):
    """
    Extracts the pregame stats of a batter from the box score.

    dict_keys(['summary', 'gamesPlayed', 'flyOuts', 'groundOuts', 'airOuts', 'runs', 'doubles', 'triples', 'homeRuns',
    'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch', 'atBats', 'caughtStealing', 'stolenBases',
      'stolenBasePercentage', 'groundIntoDoublePlay', 'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi',
        'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference', 'pickoffs', 'atBatsPerHomeRun', 'popOuts', 'lineOuts']
    """
    if 'teams' in box and 'battingOrder' in box['teams']['home']:
        batters = box['teams']['home']['battingOrder'] if home else box['teams']['away']['battingOrder']
        batter_stats = list()
        batter_sums = {stat: 0 for stat in stats}
        for batter_id in batters:
            batter = box['teams']['home']['players'][f'ID{batter_id}'] if home else box['teams']['away']['players'][f'ID{batter_id}']
            postgame = batter['seasonStats']['batting']
            game = batter['stats']['batting']
            batter_stats.append( {stat: postgame.get(stat, 0) - game.get(stat, 0) for stat in stats} )
            for stat in stats:
                batter_sums[stat] += batter_stats[-1][stat]
        batter_stats.append(batter_sums)
        return batter_stats
    return None

get_batters_pregrame_stats(box)

# %%
def get_final_score(line):
    """
    Extracts the final score from the line score.
    """
    if 'teams' in line and 'home' in line['teams'] and 'away' in line['teams']:
        home_runs = line['teams']['home']['runs']
        away_runs = line['teams']['away']['runs']
        return {
            'home_runs': home_runs,
            'away_runs': away_runs,
            'home_win': home_runs > away_runs
        }
    return None

# %%
def load_marcel_projections_csv():
    return pd.read_csv('twothree_marcel_hitters.csv')

marcel_projections = load_marcel_projections_csv()
marcel_projections.head()


# %% [markdown]
"""
## Feature Engineering

Now let's create features for our moneyline prediction model using team standings and pitcher data.
"""

# %%
LOG_EXPONENT = 10
import math

def get_team_elo(team_wins, team_losses, log_exp=LOG_EXPONENT):
    """
    Calculates the Elo rating for a team based on its wins and losses.
    """
    games_played = team_wins + team_losses
    if team_wins + team_losses == 0:
        return 1500  # Default Elo rating if no games played
    return 1500 + (team_wins / games_played) * (math.log((games_played + log_exp) /log_exp) + 1) * 100 # Simplified Elo calculation

def get_starter_averages(starter_stats):
    """
    Calculates the per inning stats for a starting pitcher.
    """
    if not starter_stats or starter_stats.get('outs', 0) == 0:
        return {stat: None for stat in starter_stats}
    total_innings = starter_stats.get('outs', 0) / 3
    return {stat: starter_stats.get(stat, 0) / total_innings * 9 for stat in starter_stats if stat not in ['outs', 'gamesPlayed', 'gamesStarted']}

# TODO: perhaps a non-weighted average would be sharper here, but for now...
def get_batting_team_averages(batter_stats):
    """
    Calculates the per game stats for a batter.
    """
    if not batter_stats or batter_stats.get('plateAppearances', 0) == 0:
        return {stat: None for stat in batter_stats}
    plateAppearances = batter_stats.get('plateAppearances', 1)  # Avoid division by zero
    return {stat: batter_stats.get(stat, 0) / plateAppearances for stat in batter_stats if stat not in ['gamesPlayed', 'battingOrder']}

# %%
def get_dataframe(game_pks):
    """
    Fetches box scores and line scores for a list of game Pks and returns a DataFrame.
    """
    data = []
    for game_pk in game_pks:
        box = get_box_score(game_pk)
        line = get_line_score(game_pk)
        if box and line:
            home_team_record = get_team_postgame_record(box, home=True)
            away_team_record = get_team_postgame_record(box, home=False)
            home_starter_stats = get_starter_pregame_stats(box, home=True)
            away_starter_stats = get_starter_pregame_stats(box, home=False)
            home_batters_stats = get_batters_pregrame_stats(box, home=True)
            away_batters_stats = get_batters_pregrame_stats(box, home=False)

            final_score = get_final_score(line)

            if final_score['home_win']:
                home_team_record['wins'] -= 1
                away_team_record['losses'] -= 1
            else:
                home_team_record['losses'] -= 1
                away_team_record['wins'] -= 1

            data.append({
                'gamePk': game_pk,
                'home_wins': home_team_record['wins'],
                'home_losses': home_team_record['losses'],
                'home_winpercentage': home_team_record['wins'] / (home_team_record['wins'] + home_team_record['losses']) if (home_team_record['wins'] + home_team_record['losses']) > 0 else 0,
                'home_elo': get_team_elo(home_team_record['wins'], home_team_record['losses']),
                'away_elo': get_team_elo(away_team_record['wins'], away_team_record['losses']),
                'away_wins': away_team_record['wins'],
                'away_losses': away_team_record['losses'],
                'away_winpercentage': away_team_record['wins'] / (away_team_record['wins'] + away_team_record['losses']) if (away_team_record['wins'] + away_team_record['losses']) > 0 else 0,
                'away_runs': final_score['away_runs'],
                'home_win': final_score['home_win'],
                **{f'home_sp_{stat}': home_starter_stats.get(stat, None) for stat in ['earnedRuns', 'hits', 'baseOnBalls', 'homeRuns', 'outs']},
                **{f'away_sp_{stat}': away_starter_stats.get(stat, None) for stat in ['earnedRuns', 'hits', 'baseOnBalls', 'homeRuns', 'outs']},
                **{f'home_{i}_{stat}': home_batters_stats[i].get(stat, None) for stat in ['hits', 'totalBases', 'homeRuns', 'plateAppearances', 'baseOnBalls'] for i in range(len(home_batters_stats) - 1)},
                **{f'away_{i}_{stat}': away_batters_stats[i].get(stat, None) for stat in ['hits', 'totalBases', 'homeRuns', 'plateAppearances', 'baseOnBalls'] for i in range(len(away_batters_stats) - 1)}
            })
    
    return pd.DataFrame(data)

twothree_df = get_dataframe(twothree_gamepks)  # Example usage to fetch DataFrame for the 2023 season

# %%
# Save dataframe to CSV
twothree_df.to_csv('twothree_df.csv', index=False)  # Save DataFrame to CSV file

# %%
# Get it back from CSV
twothree_df = pd.read_csv('twothree_df.csv')  # Load DataFrame from CSV file
twothree_df['home_sp_baseOnBalls'] = twothree_df['home_baseOnBalls']
twothree_df['home_sp_hits'] = twothree_df['home_hits']
#same but for away
twothree_df['away_sp_baseOnBalls'] = twothree_df['away_baseOnBalls']
twothree_df['away_sp_hits'] = twothree_df['away_hits']

# %%
bayesian = {
    'earnedRuns': 9,
    'sp_hits': 17,
    'sp_baseOnBalls': 8,
    'homeRuns': 3,
    'outs': 54,
    'wh': 25,  # Weighted average for WHIP
    
    #Batting stats
    'hits': 14,
    'totalBases': 23,
    'homeRuns': 1,
    'plateAppearances': 60,
    'baseOnBalls': 4
}

BAYESIAN_MULTIPLIER = 1  # Adjust this multiplier as needed for your model

for x in bayesian:
    bayesian[x] *= BAYESIAN_MULTIPLIER

# %%
twothree_df['home_era'] = (twothree_df['home_earnedRuns']  + bayesian['earnedRuns']) / ((twothree_df['home_outs'] + bayesian['outs']) / 27)
twothree_df['away_era'] = (twothree_df['away_earnedRuns'] + bayesian['earnedRuns']) / ((twothree_df['away_outs'] + bayesian['outs']) / 27)
twothree_df['home_whip9'] = (twothree_df['home_sp_hits'] + twothree_df['home_sp_baseOnBalls'] + bayesian['wh']) / ((twothree_df['home_outs'] + bayesian['outs']) / 3)
twothree_df['away_whip9'] = (twothree_df['away_sp_hits'] + twothree_df['away_sp_baseOnBalls'] + bayesian['wh']) / ((twothree_df['away_outs'] + bayesian['outs']) / 3)

for batting_stat in ['hits', 'totalBases', 'homeRuns', 'baseOnBalls']:
    twothree_df[f'home_{batting_stat}'] = sum([(twothree_df[f'home_{i}_{batting_stat}'] + bayesian[batting_stat]) / (twothree_df[f'home_{i}_plateAppearances'] + bayesian['plateAppearances']) for i in range(9)])/9
    twothree_df[f'away_{batting_stat}'] = sum([(twothree_df[f'away_{i}_{batting_stat}'] + bayesian[batting_stat]) / (twothree_df[f'away_{i}_plateAppearances'] + bayesian['plateAppearances']) for i in range(9)])/9

# %%
# Get rid of NaN values (note: this is when a pitcher makes his season debut)
twothree_df = twothree_df.dropna(subset=['home_era', 'away_era'])
twothree_df.head()  # Display the first few rows of the DataFrame

twothree_df['era_diff'] = twothree_df['home_era'] - twothree_df['away_era']
twothree_df['wp_diff'] = twothree_df['home_winpercentage'] - twothree_df['away_winpercentage']
twothree_df['elo_diff'] = twothree_df['home_elo'] - twothree_df['away_elo']
twothree_df['whip9_diff'] = twothree_df['home_whip9'] - twothree_df['away_whip9']
twothree_df['obp_diff'] = twothree_df['home_hits'] + twothree_df['home_baseOnBalls'] - twothree_df['away_hits'] - twothree_df['away_baseOnBalls']
twothree_df['slg_diff'] = twothree_df['home_totalBases'] - twothree_df['away_totalBases']
twothree_df['HRR_diff'] = twothree_df['home_homeRuns'] - twothree_df['away_homeRuns']
twothree_df.head()  # Display the updated DataFrame with new features

# %%
print(twothree_df['home_era'].mean())
print(twothree_df['away_era'].mean())
print(twothree_df['home_whip9'].mean())
print(twothree_df['away_whip9'].mean())

# %% [markdown]
"""
## Model Training and Evaluation

Now let's train regression models to predict the probability of home team wins (which can be converted to moneylines).
"""

# %%
feature_cols = [
    'era_diff','whip9_diff', 'obp_diff', 'slg_diff', 'wp_diff', 'elo_diff', 'HRR_diff',
]

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

def train_moneyline_models(features_df):
    """Train models to predict home team win probability"""
    if features_df is None or len(features_df) < 10:
        print("Insufficient data for training")
        return None, None
    
    # Select feature columns (exclude metadata and target)
    X = features_df[feature_cols]
    y = features_df['home_win']

    print(f"Training on {len(X)} games with {len(feature_cols)} features")
    print(f"Home team win rate: {y.mean():.3f}")
    
    # Normalize features
    X = (X - X.mean()) / X.std()

    # Split data
    if len(X) > 20:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random.randint(1, 10000))
    else:
        # Use all data for training if dataset is small
        X_train, X_test, y_train, y_test = X, X, y, y
        print("Small dataset - using all data for both training and testing")
    
    # Train models
    models = {}

    # KNN Regressor
    knn_model = KNeighborsRegressor(n_neighbors=50)
    knn_model.fit(X_train, y_train)
    models['KNN Regressor'] = knn_model
    
    # Logistic Regression
    lgr_model = LogisticRegression()
    lgr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lgr_model
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=random.randint(1, 10000))
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            # For classification models
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            # For regression models
            y_pred = model.predict(X_test)
        # Clip predictions to [0, 1] range for probability
        y_pred = np.clip(y_pred, 0, 1)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Classification accuracy (using 0.5 threshold)
        y_pred_class = (y_pred > 0.5).astype(int)
        accuracy = (y_pred_class == y_test).mean()
        
        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'R²': r2,
            'Accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"\n{name} Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    return models, results

# Train models
all_models, all_results = list(), list()
for i in range(10):
    print(f"Training iteration {i+1}")
    models, results = train_moneyline_models(twothree_df)
    all_models.append(models)
    all_results.append(results)

print("After training 10 iterations:")
#Print average results
avg_results = {name: {metric: np.mean([result[name][metric] for result in all_results]) for metric in all_results[0][name]} for name in all_results[0]}
for name, metrics in avg_results.items():
    print(f"\n{name} Average Results:")
    for metric, value in metrics.items():
        if metric == 'predictions':
            continue
        print(f"  {metric}: {value:.4f}")

# %%
def train_elo_model(feature_df):
    elo_df = pd.DataFrame()

    home_dict = {col: feature_df[f'home_pregame_{col}'] for col in feature_cols}
    home_dict['win'] = feature_df['home_win']

    away_dict = {col: feature_df[f'away_pregame_{col}'] for col in feature_cols}
    away_dict['win'] = 1 - feature_df['home_win']  # Invert home win to get away win

    elo_df = pd.concat([pd.DataFrame(home_dict), pd.DataFrame(away_dict)], ignore_index=True)
    print(f"Training Elo model on {len(elo_df)} games with {len(feature_cols)} features")

# %%
# Show each coefficients for the Logistic Regression model
if 'models' in locals() and 'Logistic Regression' in models:
    print("\nLogistic Regression Coefficients:")
    for feature, coef in zip(feature_cols, models['Logistic Regression'].coef_[0]):
        print(f"{feature}: {coef:.4f}")
    print("LogisticRegression constant:", models['Logistic Regression'].intercept_[0])
    print("Expected Home Win Probability based on Intercept:", 
          1 / (1 + np.exp(-models['Logistic Regression'].intercept_[0])))

    # Plotting feature importance for Random Forest model
    if 'Random Forest' in models:
        rf_importances = models['Random Forest'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

# %% [markdown]
"""
## Moneyline Conversion and Predictions

Convert win probabilities to moneylines and make predictions for upcoming games.
"""

# %%
def probability_to_moneyline(prob):
    """Convert win probability to American moneyline odds"""
    if prob <= 0 or prob >= 1:
        return None
    
    if prob > 0.5:
        # Favorite (negative odds)
        return -int(prob / (1 - prob) * 100)
    else:
        # Underdog (positive odds)
        return int((1 - prob) / prob * 100)

def moneyline_to_probability(moneyline):
    """Convert American moneyline odds to win probability"""
    if moneyline > 0:
        return 100 / (moneyline + 100)
    else:
        return abs(moneyline) / (abs(moneyline) + 100)

def get_todays_games():
    """Get today's scheduled games for predictions"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        schedule = mlbstatsapi.schedule(date=today)
        
        todays_games = []
        for game in schedule:
            if game['status']['statusCode'] in ['S', 'P']:  # Scheduled or Pre-game
                game_data = {
                    'game_id': game['gamePk'],
                    'date': today,
                    'home_team_id': game['teams']['home']['team']['id'],
                    'home_team_name': game['teams']['home']['team']['name'],
                    'away_team_id': game['teams']['away']['team']['id'],
                    'away_team_name': game['teams']['away']['team']['name'],
                    'game_time': game.get('gameDate', 'TBD')
                }
                todays_games.append(game_data)
        
        return pd.DataFrame(todays_games)
    except Exception as e:
        print(f"Error getting today's games: {e}")
        return None

# Get today's games and make predictions
if 'models' in locals() and models is not None:
    todays_games = get_todays_games()
    
    if todays_games is not None and len(todays_games) > 0:
        print(f"Found {len(todays_games)} games scheduled for today:")
        print("Note: This example would need current team data to make actual predictions")
    else:
        print("No games scheduled for today or could not retrieve schedule")
else:
    print("Models not trained - cannot make predictions")

# %%
# Train a single model for demonstration
models, results = train_moneyline_models(twothree_df)

# %%
