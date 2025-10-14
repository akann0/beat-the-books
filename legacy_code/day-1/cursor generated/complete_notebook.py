import json

# Load the notebook
with open('mlb_moneyline_regression_notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Add Bayesian priors section
cell_content = '''bayesian = {
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
    bayesian[x] *= BAYESIAN_MULTIPLIER'''

nb['cells'].append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': cell_content
})

# Add ERA and feature calculations
cell_content = '''twothree_df['home_era'] = (twothree_df['home_earnedRuns']  + bayesian['earnedRuns']) / ((twothree_df['home_outs'] + bayesian['outs']) / 27)
twothree_df['away_era'] = (twothree_df['away_earnedRuns'] + bayesian['earnedRuns']) / ((twothree_df['away_outs'] + bayesian['outs']) / 27)
twothree_df['home_whip9'] = (twothree_df['home_sp_hits'] + twothree_df['home_sp_baseOnBalls'] + bayesian['wh']) / ((twothree_df['home_outs'] + bayesian['outs']) / 3)
twothree_df['away_whip9'] = (twothree_df['away_sp_hits'] + twothree_df['away_sp_baseOnBalls'] + bayesian['wh']) / ((twothree_df['away_outs'] + bayesian['outs']) / 3)

for batting_stat in ['hits', 'totalBases', 'homeRuns', 'baseOnBalls']:
    twothree_df[f'home_{batting_stat}'] = sum([(twothree_df[f'home_{i}_{batting_stat}'] + bayesian[batting_stat]) / (twothree_df[f'home_{i}_plateAppearances'] + bayesian['plateAppearances']) for i in range(9)])/9
    twothree_df[f'away_{batting_stat}'] = sum([(twothree_df[f'away_{i}_{batting_stat}'] + bayesian[batting_stat]) / (twothree_df[f'away_{i}_plateAppearances'] + bayesian['plateAppearances']) for i in range(9)])/9'''

nb['cells'].append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': cell_content
})

# Add feature differences calculation
cell_content = '''# Get rid of NaN values (note: this is when a pitcher makes his season debut)
twothree_df = twothree_df.dropna(subset=['home_era', 'away_era'])
twothree_df.head()  # Display the first few rows of the DataFrame

twothree_df['era_diff'] = twothree_df['home_era'] - twothree_df['away_era']
twothree_df['wp_diff'] = twothree_df['home_winpercentage'] - twothree_df['away_winpercentage']
twothree_df['elo_diff'] = twothree_df['home_elo'] - twothree_df['away_elo']
twothree_df['whip9_diff'] = twothree_df['home_whip9'] - twothree_df['away_whip9']
twothree_df['obp_diff'] = twothree_df['home_hits'] + twothree_df['home_baseOnBalls'] - twothree_df['away_hits'] - twothree_df['away_baseOnBalls']
twothree_df['slg_diff'] = twothree_df['home_totalBases'] - twothree_df['away_totalBases']
twothree_df['HRR_diff'] = twothree_df['home_homeRuns'] - twothree_df['away_homeRuns']
twothree_df.head()  # Display the updated DataFrame with new features'''

nb['cells'].append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': cell_content
})

# Add basic statistics
cell_content = '''print(twothree_df['home_era'].mean())
print(twothree_df['away_era'].mean())
print(twothree_df['home_whip9'].mean())
print(twothree_df['away_whip9'].mean())'''

nb['cells'].append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': cell_content
})

# Add model training section
nb['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## Model Training and Evaluation\n\nNow let\\'s train regression models to predict the probability of home team wins (which can be converted to moneylines).'
})

# Add the main model training function
cell_content = '''feature_cols = [
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

        print(f"\\n{name} Results:")
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
    print(f"\\n{name} Average Results:")
    for metric, value in metrics.items():
        if metric == 'predictions':
            continue
        print(f"  {metric}: {value:.4f}")'''

nb['cells'].append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': cell_content
})

# Add final section for predictions
nb['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## Moneyline Conversion and Predictions\n\nConvert win probabilities to moneylines and make predictions for upcoming games.'
})

# Add prediction functions
cell_content = '''def probability_to_moneyline(prob):
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

# Train a single model for demonstration
models, results = train_moneyline_models(twothree_df)'''

nb['cells'].append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': cell_content
})

# Save the notebook
with open('mlb_moneyline_regression_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Completed the notebook with all cells!')
