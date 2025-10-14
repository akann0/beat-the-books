"""
MLB Moneyline Database Builder

This script uses the methods from mlb-moneyline-regression to create a complete database
for training moneyline prediction models.
"""

import mlbstatsapi
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

"""
Advanced Research Questions:

Does home field advantage affect certain types of teams more than others?
Does bullpen rest impact game outcomes?
How do weather conditions (temperature, humidity) correlate with game outcomes?
How do team travel schedules (e.g., long road trips) impact performance?
How do injuries to key players affect game outcomes?
How do different ballparks affect team performance?
How do team matchups (e.g., left-handed vs. right-handed pitchers) affect game outcomes?
How do team performance trends (e.g., winning streaks) affect game outcomes?
"""

class MLBDatabaseBuilder:
    def __init__(self, db_path='mlb_moneyline.db'):
        self.db_path = db_path

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

    def get_starter_pregame_stats(box, home=True):
        """
        Extracts the starting pitcher's ERA before the game from the box score.
        """
        if 'teams' in box and 'pitchers' in box['teams']['home']:
            pitchers = box['teams']['home']['pitchers'] if home else box['teams']['away']['pitchers']
            starter = box['teams']['home']['players'][f'ID{pitchers[0]}'] if home else box['teams']['away']['players'][f'ID{pitchers[0]}']
            postgame_erallowed = int(starter['seasonStats']['pitching']['earnedRuns'])
            postgame_outs = int(starter['seasonStats']['pitching']['outs'])

            game_erallowed = int(starter['stats']['pitching']['earnedRuns'])
            game_outs = int(starter['stats']['pitching']['outs'])

            if (postgame_outs - game_outs) > 0:
                pregame_era = ((postgame_erallowed - game_erallowed) / (postgame_outs - game_outs)) * 27
                return {'pregame_era': pregame_era}
            else:
                return {'pregame_era': None}
        return None
    
    def get_team_elo(team_wins, team_losses):
        """
        Calculates the Elo rating for a team based on its wins and losses.
        """
        if team_wins + team_losses == 0:
            return 1500
        return 1500 + (team_wins - team_losses) * 20