mlbgame module
mlbgame is a Python API to retrieve and read MLB GameDay data. mlbgame works with real time data, getting information as games are being played.

mlbgame uses the same data that MLB GameDay uses, and therefore is updated as soon as something happens in a game.

mlbgame documentation

mlbgame on Github (Source Code)

If you have a question or need help, the quickest way to get a response is to file an issue on the Github issue tracker

mlbgame's submodules should not really be used other than as used by the main functions of the package (in __init__.py).

Use of mlbgame must follow the terms stated in the license and on mlb.com.

Installation
mlbgame is in the [Python Package Index (PyPI)] (http://pypi.python.org/pypi/mlbgame/). Installing with pip is recommended for all systems.

mlbgame can be installed by running:

pip install mlbgame
Alternatively, the latest release of mlbgame can be downloaded as a zip or tarball. If you do not install with pip, you must also install lmxl as specified in setup.py.

If you want to help develop mlbgame, you must also install the dev dependencies, which can be done by running pip install -e .[dev] from within the directory.

Examples
Here is a quick teaser to find the scores of all home Mets games for the month of June, 2015:

#!python
from __future__ import print_function
import mlbgame

month = mlbgame.games(2015, 6, home='Mets')
games = mlbgame.combine_games(month)
for game in games:
    print(game)
And the output is:

Giants (5) at Mets (0)
Giants (8) at Mets (5)
Giants (4) at Mets (5)
Braves (3) at Mets (5)
Braves (5) at Mets (3)
Braves (8) at Mets (10)
Blue Jays (3) at Mets (4)
Blue Jays (2) at Mets (3)
Reds (1) at Mets (2)
Reds (1) at Mets (2)
Reds (1) at Mets (2)
Reds (2) at Mets (7)
Cubs (1) at Mets (0)
Maybe you want to know the pitchers for the Royals game on April 30th, 2015:

#!python
from __future__ import print_function
import mlbgame

day = mlbgame.day(2015, 4, 12, home='Royals', away='Royals')
game = day[0]
output = 'Winning pitcher: %s (%s) - Losing Pitcher: %s (%s)'
print(output % (game.w_pitcher, game.w_team, game.l_pitcher, game.l_team))
And the output is:

Winning pitcher: Y. Ventura (Royals) - Losing Pitcher: C. Wilson (Angels)
You can easily print a list of the Mets batters in the final game of the 2015 World Series:

#!python
from __future__ import print_function
import mlbgame

game = mlbgame.day(2015, 11, 1, home='Mets')[0]
stats = mlbgame.player_stats(game.game_id)
for player in stats.home_batting:
    print(player)
And the output is:

Curtis Granderson (RF)
David Wright (3B)
Daniel Murphy (2B)
Yoenis Cespedes (CF)
Juan Lagares (CF)
Lucas Duda (1B)
Travis d'Arnaud (C)
Michael Conforto (LF)
Wilmer Flores (SS)
Matt Harvey (P)
Jeurys Familia (P)
Kelly Johnson (PH)
Jonathon Niese (P)
Addison Reed (P)
Bartolo Colon (P)
Show source ≡

Module variables
var VERSION
Installed version of mlbgame.

Functions
def box_score(	game_id)
Return box score for game matching the game id.

Show source ≡

def combine_games(	games)
Combines games from multiple days into a single list.

Show source ≡

def day(	year, month, day, home=None, away=None)
Return a list of games for a certain day.

If the home and away team are the same, it will return the game(s) for that team.

Show source ≡

def game_events(	game_id)
Return dictionary of game events for game matching the game id.

Show source ≡

def games(	years, months=None, days=None, home=None, away=None)
Return a list of lists of games for multiple days.

If home and away are the same team, it will return all games for that team.

Show source ≡

def important_dates(	year=None)
Return ImportantDates object that contains MLB important dates

Show source ≡

def injury(	)
Return Injuries object that contains injury info

Show source ≡

def league(	)
Return Info object that contains league information

Show source ≡

def overview(	game_id)
Return Overview object that contains game information.

Show source ≡

def player_stats(	game_id)
Return dictionary of player stats for game matching the game id.

Hide source ≢

def player_stats(game_id):
    """Return dictionary of player stats for game matching the game id."""
    # get information for that game
    data = mlbgame.stats.player_stats(game_id)
    return mlbgame.stats.Stats(data, game_id, True)
def players(	game_id)
Return list players/coaches/umpires for game matching the game id.

Show source ≡

def roster(	team_id)
Return Roster object that contains roster info for a team

Show source ≡

def standings(	date=datetime.datetime(2018, 4, 16, 16, 39, 19, 883334))
Return Standings object that contains standings info

date should be a datetime object, leave empty to get current standings

Show source ≡

def team_stats(	game_id)
Return dictionary of team stats for game matching the game id.

Show source ≡

def teams(	)
Return list of Info objects for each team

Show source ≡

Sub-modules
mlbgame.data
This module gets the XML data that other functions use. It checks if the data is cached first, and if not, gets the data from mlb.com.

mlbgame.events
Module that is used for getting the events that occured throughout games.

mlbgame.game
Module that is used for getting basic information about a game such as the scoreboard and the box score.

mlbgame.info
Module that is used for getting information about the (MLB) league and the teams in it.

mlbgame.object
Module that is used for holding basic objects

mlbgame.stats
Module that controls getting stats and creating objects to hold that information.

mlbgame.version