#for dota api: https://gist.github.com/essramos/dbac40593b64e2193f2be68232f86b58
#adjust for use on league games

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:12:56 2018

@author: M29480
"""
# data source: http://oracleselixir.com/match-data/

#### Assume Group A plays D and B plays C -> unsure if true

# Next steps are to figure out visualizations and set up model for finals
# Function currently doesn't allow for editing in the middle of a best-of-five matchup
import pandas as pd
import numpy as np
import math
 
# Function to calculate Elo
def Probability(rating1, rating2):
 
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))
 
def EloRating(Ra, Rb, K, d, kill, death):
  
    Pb = Probability(Ra, Rb)
    Pa = Probability(Rb, Ra)
    mov = kill - death
    adj = np.log(abs(mov) + 1)
    
    if (d == 1) :
        Ra = Ra + adj * K * (1 - Pa)
        Rb = Rb + adj * K * (0 - Pb)
     
    else :
        Ra = Ra + adj * K * (0 - Pa)
        Rb = Rb + adj * K * (1 - Pb)
     
    return Ra, Rb    

def bestofFive(team1, team2):
    elo1 = wcTeam['elo'][(wcTeam['team'] == team1)].values[0]
    elo2 = wcTeam['elo'][(wcTeam['team'] == team2)].values[0]
    prob = Probability(elo2, elo1)
    prob = prob**3 + 3*prob**3*(1-prob) + 6*prob**3*(1-prob)**2
    return prob

# Load league files
summer18 = pd.read_csv('2018Summer.csv')
summer18['year'] = 2018
spring18 = pd.read_csv('2018Spring.csv')
spring18['year'] = 2018
complete17 = pd.read_csv('2017Complete.csv')
complete17['year']=2017
complete16 = pd.read_csv('2016Complete.csv')
complete16['year']= 2016
teamid = pd.read_csv('teamIDs.csv')
teamid['teamid'] = teamid['teamid'].astype('str')
leagueList = pd.read_csv('leagueList.csv')
worldResults = pd.read_csv('wcPlayinResults.csv')


# Concatenate league files
full_lol = pd.concat([pd.concat([pd.concat([complete16, complete17]), spring18]), summer18])
teamids = teamid[['teamid', 'team']]
full_lol = full_lol.merge(teamids, on='team')
full_lol['teamid'] = full_lol['teamid'].astype('str')

full_lol = full_lol[['split', 'year', 'league', 'gameid', 'date', 'week', 'game', 'playerid', 'teamid', 'team', 'result', 'k', 'd']]
full_lol = full_lol[(full_lol['playerid'] == 100) | (full_lol['playerid'] == 200)]

full_lol['date'] = pd.to_datetime(full_lol['date'])
full_lol = full_lol.sort_values(by=['date'])

# Build dataframe of teams
teams = full_lol['teamid'].unique()
teams = np.append(teams, '204')
teams = np.append(teams, '205')
teamELO = pd.DataFrame(teams)
teamELO = teamELO.rename(index=str, columns = {0: 'teamid'})
teamid = teamid[['teamid']]
teamid = teamid['teamid'].unique()
teamid=pd.DataFrame(teamid)
teamid = teamid.rename(index=str, columns = {0: 'teamid'})
teamELO = teamid.merge(teamELO, on='teamid')
leagueList['teamid'] = leagueList['teamid'].astype('str')
teamELO = teamELO.merge(leagueList, on='teamid')


teamELO['elo'] = 1500 #1600 is an abitrary number and based on other popular ELO systems
teamELO['elo'][(teamELO['league'] == 'LCK')] = 1750
teamELO['elo'][(teamELO['league'] == 'LPL')] = 1750

gameIDs = full_lol['gameid'].unique()

leagues = full_lol['league'].unique()
leagues=pd.DataFrame(leagues) 

full_lol['opp'] = ""
for game in gameIDs:
    team2 = full_lol['teamid'][(full_lol['playerid'] == 200) & (full_lol['gameid'] == game)]
    full_lol['opp'][(full_lol['playerid'] == 100) & (full_lol['gameid'] == game)] = team2.values[0]

##### This would be a good place to automate model to speed up
    
# Down select to one row per game
full_lol = full_lol[full_lol['playerid'] == 100]

worldResults = worldResults.merge(teamids, on = 'team')
oppid = teamids
oppid = oppid.rename(index=str, columns = {'team': 'opp', 'teamid': 'oppid'})
worldResults = worldResults.merge(oppid, on='opp')
worldResults = worldResults.drop(['opp', 'group'], axis=1)
worldResults = worldResults.rename(index=str, columns = {'oppid': 'opp'})
worldResults = worldResults[(worldResults['played'] == 'X')]
worldResults = worldResults.drop(['played'], axis=1)
full_lol = full_lol.drop(['playerid'], axis=1)
worldResults['teamid'] = worldResults['teamid'].astype('str')

full_lol = pd.concat([full_lol, worldResults])
full_lol['date'] = pd.to_datetime(full_lol['date'])
full_lol = full_lol.sort_values(by=['date'])


# Cycle through all games and update elo rankings accordingly
k = 15 #k needs to be tuned
for game in full_lol['gameid']:
    team1 = teamELO['teamid'] == full_lol['teamid'][(full_lol['gameid'] == game)].values[0]
    team2 = teamELO['teamid'] == full_lol['opp'][(full_lol['gameid'] == game)].values[0]
    elo1 = teamELO['elo'][team1].values[0]
    elo2 = teamELO['elo'][team2].values[0]
    result = full_lol['result'][(full_lol['gameid'] == game)].values[0]
    kill = full_lol['k'][(full_lol['gameid'] == game)].values[0]
    death = full_lol['d'][(full_lol['gameid'] == game)].values[0]

    newelo1, newelo2 = EloRating(elo1, elo2, k, result, kill, death)
    
    teamELO['elo'][team1] = newelo1
    teamELO['elo'][team2] = newelo2

# Build standings dataframe (for validation)
games = full_lol['teamid'].value_counts()
wins = full_lol.groupby('teamid')['result'].sum()
record = pd.concat([games, wins], axis=1)
record['losses'] = record['teamid'] - record['result']
record = record[['result', 'losses']]
record = record.rename(index=str, columns = {'result': 'Wins', 'losses': 'Losses'})

teamELO = teamELO.set_index('teamid')
teamELO = pd.concat([teamELO, record], axis=1)

teamELO = teamELO.reset_index()
teamELO = teamELO.rename(index=str, columns={'index': 'teamid'})
teamELO = teamELO.merge(teamid, on='teamid')



teamELO = teamELO.merge(teamids, on='teamid')
teamELO = teamELO[['team', 'teamid', 'elo', 'Wins', 'Losses', 'league']] #, 'numberOfLeagues', 'lastLeagues', 'lastYearPlayed']]

teamELO.to_csv('teamelo.csv')

wcTeam = ['Ascension Gaming', 'Dire Wolves', 'Kaos Latin Gamers', 'DetonatioN FocusMe', 'SuperMassive', 
          'Infinity eSports', 'Gambit Esports', 'KaBuM e-Sports', 'Cloud9', 'G2 Esports', 'EDward Gaming', 
          'MAD Team', 'Flash Wolves', '100 Thieves', 'Team Liquid', 'Vitality', 
          'Fnatic', 'Invictus Gaming', 'Royal Never Give Up', 'Gen.G', 'Afreeca Freecs', 'KT Rolster', 
          'Phong Vu Buffalo', 'G-Rex']

wcTeam = pd.DataFrame(wcTeam)
wcTeam = wcTeam.rename(index=str, columns={0: 'team'})
wcTeam = wcTeam.merge(teamELO, on='team')

wcGroup = pd.read_csv('wcGroups.csv')
wcTeam = wcTeam.merge(wcGroup, on='team')


playTeams = wcTeam[(wcTeam.group == 'PA') | (wcTeam.group == 'PB') | (wcTeam.group == 'PC') | (wcTeam.group == 'PD')]
playTeams['seed1Prob'] = ''
playTeams['seed2Prob'] = ''
playTeams['advProb'] = ''
playTeams['condWins'] = ''
playTeams['mainProb'] = ''
playTeams['seed'] = ''


worldResults = pd.read_csv('wcPlayinResults.csv')

worldResults['prob'] = ''
worldResults['pred'] = ''
for game in worldResults.gameid:
    worldResults['pred'][(worldResults.gameid == game)] = Probability(playTeams['elo'][(playTeams['team'] == worldResults['opp'][(worldResults.gameid == game)].values[0])].values[0], playTeams['elo'][(playTeams['team'] == worldResults['team'][(worldResults.gameid == game)].values[0])])
    if worldResults.played[(worldResults.gameid == game)].values[0] == 'X':
        worldResults['prob'][(worldResults.gameid == game)] = worldResults['result'][(worldResults.gameid == game)]
    else:
        worldResults['prob'][(worldResults.gameid == game)] = ''
        worldResults['prob'][(worldResults.gameid == game)] = Probability(playTeams['elo'][(playTeams['team'] == worldResults['opp'][(worldResults.gameid == game)].values[0])].values[0], playTeams['elo'][(playTeams['team'] == worldResults['team'][(worldResults.gameid == game)].values[0])])

for group in playTeams.group.unique():
    team1 = playTeams['team'][(playTeams['teamNum'] == 1) & (playTeams['group'] == group)].values[0]
    team2 = playTeams['team'][(playTeams['teamNum'] == 2) & (playTeams['group'] == group)].values[0]
    team3 = playTeams['team'][(playTeams['teamNum'] == 3) & (playTeams['group'] == group)].values[0]
    game1 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team1) & (worldResults.opp == team2)].values[0]
    game2 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team2) & (worldResults.opp == team3)].values[0]
    game3 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team1) & (worldResults.opp == team3)].values[0]
    game4 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team1) & (worldResults.opp == team2)].values[0]
    game5 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team2) & (worldResults.opp == team3)].values[0]
    game6 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team1) & (worldResults.opp == team3)].values[0]
    prob1 = game1 + game3 + game4 + game5
    prob2 = (1- game1) + game2 + (1 - game4) + game6
    prob3 = (1 - game2) + (1 - game3) + (1 - game5) + (1 - game6)
    playTeams['condWins'][(playTeams['team'] == team3)] = prob3
    playTeams['condWins'][(playTeams['team'] == team2)] = prob2
    playTeams['condWins'][(playTeams['team'] == team1)] = prob1
    new_prob1 = prob1 / (prob1 + prob2 + prob3)
    new_prob2 = prob2 / (prob1 + prob2 + prob3)
    new_prob3 = prob3 / (prob1 + prob2 + prob3)
    playTeams['seed1Prob'][(playTeams['team'] == team3)] = new_prob3
    playTeams['seed1Prob'][(playTeams['team'] == team2)] = new_prob2
    playTeams['seed1Prob'][(playTeams['team'] == team1)] = new_prob1
    teams=[team1, team2, team3]
    seed1 = playTeams['team'][(playTeams['seed1Prob'] == max(playTeams['seed1Prob'][(playTeams['team'] == team1)].values[0], playTeams['seed1Prob'][(playTeams['team'] == team2)].values[0], 
                      playTeams['seed1Prob'][(playTeams['team'] == team3)].values[0]))].values[0]
    teams.remove(seed1)
    team1 = teams[0]
    team2 = teams[1]
    new_prob1 = playTeams['condWins'][(playTeams['team'] == team1)].values[0]/(playTeams['condWins'][(playTeams['team'] == team2)].values[0] + playTeams['condWins'][(playTeams['team'] == team1)].values[0])
    new_prob2 = playTeams['condWins'][(playTeams['team'] == team2)].values[0]/(playTeams['condWins'][(playTeams['team'] == team2)].values[0] + playTeams['condWins'][(playTeams['team'] == team1)].values[0])
    playTeams['seed2Prob'][(playTeams['team'] == team2)] = new_prob2
    playTeams['seed2Prob'][(playTeams['team'] == team1)] = new_prob1
    seed2 = playTeams['team'][(playTeams['seed2Prob'] == max(playTeams['seed2Prob'][(playTeams['team'] == team1)].values[0], 
                      playTeams['seed2Prob'][(playTeams['team'] == team2)].values[0]))].values[0]
    teams.remove(seed2)
    seed3 = teams[0]
    
    new_prob1 = playTeams['condWins'][(playTeams['team'] == seed1)].values[0]/(playTeams['condWins'][(playTeams['team'] == seed3)].values[0] + playTeams['condWins'][(playTeams['team'] == seed1)].values[0])
    new_prob3 = playTeams['condWins'][(playTeams['team'] == seed3)].values[0]/(playTeams['condWins'][(playTeams['team'] == seed3)].values[0] + playTeams['condWins'][(playTeams['team'] == seed1)].values[0])
    playTeams['seed2Prob'][(playTeams['team'] == seed1)] = (playTeams['seed1Prob'][(playTeams['team'] == seed2)].values[0] + playTeams['seed1Prob'][(playTeams['team'] == seed3)]).values[0] * new_prob1
    playTeams['seed2Prob'][(playTeams['team'] == seed2)] = (playTeams['seed1Prob'][(playTeams['team'] == seed1)].values[0] + playTeams['seed1Prob'][(playTeams['team'] == seed3)]).values[0] * playTeams['seed2Prob'][(playTeams['team'] == seed2)].values[0]
    playTeams['seed2Prob'][(playTeams['team'] == seed3)] = 1 - playTeams['seed2Prob'][(playTeams['team'] == seed1)].values[0] - playTeams['seed2Prob'][(playTeams['team'] == seed2)].values[0]
    playTeams['advProb'] = playTeams['seed1Prob'] + playTeams['seed2Prob']
         
    
for team1 in playTeams['team'][(playTeams['group'] == 'PA')]:
    prob = 0
    for team2 in playTeams['team'][(playTeams['group'] == 'PD')]:
        lhood = playTeams['seed1Prob'][(playTeams['team'] == team1)].values[0] * playTeams['seed2Prob'][(playTeams['team'] == team2)].values[0] + playTeams['seed1Prob'][(playTeams['team'] == team2)].values[0] * playTeams['seed2Prob'][(playTeams['team'] == team1)].values[0]
        prob = prob + bestofFive(team1, team2) * lhood
    playTeams['mainProb'][(playTeams['team'] == team1)] = prob
for team1 in playTeams['team'][(playTeams['group'] == 'PD')]:
    prob = 0
    for team2 in playTeams['team'][(playTeams['group'] == 'PA')]:
        lhood = playTeams['seed1Prob'][(playTeams['team'] == team1)].values[0] * playTeams['seed2Prob'][(playTeams['team'] == team2)].values[0] + playTeams['seed1Prob'][(playTeams['team'] == team2)].values[0] * playTeams['seed2Prob'][(playTeams['team'] == team1)].values[0]
        prob = prob +  bestofFive(team1, team2) * lhood
    playTeams['mainProb'][(playTeams['team'] == team1)] = prob        
for team1 in playTeams['team'][(playTeams['group'] == 'PB')]:
    prob = 0
    for team2 in playTeams['team'][(playTeams['group'] == 'PC')]:
        lhood = playTeams['seed1Prob'][(playTeams['team'] == team1)].values[0] * playTeams['seed2Prob'][(playTeams['team'] == team2)].values[0] + playTeams['seed1Prob'][(playTeams['team'] == team2)].values[0] * playTeams['seed2Prob'][(playTeams['team'] == team1)].values[0]
        prob = prob +  bestofFive(team1, team2) * lhood
    playTeams['mainProb'][(playTeams['team'] == team1)] = prob    
for team1 in playTeams['team'][(playTeams['group'] == 'PC')]:
    prob = 0
    for team2 in playTeams['team'][(playTeams['group'] == 'PB')]:
        lhood = playTeams['seed1Prob'][(playTeams['team'] == team1)].values[0] * playTeams['seed2Prob'][(playTeams['team'] == team2)].values[0] + playTeams['seed1Prob'][(playTeams['team'] == team2)].values[0] * playTeams['seed2Prob'][(playTeams['team'] == team1)].values[0]
        prob = prob +  bestofFive(team1, team2) * lhood 
    playTeams['mainProb'][(playTeams['team'] == team1)] = prob

# Visualizations    
tomorrowGames = worldResults[(worldResults.date == '10/3/2018')]
tomorrowGames = tomorrowGames[['date', 'group', 'team', 'opp', 'prob']]
tomorrowGames['prob'] = tomorrowGames['prob'].astype('float')
elos = teamELO[['team', 'elo']]
tomorrowGames = tomorrowGames.merge(elos, on='team')
tomorrowGames = tomorrowGames.rename(index=str, columns={'team': 'team1', 'elo': 'elo1', 'opp': 'team'})
tomorrowGames = tomorrowGames.merge(elos, on='team')
tomorrowGames = tomorrowGames.rename(index=str, columns={'team': 'team2', 'elo': 'elo2'})
tomorrowGames = tomorrowGames[['team1', 'elo1', 'team2', 'elo2', 'group', 'prob']]
tomorrowGames.to_html('test.html')
