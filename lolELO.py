#for dota api: https://gist.github.com/essramos/dbac40593b64e2193f2be68232f86b58
#adjust for use on league games

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:12:56 2018

@author: M29480
"""
# data source: http://oracleselixir.com/match-data/

#### Assume Group A plays D and B plays C -> unsure if true

# looked into dota data but more work is needed to get right -> not as clean as LoL data
# big push is to get LoL data ready for championship
# Use the play-in games as a final calibration (are SK and China bias too high or low?)
# Tables predicting all the outcomes of group play and overall table predicting likely winner 
# Will have it set up so that I can enter new results and re-run the projections every day
# Matchups end by 10am every day so should be able to post new projections by noon
# Should CarryGG launch on 10/8 or 10/9 to get ahead of the start?
# Our rankings match current betting markets pretty closely with 4 clear top teams

import pandas as pd
import numpy as np
import math


a1 = 1500
a2 = 1800

####### Need  to include number of permutations for each game (see the 2 in the best of 3 equation)

#Probability(a2, a1)**3 *(1 + (1-Probability(a2,a1)) + (1-Probability(a2,a1))**2)

Pro

#Probability(a2, a1)
 
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
#complete17 = complete17[complete17['league'] != 'WC']
complete16 = pd.read_csv('2016Complete.csv')
complete16['year']= 2016
teamid = pd.read_csv('teamIDs.csv')
teamid['teamid'] = teamid['teamid'].astype('str')
leagueList = pd.read_csv('leagueList.csv')


# Concatenate league files
#full_lol = pd.concat([complete16, complete17])
full_lol = pd.concat([pd.concat([pd.concat([complete16, complete17]), spring18]), summer18])
teamids = teamid[['teamid', 'team']]
full_lol = full_lol.merge(teamids, on='team')
full_lol['teamid'] = full_lol['teamid'].astype('str')

full_lol = full_lol[['split', 'year', 'league', 'gameid', 'date', 'week', 'game', 'playerid', 'teamid', 'team', 'result', 'k', 'd']]
full_lol = full_lol[(full_lol['playerid'] == 100) | (full_lol['playerid'] == 200)]

# Remove LPL since dates are bad
#full_lol = full_lol[full_lol['league'] != 'LPL']

full_lol['date'] = pd.to_datetime(full_lol['date'])
full_lol = full_lol.sort_values(by=['date'])

# Build dataframe of teams
teams = full_lol['teamid'].unique()
teams = np.append(teams, '204')
teams = np.append(teams, '205')
#eagueList= teamid[['teamid', 'league']]
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

# Cycle through all games and update elo rankings accordingly
gameIDs = full_lol['gameid'].unique()
k = 15 #k needs to be tuned
for game in gameIDs:
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

#teamELO = pd.read_csv('teamelo.csv')
'''
byLeague = full_lol[['teamid', 'league']]
byLeague = byLeague.drop_duplicates(keep='last')
byLeague = byLeague.groupby('teamid').count()
byLeague = byLeague.reset_index()
byLeague = byLeague.rename(index=str, columns = {'league': 'numberOfLeagues'})
teamELO = teamELO.merge(byLeague, on='teamid')

lastLeague = full_lol[['teamid', 'league']]
lastLeague = lastLeague.groupby('teamid').last()
lastLeague = lastLeague.reset_index()
lastLeague = lastLeague.rename(index=str, columns = {'league': 'lastLeagues'})
teamELO = teamELO.merge(lastLeague, on='teamid')

lastYear = full_lol[['teamid', 'year']]
lastYear = lastYear.groupby('teamid').last()
lastYear = lastYear.reset_index()
lastYear = lastYear.rename(index=str, columns = {'year': 'lastYearPlayed'})
teamELO = teamELO.merge(lastYear, on='teamid')
'''

teamELO = teamELO.merge(teamids, on='teamid')
#teamELO = teamELO.merge(leagueList, on='teamid')
teamELO = teamELO[['team', 'teamid', 'elo', 'Wins', 'Losses', 'league']] #, 'numberOfLeagues', 'lastLeagues', 'lastYearPlayed']]

teamELO.to_csv('teamelo.csv')

#teamELO = pd.read_csv('teamelo.csv')

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
playTeams['advProb'] = ''
playTeams['mainProb'] = ''
playTeams['seed'] = ''

for group in playTeams.group.unique():
    team1 = playTeams['team'][(playTeams['teamNum'] == 1) & (playTeams['group'] == group)].values[0]
    team2 = playTeams['team'][(playTeams['teamNum'] == 2) & (playTeams['group'] == group)].values[0]
    team3 = playTeams['team'][(playTeams['teamNum'] == 3) & (playTeams['group'] == group)].values[0]
    elo1 = playTeams['elo'][(playTeams['team'] == team1)].values[0]
    elo2 = playTeams['elo'][(playTeams['team'] == team2)].values[0]
    elo3 = playTeams['elo'][(playTeams['team'] == team3)].values[0]
    game1 = Probability(elo2, elo1)
    game2 = Probability(elo3, elo2)
    game3 = Probability(elo2, elo1)
    prob1 = (1-game1)*(1-game1)*(1-game3)*(1-game3)
    prob2 = game1*game1*(1-game2)*(1-game2)
    prob3 = game2*game2*game3*game3
    new_prob1 = 1- (prob1 / (prob1 + prob2 + prob3))
    new_prob2 = 1 - (prob2 / (prob1 + prob2 + prob3))
    new_prob3 = 1 - (prob3 / (prob1 + prob2 + prob3))
    playTeams['advProb'][(playTeams['team'] == team3)] = new_prob3
    playTeams['advProb'][(playTeams['team'] == team2)] = new_prob2
    playTeams['advProb'][(playTeams['team'] == team1)] = new_prob1
    teams=[team1, team2, team3]
    seed1 = playTeams['team'][(playTeams['advProb'] == max(playTeams['advProb'][(playTeams['team'] == team1)].values[0], playTeams['advProb'][(playTeams['team'] == team2)].values[0], 
                      playTeams['advProb'][(playTeams['team'] == team3)].values[0]))].values[0]
    teams.remove(seed1)
    team1 = teams[0]
    team2 = teams[1]
    seed2 = playTeams['team'][(playTeams['advProb'] == max(playTeams['advProb'][(playTeams['team'] == team1)].values[0], 
                      playTeams['advProb'][(playTeams['team'] == team2)].values[0]))].values[0]
    teams.remove(seed2)
    seed3 = teams[0]
    playTeams['seed'][(playTeams['team'] == seed1)] = 1
    playTeams['seed'][(playTeams['team'] == seed2)] = 2
    playTeams['seed'][(playTeams['team'] == seed3)] = 3
    
playTeams['mainProb'][(playTeams['seed'] == 3)] = 0
a1 = playTeams['team'][(playTeams['seed'] == 1) & (playTeams['group'] == 'PA')] .values[0] 
a2 = playTeams['team'][(playTeams['seed'] == 2) & (playTeams['group'] == 'PA')] .values[0] 
b1 = playTeams['team'][(playTeams['seed'] == 1) & (playTeams['group'] == 'PB')] .values[0]   
b2 = playTeams['team'][(playTeams['seed'] == 2) & (playTeams['group'] == 'PB')] .values[0] 
c1 = playTeams['team'][(playTeams['seed'] == 1) & (playTeams['group'] == 'PC')] .values[0] 
c2 = playTeams['team'][(playTeams['seed'] == 2) & (playTeams['group'] == 'PC')] .values[0] 
d1 = playTeams['team'][(playTeams['seed'] == 1) & (playTeams['group'] == 'PD')] .values[0] 
d2 = playTeams['team'][(playTeams['seed'] == 2) & (playTeams['group'] == 'PD')] .values[0]

playTeams['mainProb'] = 0

for team1 in playTeams['team'][(playTeams['group'] == 'PA')]:
    prob = 0
    for team2 in playTeams['team'][(playTeams['group'] == 'PD')]:
        prob = prob + bestofFive(team1, team2)
    prob = prob/3
    playTeams['mainProb'][(playTeams['team'] == team1)] = prob
for team1 in playTeams['team'][(playTeams['group'] == 'PD')]:
    prob = 0
    for team2 in playTeams['team'][(playTeams['group'] == 'PA')]:
        prob = prob +  bestofFive(team1, team2) 
    prob = prob/3
    playTeams['mainProb'][(playTeams['team'] == team1)] = prob        
for team1 in playTeams['team'][(playTeams['group'] == 'PB')]:
    prob = 0
    for team2 in playTeams['team'][(playTeams['group'] == 'PC')]:
        prob = prob + bestofFive(team1, team2) 
    prob = prob/3
    playTeams['mainProb'][(playTeams['team'] == team1)] = prob    
for team1 in playTeams['team'][(playTeams['group'] == 'PC')]:
    prob = 0
    for team2 in playTeams['team'][(playTeams['group'] == 'PB')]:
        prob = prob + bestofFive(team1, team2) 
    prob = prob/3
    playTeams['mainProb'][(playTeams['team'] == team1)] = prob
    
playTeams['mainProb'] = playTeams['mainProb'] * playTeams['advProb']

#sum(playTeams['mainProb'][(playTeams['group'] == 'PA') | (playTeams['group'] == 'PD')])

for team1 in playTeams['team'][(playTeams['group'] == 'PA') | (playTeams['group'] == 'PD')]:
    playTeams['mainProb'][(playTeams['team'] == team1)] = (2 * playTeams['mainProb'][(playTeams['team'] == team1)]) / sum(playTeams['mainProb'][(playTeams['group'] == 'PA') | (playTeams['group'] == 'PD')])
for team1 in playTeams['team'][(playTeams['group'] == 'PB') | (playTeams['group'] == 'PC')]:
    playTeams['mainProb'][(playTeams['team'] == team1)] = (2 * playTeams['mainProb'][(playTeams['team'] == team1)]) / sum(playTeams['mainProb'][(playTeams['group'] == 'PB') | (playTeams['group'] == 'PC')])    

    
playTeams['mainProb'][(playTeams['team'] == a1)] = bestofFive(a1, d2)
playTeams['mainProb'][(playTeams['team'] == d2)] = bestofFive(d2, a1)
playTeams['mainProb'][(playTeams['team'] == d1)] = bestofFive(d1, a2)
playTeams['mainProb'][(playTeams['team'] == a2)] = bestofFive(a2, d1)
playTeams['mainProb'][(playTeams['team'] == b1)] = bestofFive(b1, c2)
playTeams['mainProb'][(playTeams['team'] == c2)] = bestofFive(c2, b1)
playTeams['mainProb'][(playTeams['team'] == c1)] = bestofFive(c1, b2)
playTeams['mainProb'][(playTeams['team'] == b2)] = bestofFive(b2, c1) 

###### cOMPLETE PROOF OF ADDING UP TOURNEY
    
    
'''    
    
    #highTeam = playTeams['team'][(playTeams['winProb'] == max(playTeams['winProb'][(playTeams['team'] == team1)].values[0], playTeams['winProb'][(playTeams['team'] == team2)].values[0], 
    #                  playTeams['winProb'][(playTeams['team'] == team3)].values[0]))].values[0]
    playTeams['winProb'][(playTeams['team'] == highTeam)] = 1
    teams=[team1, team2, team3]
    teams.remove(highTeam)
    team1 = teams[0]
    team2 = teams[1]
    prob1 = playTeams['winProb'][(playTeams['team'] == team1)].values[0]
    prob2 = playTeams['winProb'][(playTeams['team'] == team2)].values[0]
    playTeams['winProb'][(playTeams['team'] == team1)] = prob1 / (prob1 + prob2)
    playTeams['winProb'][(playTeams['team'] == team2)] = prob2 / (prob1 + prob2)
    
    #if team1 == highTeam:
    #    pass
    #else:
Probability(a2, a1)**3 *(1 + (1-Probability(a2,a1)) + (1-Probability(a2,a1))**2)
        
prob***3 + 3*prob**3*(1-prob) + 6*prob**3*1-prob)**2
    
#playTeams['mainProb'][(playTeams['team'] == d1)] = bestofFive(d1, a2)
#playTeams['mainProb'][(playTeams['team'] == a2)] = bestofFive(a2, d1)
#playTeams['mainProb'][(playTeams['team'] == b1)] = bestofFive(b1, c2)
#playTeams['mainProb'][(playTeams['team'] == c2)] = bestofFive(c2, b1)
#playTeams['mainProb'][(playTeams['team'] == c1)] = bestofFive(c1, b2)
#playTeams['mainProb'][(playTeams['team'] == b2)] = bestofFive(b2, c1)    
    

def ProbByTeam(team1, team2):
    elo1 = wcTeam['elo'][(wcTeam['team'] == team1)].values[0]
    elo2 = wcTeam['elo'][(wcTeam['team'] == team2)].values[0]
    prob = Probability(elo2, elo1)
    prob = prob**2 + 2*prob**2*(1-prob)
    return prob

ProbByTeam('Flash Wolves', 'Dire Wolves')

new_prob1 + new_prob2 + new_prob3
'''