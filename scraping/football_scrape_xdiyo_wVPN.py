import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options

from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import os
import time

def get_league(matches,standings,stats,shots):
    '''Gets the matches information for a league'''
    data={}
    index=0
    for i in range(len(matches)):
        for j in range(len(matches[i]['events'])):
            if matches[i]['events'][j]['status']['description'] == 'Ended':
                #print(matches[i]['events'][j]['roundInfo']['round'],matches[i]['events'][j]['homeTeam']['slug'],matches[i]['events'][j]['awayTeam']['slug'])
                match_id = matches[i]['events'][j]['id']
                temp_dic={
                    'custom_id': matches[i]['events'][j]['customId'],
                    'match_id': match_id,
                    'season_id': matches[i]['events'][j]['season']['id'],
                    'season_year': matches[i]['events'][j]['season']['year'],
                    'round': matches[i]['events'][j]['roundInfo']['round'],
                    'home_id': matches[i]['events'][j]['homeTeam']['id'],
                    'home_standing': standings[match_id]['home'],
                    'home_team': matches[i]['events'][j]['homeTeam']['slug'],
                    'home_score': matches[i]['events'][j]['homeScore'].get('display',0),
                    'home_45_min': matches[i]['events'][j]['homeScore'].get('period1',0),
                    'home_90_min': matches[i]['events'][j]['homeScore'].get('display',0)-\
                                    matches[i]['events'][j]['homeScore'].get('period1',0),
                    'away_id': matches[i]['events'][j]['awayTeam']['id'],
                    'away_standing': standings[match_id]['away'],
                    'away_team': matches[i]['events'][j]['awayTeam']['slug'],
                    'away_score': matches[i]['events'][j]['awayScore'].get('display',0),
                    'away_45_min': matches[i]['events'][j]['awayScore'].get('period1',0),
                    'away_90_min': matches[i]['events'][j]['awayScore'].get('display',0)-\
                                    matches[i]['events'][j]['awayScore'].get('period1',0),
                }
                temp_dic.update(stats[match_id])
                temp_dic.update(shots[match_id])
                data[index]=temp_dic
                index+=1
        league = {i: v for i, (_, v) in enumerate(sorted(data.items(), key=lambda item: item[1]['round']))}

    return league

def get_recent_league(matches,standings,stats,shots):
    '''Gets the matches information for a league'''
    data={}
    index=0
    for i in range(len(matches)):
        for j in range(len(matches[i]['events'])):
            #print(matches[i]['events'][j]['roundInfo']['round'],matches[i]['events'][j]['homeTeam']['slug'],matches[i]['events'][j]['awayTeam']['slug'])
            match_id = matches[i]['events'][j]['id']
            temp_dic={
                'custom_id': matches[i]['events'][j]['customId'],
                'match_id': match_id,
                'season_id': matches[i]['events'][j]['season']['id'],
                'season_year': matches[i]['events'][j]['season']['year'],
                'round': matches[i]['events'][j]['roundInfo']['round'],
                'home_id': matches[i]['events'][j]['homeTeam']['id'],
                'home_standing': standings[match_id]['home'],
                'home_team': matches[i]['events'][j]['homeTeam']['slug'],
                'home_score': matches[i]['events'][j]['homeScore'].get('display',0),
                'home_45_min': matches[i]['events'][j]['homeScore'].get('period1',0),
                'home_90_min': matches[i]['events'][j]['homeScore'].get('display',0)-\
                                matches[i]['events'][j]['homeScore'].get('period1',0),
                'away_id': matches[i]['events'][j]['awayTeam']['id'],
                'away_standing': standings[match_id]['away'],
                'away_team': matches[i]['events'][j]['awayTeam']['slug'],
                'away_score': matches[i]['events'][j]['awayScore'].get('display',0),
                'away_45_min': matches[i]['events'][j]['awayScore'].get('period1',0),
                'away_90_min': matches[i]['events'][j]['awayScore'].get('display',0)-\
                                matches[i]['events'][j]['awayScore'].get('period1',0),
            }
            temp_dic.update(stats[match_id])
            temp_dic.update(shots[match_id])
            data[index]=temp_dic
            index+=1
        league = {i: v for i, (_, v) in enumerate(sorted(data.items(), key=lambda item: item[1]['round']))}

    return league

def remove_duplicates(data):
    unique=[]
    for key, value in enumerate(data):
        if value not in unique and 'error' not in value:
            unique.append(value)
    
    return unique

def get_standings_team(team_data):
    '''Extracts the name of the team for given standings data'''
    
    team_names=[team_data['graphData'][0]['events'][0]['homeTeam'],
                team_data['graphData'][0]['events'][0]['awayTeam'],
                team_data['graphData'][1]['events'][0]['homeTeam'],
                team_data['graphData'][1]['events'][0]['awayTeam'],
                team_data['graphData'][2]['events'][0]['homeTeam'],
                team_data['graphData'][2]['events'][0]['awayTeam']]
    extracted_name=max(team_names, key=team_names.count)
    
    return extracted_name['slug']

def get_weekly_standings(raw_weekly_standings):
    weekly_standings = {}
    for i in range(len(raw_weekly_standings)):
        weekly_standings[raw_weekly_standings[i]['week']] = \
            raw_weekly_standings[i]['position']
    for i in range (11):
        a=len(raw_weekly_standings)+i+1
        weekly_standings[a] = weekly_standings[a-1]
        
    return weekly_standings

def get_standings(raw_standings):
    '''Gets the standings information for a league'''
    
    #Remove duplicates from raw data
    raw_standings = remove_duplicates(raw_standings)
    
    
    #get the team names
    team_names=[]
    for i in range(len(raw_standings)):
        team_names.append(get_standings_team(raw_standings[i]))
    
    #get the team standings during the season
    standings={}
    for i in range(len(raw_standings)):
        standings[team_names[i]]=get_weekly_standings(raw_standings[i]['graphData'])
    
    return standings

def get_formation(formation_string):
    formation_elements = tuple(map(int, formation_string.split('-')))
    
    # Pad the tuple to ensure it has exactly 4 elements
    padded_formation = formation_elements + (0,) * (4 - len(formation_elements))
    
    return tuple(padded_formation)

def get_heatmaps(league, path, profile_path, waiting_time):
    """
    Scrapes heatmap data for each match in the league.
    Retrieves both player (playerPoints) and goalkeeper (goalkeeperPoints) heatmaps
    for the home and away teams, incorporating error handling.

    Parameters:
    league (dict): League dictionary with match info; each match must contain 'match_id', 'home_id', and 'away_id'.
    path (str): Path to the chromedriver executable.
    waiting_time (int): Max time to wait for a page load.

    Returns:
    dict: A dictionary where keys are match IDs and values are dictionaries with:
            "home": heatmap data or error message,
            "away": heatmap data or error message.
    """
    options = webdriver.ChromeOptions()
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL', 'browser': 'ALL'})
    service = ChromeService(executable_path=path)
    options.add_extension("C:/Users/luisi/Documents/Programming/Chromedriver/windscribe.crx")
    options.add_argument(f"{profile_path}")
    driver = webdriver.Chrome(service=service, options=options)

    
    heatmaps = {}
    
    for i in range(len(league)):
        match_id = league[i]['match_id']
        home_team_id = league[i]['home_id']
        away_team_id = league[i]['away_id']
        
        # Initialize containers for the home and away heatmaps
        home_heatmap = {}
        away_heatmap = {}
        
        # Attempt to fetch the home team heatmap
        try:
            driver.set_page_load_timeout(waiting_time)
            driver.get(f'https://www.sofascore.com/api/v1/event/{match_id}/heatmap/{home_team_id}')
            home_json = driver.find_element("tag name", "body").text
            home_json_data = json.loads(home_json)
            # Check if the response contains an error object
            if "error" in home_json_data:
                home_heatmap = {"error": home_json_data["error"].get("message", "Unknown error")}
            else:
                home_heatmap = home_json_data
        except Exception as e:
            home_heatmap = {"error": str(e)}
        
        # Attempt to fetch the away team heatmap
        try:
            driver.set_page_load_timeout(waiting_time)
            driver.get(f'https://www.sofascore.com/api/v1/event/{match_id}/heatmap/{away_team_id}')
            away_json = driver.find_element("tag name", "body").text
            away_json_data = json.loads(away_json)
            if "error" in away_json_data:
                away_heatmap = {"error": away_json_data["error"].get("message", "Unknown error")}
            else:
                away_heatmap = away_json_data
        except Exception as e:
            away_heatmap = {"error": str(e)}
        
        heatmaps[match_id] = {"home": home_heatmap, "away": away_heatmap}
    
    driver.quit()
    return heatmaps


def get_recent_heatmaps(league, path, profile_path, waiting_time, num_previous_matches):
    """
    Scrapes heatmap data for the most recent matches in a league with error handling.
    
    For the most recent matches (determined by num_previous_matches), this function attempts to
    fetch heatmap data. If the API call fails or returns an error, an error message is stored.
    For older matches, empty dictionaries are stored.

    Parameters:
    league (dict): League dictionary with match info.
    path (str): Path to the chromedriver executable.
    waiting_time (int): Time to wait for a page load.
    num_previous_matches (int): Number of most recent matches to fetch heatmap data for.

    Returns:
    dict: A dictionary where keys are match IDs and values are dictionaries with the heatmap data,
            or error details if applicable.
    """
    options = webdriver.ChromeOptions()
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL', 'browser': 'ALL'})
    service = ChromeService(executable_path=path)
    options.add_extension("C:/Users/luisi/Documents/Programming/Chromedriver/windscribe.crx")
    options.add_argument(f"{profile_path}")
    driver = webdriver.Chrome(service=service, options=options)

    
    heatmaps = {}
    
    for i in range(len(league)):
        match_id = league[i]['match_id']
        home_team_id = league[i]['home_id']
        away_team_id = league[i]['away_id']
        
        # For recent matches, fetch fresh data; for older matches, leave as empty dicts
        if i < num_previous_matches:
            home_heatmap = {}
            away_heatmap = {}
            
            try:
                driver.set_page_load_timeout(waiting_time)
                driver.get(f'https://www.sofascore.com/api/v1/event/{match_id}/heatmap/{home_team_id}')
                home_json = driver.find_element("tag name", "body").text
                home_json_data = json.loads(home_json)
                if "error" in home_json_data:
                    home_heatmap = {"error": home_json_data["error"].get("message", "Unknown error")}
                else:
                    home_heatmap = home_json_data
            except Exception as e:
                home_heatmap = {"error": str(e)}
            
            try:
                driver.set_page_load_timeout(waiting_time)
                driver.get(f'https://www.sofascore.com/api/v1/event/{match_id}/heatmap/{away_team_id}')
                away_json = driver.find_element("tag name", "body").text
                away_json_data = json.loads(away_json)
                if "error" in away_json_data:
                    away_heatmap = {"error": away_json_data["error"].get("message", "Unknown error")}
                else:
                    away_heatmap = away_json_data
            except Exception as e:
                away_heatmap = {"error": str(e)}
        else:
            home_heatmap, away_heatmap = {}, {}
        
        heatmaps[match_id] = {"home": home_heatmap, "away": away_heatmap}
    
    driver.quit()
    return heatmaps

def get_position_tuple(position_string):
    if position_string == 'G':
        return (1,0,0,0)
    elif position_string == 'D':
        return (0,1,0,0)
    elif position_string == 'M':
        return (0,0,1,0)
    elif position_string == 'F':
        return (0,0,0,1)
    else:
        return (0,0,0,0)

def get_substitution_tuple(substitution_bool):
    if substitution_bool:
        return (0,1)
    else:
        return (1,0)
    
def get_players(text_data):
    
    one_roster={}
    for player in text_data['players']:
        if 'statistics' in player and 'position' in player['player']:
            one_roster[player['player']['slug']] = {
                'position': get_position_tuple(player['position']),
                'substitution': get_substitution_tuple(player['substitute']),
                'statistics': player['statistics']
            }
        if 'statistics' not in player and 'position' in player['player']:
            one_roster[player['player']['slug']] = {
                'position': get_position_tuple(player['position']),
                'substitution': get_substitution_tuple(player['substitute']),
                'statistics': {}
            }
    
    return one_roster

def get_recent_players(text_data):
    
    one_roster={}
    for player in text_data['players']:
        if 'statistics' in player and 'position' in player['player']:
            one_roster[player['player']['slug']] = {
                'position': get_position_tuple(player['position']),
                'substitution': get_substitution_tuple(player['substitute']),
                'statistics': {}
            }
        if 'statistics' not in player and 'position' in player['player']:
            one_roster[player['player']['slug']] = {
                'position': get_position_tuple(player['position']),
                'substitution': get_substitution_tuple(player['substitute']),
                'statistics': {}
            }
    
    return one_roster

def get_lineups(league,path,profile_path,waiting_time):
    options = webdriver.ChromeOptions()
    options.set_capability(
        'goog:loggingPrefs', {'performance': 'ALL', 'browser': 'ALL'}
    )
    service = ChromeService(executable_path=path)
    options.add_extension("C:/Users/luisi/Documents/Programming/Chromedriver/windscribe.crx")
    options.add_argument(f"{profile_path}")
    driver = webdriver.Chrome(service=service, options=options)

    #C:/Users/luisi/Documents/Programming/Chromedriver/chromedriver.exe
    #C:/Users/luisi/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe    
    
    lineup={}
    
    for i in range(len(league)):
        id = league[i]['match_id']
        driver.set_page_load_timeout(waiting_time)
        try:
            driver.get(f'https://www.sofascore.com/api/v1/event/{id}/lineups')
        except:
            pass
        
        # Get the page source, which should contain the JSON text
        json_text = driver.find_element("tag name", "body").text

        # Parse the JSON text
        text_data = json.loads(json_text)
        if text_data!={"error":{"code":404,"message":"Not Found"}}:
            if 'formation' in text_data['home'].keys() and 'formation' in text_data['away'].keys():
                home_formation = get_formation(text_data['home']['formation'])
                away_formation = get_formation(text_data['away']['formation'])
                home_roster = get_players(text_data['home'])
                away_roster = get_players(text_data['away'])
            else:
                home_formation = (0,0,0,0)
                away_formation = (0,0,0,0)
                home_roster = {}
                away_roster = {}
        
        
        
        
        home_lineup = {'formation': home_formation, 'roster': home_roster}
        away_lineup = {'formation': away_formation, 'roster': away_roster}
        
        lineup[id]={
            'home': home_lineup,
            'away': away_lineup
        }
        
    driver.quit()
    
    return lineup

def get_recent_lineups(league,path,profile_path,waiting_time,num_previous_matches):
    options = webdriver.ChromeOptions()
    options.set_capability(
        'goog:loggingPrefs', {'performance': 'ALL', 'browser': 'ALL'}
    )
    service = ChromeService(executable_path=path)
    options.add_extension("C:/Users/luisi/Documents/Programming/Chromedriver/windscribe.crx")
    options.add_argument(f"{profile_path}")
    driver = webdriver.Chrome(service=service, options=options)

    #C:/Users/luisi/Documents/Programming/Chromedriver/chromedriver.exe
    #C:/Users/luisi/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe    
    
    lineup={}
    names = {}
    
    for i in range(len(league)):
        id = league[i]['match_id']
        home_team_slug = league[i]['home_team']
        away_team_slug = league[i]['away_team']
        if i < num_previous_matches:
            driver.set_page_load_timeout(waiting_time)
            try:
                driver.get(f'https://www.sofascore.com/api/v1/event/{id}/lineups')
            except:
                pass
            
            # Get the page source, which should contain the JSON text
            json_text = driver.find_element("tag name", "body").text

            # Parse the JSON text
            text_data = json.loads(json_text)
            if text_data!={"error":{"code":404,"message":"Not Found"}}:
                if 'formation' in text_data['home'].keys() and 'formation' in text_data['away'].keys():
                    home_formation = get_formation(text_data['home']['formation'])
                    away_formation = get_formation(text_data['away']['formation'])
                    home_roster = get_players(text_data['home'])
                    away_roster = get_players(text_data['away'])
                else:
                    home_formation = (0,0,0,0)
                    away_formation = (0,0,0,0)
                    home_roster = {}
                    away_roster = {}
                names[home_team_slug] = {player:{'position':home_roster[player]['position'],
                                                'substitution':home_roster[player]['substitution'],
                                                'statistics':{}} for player in home_roster.keys()}
                names[away_team_slug] = {player:{'position':away_roster[player]['position'],
                                                'substitution':away_roster[player]['substitution'],
                                                'statistics':{}} for player in away_roster.keys()}
        else:
            home_formation = (0,0,0,0)
            away_formation = (0,0,0,0)
            home_roster = {player:names[home_team_slug][player] for player in names[home_team_slug]} 
            away_roster = {player:names[away_team_slug][player] for player in names[away_team_slug]}
        
        home_lineup = {'formation': home_formation, 'roster': home_roster}
        away_lineup = {'formation': away_formation, 'roster': away_roster}

        lineup[id]={
            'home': home_lineup,
            'away': away_lineup
        }
        
    driver.quit()
    
    return lineup

    
def force_get_lineups(lineups_df,path,profile_path,waiting_time):

    options = webdriver.ChromeOptions()
    options.set_capability(
        'goog:loggingPrefs', {'performance': 'ALL', 'browser': 'ALL'}
    )
    service = ChromeService(executable_path=path)
    options.add_extension("C:/Users/luisi/Documents/Programming/Chromedriver/windscribe.crx")
    options.add_argument(f"{profile_path}")
    driver = webdriver.Chrome(service=service, options=options)

    #C:/Users/luisi/Documents/Programming/Chromedriver/chromedriver.exe
    #C:/Users/luisi/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe   
    
    empty_ids = [i for i in lineups_df.index if lineups_df.loc[i]['home']['roster']=={} ]
    
    
    for empty_id in empty_ids:
        rosters={}
        driver.set_page_load_timeout(waiting_time)
        try:
            driver.get(f'https://www.sofascore.com/api/v1/event/{empty_id}/lineups')
        except:
            pass

        # Get the page source, which should contain the JSON text
        json_text = driver.find_element("tag name", "body").text

        # Parse the JSON text
        text_data = json.loads(json_text)

        home_roster = get_players(text_data['home'])
        away_roster = get_players(text_data['away'])

        rosters={
            'home': home_roster,
            'away': away_roster
        }
        
        lineups_df.loc[empty_id]['home']['roster'] = rosters['home']
        lineups_df.loc[empty_id]['away']['roster'] = rosters['away']
    # Close the driver
    driver.quit()
    
    return lineups_df

def scrape_data(league_id,season_id,num_rounds,num_teams,path, profile_path, waiting_time,with_xG,with_comments):
    options = webdriver.ChromeOptions()
    options.set_capability(
        'goog:loggingPrefs', {'performance': 'ALL', 'browser': 'ALL'}
    )
    service = ChromeService(executable_path=path)
    options.add_extension("C:/Users/luisi/Documents/Programming/Chromedriver/windscribe.crx")
    options.add_argument(f"{profile_path}")
    driver = webdriver.Chrome(service=service, options=options)

    #C:/Users/luisi/Documents/Programming/Chromedriver/chromedriver.exe
    #C:/Users/luisi/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe
    periods = ['total','45_min','90_min']
    groups = ['match_overview','shots','attack','passes','duels','defending','goalkeeping']
    matches=[]
    standings={}
    stats={}
    shots={}
    comments={}
    for i in range(1,num_rounds+1):
        driver.set_page_load_timeout(waiting_time)
        try:
            driver.get(f'https://www.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/events/round/{i}')
        except:
            pass
        
        # Get the page source, which should contain the JSON text
        json_text = driver.find_element("tag name", "body").text

        # Parse the JSON text
        text_data = json.loads(json_text)
        matches.append(text_data)
        
        match_ids = [text_data['events'][i]['id'] for i in range(len(text_data['events']))]
        
        for id in match_ids:
            driver.set_page_load_timeout(waiting_time)
            try:
                driver.get(f'https://www.sofascore.com/api/v1/event/{id}/pregame-form')
            except:
                pass
            
            json_match = driver.find_element("tag name", "body").text
            match_text = json.loads(json_match)
            
            if match_text=={"error":{"code":404,"message":"Not Found"}}:
                match_standings = {'home': num_teams/2,
                            'away': num_teams/2}
                standings[id]=match_standings
            else:  
                match_standings = {'home': match_text['homeTeam']['position'],
                                'away': match_text['awayTeam']['position']}
                standings[id]=match_standings

            #Get match statistics
            driver.set_page_load_timeout(waiting_time)
            try:
                driver.get(f'https://www.sofascore.com/api/v1/event/{id}/statistics')
            except:
                pass
            
            json_match = driver.find_element("tag name", "body").text
            stats_data = json.loads(json_match)


            match_stats={}
            if stats_data!={"error":{"code":404,"message":"Not Found"}}:
                for i, period in enumerate(periods):
                    if i>=len(stats_data['statistics']):
                        break
                    for j in range(len(stats_data['statistics'][i]['groups'])):
                        for k in range(len(stats_data['statistics'][i]['groups'][j]['statisticsItems'])):
                            stat = stats_data['statistics'][i]['groups'][j]['statisticsItems'][k]['key'].lower().replace(' ', '_')
                            for team in ['home','away']:
                                if 'homeTotal' in stats_data['statistics'][i]['groups'][j]['statisticsItems'][k].keys():
                                    match_stats[f'{team}_{period}_{stat}_value'] = stats_data['statistics'][i]['groups'][j]['statisticsItems'][k][f'{team}Value']
                                    match_stats[f'{team}_{period}_{stat}_all'] = stats_data['statistics'][i]['groups'][j]['statisticsItems'][k][f'{team}Total']
                                else:
                                    match_stats[f'{team}_{period}_{stat}'] = stats_data['statistics'][i]['groups'][j]['statisticsItems'][k][f'{team}Value']

            stats[id]=match_stats
            
            if with_comments==True:
                driver.set_page_load_timeout(waiting_time)
                try:
                    driver.get(f'https://www.sofascore.com/api/v1/event/{id}/comments')
                except:
                    pass
                
                json_comments = driver.find_element("tag name", "body").text
                comments_data = json.loads(json_comments)
                
                driver.set_page_load_timeout(waiting_time)
                try:
                    driver.get(f'https://www.sofascore.com/api/v1/event/{id}')
                except:
                    pass
                
                json_names = driver.find_element("tag name", "body").text
                names_data = json.loads(json_names)
                
                home_name = names_data['event']['homeTeam']['slug']
                away_name = names_data['event']['awayTeam']['slug']
                round_num = names_data['event']['roundInfo']['round']
                
                match_comments = {}
                if comments_data!={"error":{"code":404,"message":"Not Found"}}:
                    comments_text = comments_data['comments']
                    comments_text.reverse()
                    for i in range(len(comments_text)):
                        if 'isHome' in comments_text[i].keys():
                            team = 'home team' if comments_text[i]['isHome'] else 'away team'
                        else:
                            team = 'general'
                        match_comments[i] = {'minute': comments_text[i]['time'],
                                        'event': comments_text[i]['text'],
                                        'team': team}

                comments[id]={'comments': match_comments,
                            'home_team': home_name,
                            'away_team': away_name,
                            'round': round_num}
            
            #Get match shots
            if with_xG==True:
                driver.set_page_load_timeout(waiting_time)
                try:
                    driver.get(f'https://www.sofascore.com/api/v1/event/{id}/shotmap')
                except:
                    pass
                
                json_match = driver.find_element("tag name", "body").text
                shots_data = json.loads(json_match)
                
                home_shots, away_shots = [], []
                if shots_data!={"error":{"code":404,"message":"Not Found"}} and 'shotmap' in shots_data.keys():
                    for i in range(len(shots_data['shotmap'])):
                        if shots_data['shotmap'][i]['isHome']:
                            if 'xg' in shots_data['shotmap'][i].keys():
                                home_shots.append({'xg': shots_data['shotmap'][i].get('xg',0.0),
                                                'xgot':shots_data['shotmap'][i].get('xgot',0.0),
                                                'shotType': shots_data['shotmap'][i].get('shotType','None'),
                                                'situation': shots_data['shotmap'][i].get('situation','None'),
                                                'time': shots_data['shotmap'][i].get('time',0)})
                            else:
                                home_shots.append({'xg': 1.0,# Account for own goals
                                                'xgot':1.0,
                                                'shotType': shots_data['shotmap'][i].get('shotType','None'),
                                                'situation': shots_data['shotmap'][i].get('situation','None'),
                                                'time': shots_data['shotmap'][i].get('time',0)})
                        else:
                            if 'xg' in shots_data['shotmap'][i].keys():
                                away_shots.append({'xg': shots_data['shotmap'][i].get('xg',0.0),
                                                'xgot':shots_data['shotmap'][i].get('xgot',0.0),
                                                'shotType': shots_data['shotmap'][i].get('shotType','None'),
                                                'situation': shots_data['shotmap'][i].get('situation','None'),
                                                'time': shots_data['shotmap'][i].get('time',0)})
                            else:
                                away_shots.append({'xg': 1.0,
                                                'xgot':1.0,	
                                                'shotType': shots_data['shotmap'][i].get('shotType','None'),
                                                'situation': shots_data['shotmap'][i].get('situation','None'),
                                                'time': shots_data['shotmap'][i].get('time',0)})
                            
                shots[id]={'home_shots': home_shots, 'away_shots': away_shots}
            else:
                shots[id]={'home_shots': [], 'away_shots': []}
            
    driver.quit()
        
    return matches,standings,stats,shots,comments

def scrape_recent_data(league_id,season_id,round,num_teams,path,profile_path,waiting_time,with_xG):
    options = webdriver.ChromeOptions()
    options.set_capability(
        'goog:loggingPrefs', {'performance': 'ALL', 'browser': 'ALL'}
    )
    service = ChromeService(executable_path=path)
    options.add_extension("C:/Users/luisi/Documents/Programming/Chromedriver/windscribe.crx")
    options.add_argument(f"{profile_path}")
    driver = webdriver.Chrome(service=service, options=options)



    #C:/Users/luisi/Documents/Programming/Chromedriver/chromedriver.exe
    #C:/Users/luisi/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe
    periods = ['total','45_min','90_min']
    groups = ['match_overview','shots','attack','passes','duels','defending','goalkeeping']
    matches=[]
    standings={}
    stats={}
    shots={}
    num_previous_matches = 0
    for i in range(round-1,round+1):
        driver.set_page_load_timeout(waiting_time)
        try:
            driver.get(f'https://www.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/events/round/{i}')
        except:
            pass
        
        # Get the page source, which should contain the JSON text
        json_text = driver.find_element("tag name", "body").text

        # Parse the JSON text
        text_data = json.loads(json_text)
        matches.append(text_data)
        
        match_ids = [text_data['events'][i]['id'] for i in range(len(text_data['events']))]
        if i==round-1:
            num_previous_matches = len(match_ids)
        
        for id in match_ids:
            driver.set_page_load_timeout(waiting_time)
            try:
                driver.get(f'https://www.sofascore.com/api/v1/event/{id}/pregame-form')
            except:
                pass
            
            json_match = driver.find_element("tag name", "body").text
            match_text = json.loads(json_match)
            
            if match_text=={"error":{"code":404,"message":"Not Found"}}:
                match_standings = {'home': num_teams/2,
                            'away': num_teams/2}
                standings[id]=match_standings
            else:  
                match_standings = {'home': match_text['homeTeam']['position'],
                                'away': match_text['awayTeam']['position']}
                standings[id]=match_standings

            #Get match statistics
            match_stats={}
            #if i==round-1:
            driver.set_page_load_timeout(waiting_time)
            try:
                driver.get(f'https://www.sofascore.com/api/v1/event/{id}/statistics')
            except:
                pass
            
            json_match = driver.find_element("tag name", "body").text
            stats_data = json.loads(json_match)

            if stats_data!={"error":{"code":404,"message":"Not Found"}}:
                for i, period in enumerate(periods):
                    if i>=len(stats_data['statistics']):
                        break
                    for j in range(len(stats_data['statistics'][i]['groups'])):
                        for k in range(len(stats_data['statistics'][i]['groups'][j]['statisticsItems'])):
                            stat = stats_data['statistics'][i]['groups'][j]['statisticsItems'][k]['key'].lower().replace(' ', '_')
                            for team in ['home','away']:
                                if 'homeTotal' in stats_data['statistics'][i]['groups'][j]['statisticsItems'][k].keys():
                                    match_stats[f'{team}_{period}_{stat}_value'] = stats_data['statistics'][i]['groups'][j]['statisticsItems'][k][f'{team}Value']
                                    match_stats[f'{team}_{period}_{stat}_all'] = stats_data['statistics'][i]['groups'][j]['statisticsItems'][k][f'{team}Total']
                                else:
                                    match_stats[f'{team}_{period}_{stat}'] = stats_data['statistics'][i]['groups'][j]['statisticsItems'][k][f'{team}Value']

            stats[id]=match_stats
            
            #Get match shots
            if with_xG==True:
                driver.set_page_load_timeout(waiting_time)
                try:
                    driver.get(f'https://www.sofascore.com/api/v1/event/{id}/shotmap')
                except:
                    pass
                
                json_match = driver.find_element("tag name", "body").text
                shots_data = json.loads(json_match)
                
                home_shots, away_shots = [], []
                if shots_data!={"error":{"code":404,"message":"Not Found"}}:
                    for i in range(len(shots_data['shotmap'])):
                        if shots_data['shotmap'][i]['isHome']:
                            if 'xg' in shots_data['shotmap'][i].keys():
                                home_shots.append({'xg': shots_data['shotmap'][i].get('xg',0.0),
                                                'xgot': shots_data['shotmap'][i].get('xgot',0.0),
                                                'shotType': shots_data['shotmap'][i]['shotType'],
                                                'situation': shots_data['shotmap'][i]['situation'],
                                                'time': shots_data['shotmap'][i]['time']})
                            else:
                                home_shots.append({'xg': 1.0,# Account for own goals
                                                'xgot': 1.0,
                                                'shotType': shots_data['shotmap'][i].get('goalType','None'),
                                                'situation': shots_data['shotmap'][i].get('situation','None'),
                                                'time': shots_data['shotmap'][i].get('time',0)})
                        else:
                            if 'xg' in shots_data['shotmap'][i].keys():
                                away_shots.append({'xg': shots_data['shotmap'][i].get('xg',0.0),
                                                'xgot': shots_data['shotmap'][i].get('xgot',0.0),
                                                'shotType': shots_data['shotmap'][i]['shotType'],
                                                'situation': shots_data['shotmap'][i]['situation'],
                                                'time': shots_data['shotmap'][i]['time']})
                            else:
                                away_shots.append({'xg': 1.0,
                                                'xgot': 1.0,
                                                'shotType': shots_data['shotmap'][i].get('goalType','None'),
                                                'situation': shots_data['shotmap'][i].get('situation','None'),
                                                'time': shots_data['shotmap'][i].get('time',0)})
                            
                shots[id]={'home_shots': home_shots, 'away_shots': away_shots}
            else:
                shots[id]={'home_shots': [], 'away_shots': []}
            
    driver.quit()
    return matches,standings,stats,shots,num_previous_matches

def scrape_league_data(league_id, season_id, num_rounds, num_teams,
                    league_name, season_start, season_end, path, profile_path, waiting_time=10,
                    with_lineups=False, with_xG=False, with_comments=False, with_heatmaps=False):
    
    matches, standings, stats, shots, comments = scrape_data(league_id, season_id, num_rounds,
                                                            num_teams, path, profile_path, waiting_time, with_xG, with_comments)
    league = get_league(matches, standings, stats, shots)
    
    # Add heatmap data if required
    if with_heatmaps:
        heatmaps = get_heatmaps(league, path, profile_path, waiting_time)
        for key, match_data in league.items():
            match_id = match_data['match_id']
            match_data['home_heatmap'] = heatmaps.get(match_id, {}).get('home', {"error": "Heatmap not fetched"})
            match_data['away_heatmap'] = heatmaps.get(match_id, {}).get('away', {"error": "Heatmap not fetched"})
    
    # Convert the league dictionary to a DataFrame
    df = pd.DataFrame.from_dict(league, orient='index')
    
    # Optionally, if you want to store the heatmap dictionaries in the CSV as strings,
    # you might convert them using json.dumps:
    if with_heatmaps:
        df['home_heatmap'] = df['home_heatmap'].apply(json.dumps)
        df['away_heatmap'] = df['away_heatmap'].apply(json.dumps)
    
    # Export the DataFrame to a CSV file
    df.to_csv(f'{league_name}_{season_start}_{season_end}.csv', index_label='index')
    
    if with_comments:
        os.makedirs("comments", exist_ok=True)
        for i in comments.keys():
            filepath = f"comments/round_{comments[i]['round']}_{comments[i]['home_team']}_{comments[i]['away_team']}.txt"
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(f"Home team: {comments[i]['home_team']}. Away team: {comments[i]['away_team']}. Round: {comments[i]['round']}\n\n")
                for key, value in comments[i]['comments'].items():
                    file.write(f"Event {key}:\n")
                    for sub_key, sub_value in value.items():
                        file.write(f"  {sub_key}: {sub_value}\n")
                    file.write("\n")
    
    if with_lineups:
        lineups = get_lineups(league, path, profile_path, waiting_time)
        lineup_df = pd.DataFrame.from_dict(lineups, orient='index')
        lineup_df = force_get_lineups(lineup_df, path, profile_path, waiting_time)
        lineup_df.to_csv(f'{league_name}_{season_start}_{season_end}_lineups.csv', index_label='match_id')


def scrape_league_recent(league_id, season_id, num_rounds, num_teams,
                        league_name, season_start, season_end, path, profile_path, waiting_time=10,
                        with_lineups=False, with_xG=False, with_heatmaps=False):
    
    matches, standings, stats, shots, num_previous_matches = scrape_recent_data(
        league_id, season_id, num_rounds, num_teams, path, profile_path, waiting_time, with_xG
    )
    league = get_recent_league(matches, standings, stats, shots)
    
    if with_heatmaps:
        heatmaps = get_heatmaps(league, path, profile_path, waiting_time)
        for key, match_data in league.items():
            match_id = match_data['match_id']
            match_data['home_heatmap'] = heatmaps.get(match_id, {}).get('home', {"error": "Heatmap not fetched"})
            match_data['away_heatmap'] = heatmaps.get(match_id, {}).get('away', {"error": "Heatmap not fetched"})
    
    df = pd.DataFrame.from_dict(league, orient='index')
    
    if with_heatmaps:
        df['home_heatmap'] = df['home_heatmap'].apply(json.dumps)
        df['away_heatmap'] = df['away_heatmap'].apply(json.dumps)
    
    df.to_csv(f'{league_name}_{season_start}_{season_end}_recent.csv', index_label='index')
    
    if with_lineups:
        lineups = get_recent_lineups(league, path, profile_path, waiting_time, num_previous_matches)
        lineup_df = pd.DataFrame.from_dict(lineups, orient='index')
        lineup_df.to_csv(f'{league_name}_{season_start}_{season_end}_recent_lineups.csv', index_label='match_id')
        
def merge_recent_leagues_columns(leagues,columns):
    merged_df = pd.DataFrame()
    old_league = leagues[0]
    new_league = leagues[1]
    for league in leagues:
        #Read leagues
        df_old = pd.read_csv(f'{old_league}.csv')
        df_new = pd.read_csv(f'{new_league}.csv')
        old_season = df_old['season_year'].iloc[-1]
        old_round = df_old['round'].iloc[-1]
        removed = (df_old['season_year'] == old_season) & (df_old['round'] == old_round)
        #remaining_rows = len(df_old[~removed])
        for col in columns:
            if col not in df_new.columns:
                df_new[col] = 0  # Fill missing columns with zeros
        df_new = df_new[columns]
        #Merge leagues
        merged_df = pd.concat([df_old[~removed], df_new], axis=0, ignore_index=True)

        # Normalize standings
        #merged_df['home_standing'] = merged_df['home_standing'] / merged_df['home_standing'].max()
        #merged_df['away_standing'] = merged_df['away_standing'] / merged_df['away_standing'].max()

    # Drop the 'index' column if it exists
    #if 'index' in merged_df.columns:
    #    merged_df.drop('index', axis=1, inplace=True)
    merged_df.drop('Unnamed: 0', axis=1, inplace=True)

    # Fill missing values with 0
    merged_df.fillna(0, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_csv(f'{old_league}.csv', index=True)
    
    # Delete the recent file
    os.remove(f'{new_league}.csv')
