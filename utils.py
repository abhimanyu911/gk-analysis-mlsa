import requests
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from datetime import datetime
import time
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import sys, getopt
import csv

league_url_idx = {
        'Premier League':9,
        'Bundesliga': 20,
        'La Liga' : 12,
        'Serie A': 11,
        'Ligue 1': 13
    }

feature_transcriber = {
    'gk_passes_length_avg' : 'average length of GK passes',
    'gk_goal_kick_length_avg' : 'average length of goalkicks',
    'gk_psxg_net' : 'average PSxG-GA',
    'gk_save_pct' : 'average Save Percentage'
}

y_axis_unit = {
    'gk_passes_length_avg' : '(yards)',
    'gk_goal_kick_length_avg': '(yards)',
    'gk_psxg_net' : '',
    'gk_save_pct' : '(%)'
}

def encode_season(season_str: str) -> str:
    """
    If 'season_str' matches the current soccer season (e.g., '2024-2025'),
    return an empty string; otherwise, return the original 'season_str'.
    """
    this_year = datetime.now().year
    next_year = this_year + 1

    current_season = f"{this_year}-{next_year}"
    
    if season_str == current_season:
        return "",""
    return season_str.split("-")[0],season_str+'-'


#standard(stats)
stats = ["player","nationality","position","team","age","birth_year","games","games_starts","minutes","goals","assists","pens_made","pens_att","cards_yellow","cards_red","goals_per90","assists_per90","goals_assists_per90","goals_pens_per90","goals_assists_pens_per90","xg","npxg","xa","xg_per90","xa_per90","xg_xa_per90","npxg_per90","npxg_xa_per90"]
stats3 = ["players_used","possession","games","games_starts","minutes","goals","assists","pens_made","pens_att","cards_yellow","cards_red","goals_per90","assists_per90","goals_assists_per90","goals_pens_per90","goals_assists_pens_per90","xg","npxg","xa","xg_per90","xa_per90","xg_xa_per90","npxg_per90","npxg_xa_per90"] 
#goalkeeping(keepers)
keepers = ["player","nationality","position","team","age","birth_year","gk_games","gk_games_starts","gk_minutes","gk_goals_against","gk_goals_against_per90","gk_shots_on_target_against","gk_saves","gk_save_pct","gk_wins","gk_ties","gk_losses","gk_clean_sheets","gk_clean_sheets_pct","gk_pens_att","gk_pens_allowed","gk_pens_saved","gk_pens_missed"]
keepers3 = ["players_used","gk_games","gk_games_starts","gk_minutes","gk_goals_against","gk_goals_against_per90","gk_shots_on_target_against","gk_saves","gk_save_pct","gk_wins","gk_ties","gk_losses","gk_clean_sheets","gk_clean_sheets_pct","gk_pens_att","gk_pens_allowed","gk_pens_saved","gk_pens_missed"]
#advance goalkeeping(keepersadv)
keepersadv = ["player","nationality","position","team","age","birth_year","minutes_90s","gk_goals_against","gk_pens_allowed","gk_free_kick_goals_against","gk_corner_kick_goals_against","gk_own_goals_against","gk_psxg","gk_psnpxg_per_shot_on_target_against","gk_psxg_net","gk_psxg_net_per90","gk_passes_completed_launched","gk_passes_launched","gk_passes_pct_launched","gk_passes","gk_passes_throws","gk_pct_passes_launched","gk_passes_length_avg","gk_goal_kicks","gk_pct_goal_kicks_launched","gk_goal_kick_length_avg","gk_crosses","gk_crosses_stopped","gk_crosses_stopped_pct","gk_def_actions_outside_pen_area","gk_def_actions_outside_pen_area_per90","gk_avg_distance_def_actions"]
keepersadv2 = ["minutes_90s","gk_goals_against","gk_pens_allowed","gk_free_kick_goals_against","gk_corner_kick_goals_against","gk_own_goals_against","gk_psxg","gk_psnpxg_per_shot_on_target_against","gk_psxg_net","gk_psxg_net_per90","gk_passes_completed_launched","gk_passes_launched","gk_passes_pct_launched","gk_passes","gk_passes_throws","gk_pct_passes_launched","gk_passes_length_avg","gk_goal_kicks","gk_pct_goal_kicks_launched","gk_goal_kick_length_avg","gk_crosses","gk_crosses_stopped","gk_crosses_stopped_pct","gk_def_actions_outside_pen_area","gk_def_actions_outside_pen_area_per90","gk_avg_distance_def_actions"]
#shooting(shooting)
shooting = ["player","nationality","position","team","age","birth_year","minutes_90s","goals","pens_made","pens_att","shots_total","shots_on_target","shots_free_kicks","shots_on_target_pct","shots_total_per90","shots_on_target_per90","goals_per_shot","goals_per_shot_on_target","xg","npxg","npxg_per_shot","xg_net","npxg_net"]
shooting2 = ["minutes_90s","goals","pens_made","pens_att","shots_total","shots_on_target","shots_free_kicks","shots_on_target_pct","shots_total_per90","shots_on_target_per90","goals_per_shot","goals_per_shot_on_target","xg","npxg","npxg_per_shot","xg_net","npxg_net"]
shooting3 = ["goals","pens_made","pens_att","shots_total","shots_on_target","shots_free_kicks","shots_on_target_pct","shots_total_per90","shots_on_target_per90","goals_per_shot","goals_per_shot_on_target","xg","npxg","npxg_per_shot","xg_net","npxg_net"]
#passing(passing)
passing = ["player","nationality","position","team","age","birth_year","minutes_90s","passes_completed","passes","passes_pct","passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa","xa_net","assisted_shots","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
passing2 = ["passes_completed","passes","passes_pct","passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa","xa_net","assisted_shots","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
#passtypes(passing_types)
passing_types = ["player","nationality","position","team","age","birth_year","minutes_90s","passes","passes_live","passes_dead","passes_free_kicks","through_balls","passes_pressure","passes_switches","crosses","corner_kicks","corner_kicks_in","corner_kicks_out","corner_kicks_straight","passes_ground","passes_low","passes_high","passes_left_foot","passes_right_foot","passes_head","throw_ins","passes_other_body","passes_completed","passes_offsides","passes_oob","passes_intercepted","passes_blocked"]
passing_types2 = ["passes","passes_live","passes_dead","passes_free_kicks","through_balls","passes_pressure","passes_switches","crosses","corner_kicks","corner_kicks_in","corner_kicks_out","corner_kicks_straight","passes_ground","passes_low","passes_high","passes_left_foot","passes_right_foot","passes_head","throw_ins","passes_other_body","passes_completed","passes_offsides","passes_oob","passes_intercepted","passes_blocked"]
#goal and shot creation(gca)
gca = ["player","nationality","position","team","age","birth_year","minutes_90s","sca","sca_per90","sca_passes_live","sca_passes_dead","sca_dribbles","sca_shots","sca_fouled","gca","gca_per90","gca_passes_live","gca_passes_dead","gca_dribbles","gca_shots","gca_fouled","gca_defense"]
gca2 = ["sca","sca_per90","sca_passes_live","sca_passes_dead","sca_dribbles","sca_shots","sca_fouled","gca","gca_per90","gca_passes_live","gca_passes_dead","gca_dribbles","gca_shots","gca_fouled","gca_defense"]
#defensive actions(defense)
defense = ["player","nationality","position","team","age","birth_year","minutes_90s","tackles","tackles_won","tackles_def_3rd","tackles_mid_3rd","tackles_att_3rd","dribble_tackles","dribbles_vs","dribble_tackles_pct","dribbled_past","pressures","pressure_regains","pressure_regain_pct","pressures_def_3rd","pressures_mid_3rd","pressures_att_3rd","blocks","blocked_shots","blocked_shots_saves","blocked_passes","interceptions","clearances","errors"]
defense2 = ["tackles","tackles_won","tackles_def_3rd","tackles_mid_3rd","tackles_att_3rd","dribble_tackles","dribbles_vs","dribble_tackles_pct","dribbled_past","pressures","pressure_regains","pressure_regain_pct","pressures_def_3rd","pressures_mid_3rd","pressures_att_3rd","blocks","blocked_shots","blocked_shots_saves","blocked_passes","interceptions","clearances","errors"]
#possession(possession)
possession = ["player","nationality","position","team","age","birth_year","minutes_90s","touches","touches_def_pen_area","touches_def_3rd","touches_mid_3rd","touches_att_3rd","touches_att_pen_area","touches_live_ball","dribbles_completed","dribbles","dribbles_completed_pct","players_dribbled_past","nutmegs","carries","carry_distance","carry_progressive_distance","progressive_carries","carries_into_final_third","carries_into_penalty_area","pass_targets","passes_received","passes_received_pct","miscontrols","dispossessed"]
possession2 = ["touches","touches_def_pen_area","touches_def_3rd","touches_mid_3rd","touches_att_3rd","touches_att_pen_area","touches_live_ball","dribbles_completed","dribbles","dribbles_completed_pct","players_dribbled_past","nutmegs","carries","carry_distance","carry_progressive_distance","progressive_carries","carries_into_final_third","carries_into_penalty_area","pass_targets","passes_received","passes_received_pct","miscontrols","dispossessed"]
#playingtime(playingtime)
playingtime = ["player","nationality","position","team","age","birth_year","minutes_90s","games","minutes","minutes_per_game","minutes_pct","games_starts","minutes_per_start","games_subs","minutes_per_sub","unused_subs","points_per_match","on_goals_for","on_goals_against","plus_minus","plus_minus_per90","plus_minus_wowy","on_xg_for","on_xg_against","xg_plus_minus","xg_plus_minus_per90","xg_plus_minus_wowy"]
playingtime2 = ["games","minutes","minutes_per_game","minutes_pct","games_starts","minutes_per_start","games_subs","minutes_per_sub","unused_subs","points_per_match","on_goals_for","on_goals_against","plus_minus","plus_minus_per90","plus_minus_wowy","on_xg_for","on_xg_against","xg_plus_minus","xg_plus_minus_per90","xg_plus_minus_wowy"]
#miscallaneous(misc)
misc = ["player","nationality","position","team","age","birth_year","minutes_90s","cards_yellow","cards_red","cards_yellow_red","fouls","fouled","offsides","crosses","interceptions","tackles_won","pens_won","pens_conceded","own_goals","ball_recoveries","aerials_won","aerials_lost","aerials_won_pct"]
misc2 = ["cards_yellow","cards_red","cards_yellow_red","fouls","fouled","offsides","crosses","interceptions","tackles_won","pens_won","pens_conceded","own_goals","ball_recoveries","aerials_won","aerials_lost","aerials_won_pct"]






#Functions to get the data in a dataframe using BeautifulSoup

'''def get_tables(url, text):
    all_tables = []
    
    while not all_tables:
        res = requests.get(url)
        comm = re.compile("<!--|-->")
        soup = BeautifulSoup(comm.sub("", res.text), 'lxml')
        all_tables = soup.findAll("tbody")
    
    team_table = all_tables[0]
    team_vs_table = all_tables[1]
    player_table = all_tables[2]
    
    if text == 'for':
        return player_table, team_table
    if text == 'vs':
        return player_table, team_vs_table

    return None, None'''

def get_tables(url,text):
    res = requests.get(url)
    ## The next two lines get around the issue with comments breaking the parsing.
    comm = re.compile("")
    soup = BeautifulSoup(comm.sub("",res.text),'lxml')
    all_tables = soup.findAll("tbody")
    
    team_table = all_tables[0]
    team_vs_table = all_tables[1]
    player_table = all_tables[2]
    if text == 'for':
      return player_table, team_table
    if text == 'vs':
      return player_table, team_vs_table

def get_frame(features, player_table):
    pre_df_player = dict()
    features_wanted_player = features
    rows_player = player_table.find_all('tr')
    for row in rows_player:
        if(row.find('th',{"scope":"row"}) != None):
    
            for f in features_wanted_player:
                cell = row.find("td",{"data-stat": f})
                if cell is not None:
                    a = cell.text.strip().encode()
                text=a.decode("utf-8")
                if(text == ''):
                    text = '0'
                if(not(f == 'player') and not(f == 'nationality') and not(f == 'position') and not(f == 'team') and not(f == 'age') and not(f == 'birth_year') ): #and not(isinstance(text,str))
                    text = float(text.replace(',',''))
                if f in pre_df_player:
                    pre_df_player[f].append(text)
                else:
                    pre_df_player[f] = [text]
    df_player = pd.DataFrame.from_dict(pre_df_player)
    return df_player

'''def get_frame_team(features, team_table):
    pre_df_squad = dict()
    #Note: features does not contain squad name, it requires special treatment
    features_wanted_squad = features
    rows_squad = team_table.find_all('tr')
    for row in rows_squad:
        print(row)
        if(row.find('th',{"scope":"row"}) != None):
            name = row.find('th',{"data-stat":"squad"}).text.strip().encode().decode("utf-8")
            if 'squad' in pre_df_squad:
                pre_df_squad['squad'].append(name)
            else:
                pre_df_squad['squad'] = [name]
            for f in features_wanted_squad:
                cell = row.find("td",{"data-stat": f})
                if cell is not None:
                    a = cell.text.strip().encode()
                text=a.decode("utf-8")
                if(text == ''):
                    text = '0'
                if(not(f == 'player') and not(f == 'nationality') and not(f == 'position') and not(f == 'squad') and not(f == 'age') and not(f == 'birth_year') and not(isinstance(text,str))):
                    text = float(text.replace(',',''))
                if f in pre_df_squad:
                    pre_df_squad[f].append(text)
                else:
                    pre_df_squad[f] = [text]
    df_squad = pd.DataFrame.from_dict(pre_df_squad)
    return df_squad'''

def get_frame_team(features, team_table):
    pre_df_squad = dict()
    features_wanted_squad = features
    rows_squad = team_table.find_all('tr')

    for row in rows_squad:
        name_cell = row.find('th', {"data-stat": "team"})
        if name_cell:
            name = name_cell.text.strip()
            if 'squad' in pre_df_squad:
                pre_df_squad['squad'].append(name)
            else:
                pre_df_squad['squad'] = [name]
            cells = row.find_all('td')
            for f in features_wanted_squad:
                data_cell = row.find('td', {"data-stat": f})
                if data_cell:
                    text = data_cell.text.strip()
                    if text == '':
                        text = '0'
                    if f != 'squad':
                        text = float(text.replace(',', ''))
                    if f in pre_df_squad:
                        pre_df_squad[f].append(text)
                    else:
                        pre_df_squad[f] = [text]
    
    df_squad = pd.DataFrame.from_dict(pre_df_squad)
    return df_squad




def frame_for_category(category,top,end,features):
    url = (top + category + end)
    player_table, team_table = get_tables(url,'for')
    df_player = get_frame(features, player_table)
    return df_player

def frame_for_category_team(category,top,end,features,text):
    url = (top + category + end)
    player_table, team_table = get_tables(url,text)
    df_team = get_frame_team(features, team_table)
    return df_team

#Function to get the player data for outfield player, includes all categories - standard stats, shooting
#passing, passing types, goal and shot creation, defensive actions, possession, and miscallaneous
def get_outfield_data(top, end):
    df1 = frame_for_category('stats',top,end,stats)
    df2 = frame_for_category('shooting',top,end,shooting2)
    df3 = frame_for_category('passing',top,end,passing2)
    df4 = frame_for_category('passing_types',top,end,passing_types2)
    df5 = frame_for_category('gca',top,end,gca2)
    df6 = frame_for_category('defense',top,end,defense2)
    df7 = frame_for_category('possession',top,end,possession2)
    df8 = frame_for_category('misc',top,end,misc2)
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    return df


#Function to get keeping and advance goalkeeping data
def get_keeper_data(top,end):
    df1 = frame_for_category('keepers',top,end,keepers)
    df2 = frame_for_category('keepersadv',top,end,keepersadv2)
    df = pd.concat([df1, df2], axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    return df



#Function to get team-wise data accross all categories as mentioned above
def get_team_data(top,end,text):
    df1 = frame_for_category_team('stats',top,end,stats3,text)
    df2 = frame_for_category_team('keepers',top,end,keepers3,text)
    df3 = frame_for_category_team('keepersadv',top,end,keepersadv2,text)
    df4 = frame_for_category_team('shooting',top,end,shooting3,text)
    df5 = frame_for_category_team('passing',top,end,passing2,text)
    df6 = frame_for_category_team('passing_types',top,end,passing_types2,text)
    df7 = frame_for_category_team('gca',top,end,gca2,text)
    df8 = frame_for_category_team('defense',top,end,defense2,text)
    df9 = frame_for_category_team('possession',top,end,possession2,text)
    df10 = frame_for_category_team('misc',top,end,misc2,text)
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    return df


def get_adv_keeper_metrics_per_team(top, end, text):

    df = frame_for_category_team('keepersadv',top,end,keepersadv2,text)

    df = df.loc[:,~df.columns.duplicated()]

    return df

def get_trad_keeper_metrics_per_team(top, end, text):

    df = frame_for_category_team('keepers',top,end,keepers3,text)

    df = df.loc[:,~df.columns.duplicated()]

    return df


def calc_and_plot_adv_trends(feature = None, seasons = None):

    time_series_values = []

    for season in seasons:
        if season == '2024-2025':
            top_year = ''
            end_year = ''
        else:
            top_year = season
            end_year = season + '-'
        top = f'https://fbref.com/en/comps/9/{top_year}/'
        end = f'/{end_year}Premier-League-Stats'

        df = get_adv_keeper_metrics_per_team(
            top = top,
            end = end,
            text = 'for'
        )

        print(df[feature].mean())
        time_series_values.append(df[feature].mean())

        indices = [i for i in range(len(time_series_values))]
        tau, p_value = kendalltau(indices,time_series_values)

        if p_value < 0.05:
            if tau > 0:
                print(f"Increasing trend for p = {p_value} and tau = {tau}")
            else:
                print(f"Decreasing trend for p = {p_value} and tau = {tau}")
        else:
            print(f"No significant trend detected for p = {p_value} and tau = {tau}")
        

def calc_and_plot_adv_trends(feature=None, seasons=None):
    leagues = ['Premier League', 'La Liga']
    time_series_values = [[] for _ in range(len(leagues))]
    
    for season in seasons:
        if season == '2024-2025':
            top_year = ''
            end_year = ''
        else:
            top_year = season
            end_year = season + '-'
        
        for i, league in enumerate(leagues):
            url_idx_value = league_url_idx[league]
            top = f'https://fbref.com/en/comps/{url_idx_value}/{top_year}/'
            end = f'/{end_year}{league.replace(" ", "-")}-Stats'

            #print(top, end)

            df = get_adv_keeper_metrics_per_team(
                top=top,
                end=end,
                text='for'
            )

            #print(f"League: {league}, Season: {season}")
            #print(df[feature].mean())
            time_series_values[i].append(df[feature].mean())

    plt.figure(figsize=(10, 6))

    for i, league in enumerate(leagues):
        plt.plot(seasons, time_series_values[i], label=league)

    plt.xlabel('Season')
    plt.ylabel(f'Mean of the {feature_transcriber[feature]}{y_axis_unit[feature]}')
    plt.title(f"Mean of the {feature_transcriber[feature]} for all teams across 6 seasons")
    plt.legend()
    plt.show()
    plt.savefig(f'./{feature}.eps', format='eps')

    table_data = []
    table_header = ['League', 'Tau', 'P-Value', 'Trend']

    for i, league in enumerate(leagues):
        indices = [j for j in range(len(time_series_values[i]))]
        tau, p_value = kendalltau(indices, time_series_values[i])

        if p_value < 0.05:
            if tau > 0:
                trend = f"Increasing trend (p = {p_value:.4f}, tau = {tau:.4f})"
            else:
                trend = f"Decreasing trend (p = {p_value:.4f}, tau = {tau:.4f})"
        else:
            trend = f"No significant trend (p = {p_value:.4f}, tau = {tau:.4f})"

        table_data.append([league, tau, p_value, trend])

    trends_df = pd.DataFrame(table_data, columns=table_header)
    print(trends_df)


def get_passing_stats_per_team(league, season):
    # Function implementation to retrieve passing stats per team for a given league and season
    if season == '2024-2025':
        top_year = ''
        end_year = ''
    else:
        top_year = season
        end_year = season + '-'
    
    url_idx_value = league_url_idx[league]
    top = f'https://fbref.com/en/comps/{url_idx_value}/{top_year}/'
    end = f'/{end_year}{league.replace(" ", "-")}-Stats'
    
    pass_stats_df = frame_for_category_team('passing',top,end,passing2,'for')
    return pass_stats_df

def plot_passing_boxplots():
    leagues = ['Premier League', 'La Liga']
    features = ['passes_pct_short', 'passes_pct_long']
    seasons = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']
    plt.figure(figsize=(4, 4))

    all_pass_stats_short = []
    all_pass_stats_long = []

    for league in leagues:
        for season in seasons:
            pass_stats_df = get_passing_stats_per_team(league, season)
            pass_stats_short = list(pass_stats_df[features[0]])
            pass_stats_long = list(pass_stats_df[features[1]])

            all_pass_stats_short.extend(pass_stats_short)
            all_pass_stats_long.extend(pass_stats_long)

    plt.boxplot(all_pass_stats_short, positions=[1], widths=0.6, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='skyblue'), capprops=dict(color='black'), whiskerprops=dict(color='black'),
                medianprops=dict(color='black'), flierprops=dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='black'))
    plt.boxplot(all_pass_stats_long, positions=[2], widths=0.6, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='lightgreen'), capprops=dict(color='black'), whiskerprops=dict(color='black'),
                medianprops=dict(color='black'), flierprops=dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='black'))

    plt.xticks(range(1, 3), ['Short Passes', 'Long Passes'])
    plt.xlabel('Pass Type')
    plt.ylabel('Pass Accuracy(%)')
    plt.rcParams['axes.titlesize'] = 10
    title_lines = ['Boxplots of', 'Short and Long Passes', 'across the La Liga and Premier League', '(2017-2023)']
    title = '\n'.join(title_lines)
    print(f"Short mean = {np.mean(all_pass_stats_short)}, short std = {np.std(all_pass_stats_short)}")
    print(f"Long mean = {np.mean(all_pass_stats_long)}, long std = {np.std(all_pass_stats_long)}")
    plt.title(title, pad=10, loc='center')

    # Create a legend
    short_pass_box = plt.Rectangle((0, 0), 1, 1, fc='skyblue')
    long_pass_box = plt.Rectangle((0, 0), 1, 1, fc='lightgreen')
    plt.legend([short_pass_box, long_pass_box], ['Short Passes', 'Long Passes'])

    plt.tight_layout()
    plt.show()
    plt.savefig('./boxplot.eps', format='eps')


def calc_and_plot_trad_trends(feature=None, seasons=None):
    leagues = ['Premier League', 'La Liga']
    time_series_values = [[] for _ in range(len(leagues))]
    
    for season in seasons:
        if season == '2024-2025':
            top_year = ''
            end_year = ''
        else:
            top_year = season
            end_year = season + '-'
        
        for i, league in enumerate(leagues):
            url_idx_value = league_url_idx[league]
            top = f'https://fbref.com/en/comps/{url_idx_value}/{top_year}/'
            end = f'/{end_year}{league.replace(" ", "-")}-Stats'

            #print(top, end)

            df = get_trad_keeper_metrics_per_team(
                top=top,
                end=end,
                text='for'
            )

            #print(f"League: {league}, Season: {season}")
            #print(df[feature].mean())
            time_series_values[i].append(df[feature].mean())

    plt.figure(figsize=(10, 6))

    for i, league in enumerate(leagues):
        plt.plot(seasons, time_series_values[i], label=league)

    plt.xlabel('Season')
    plt.ylabel(f'Mean of the {feature_transcriber[feature]}{y_axis_unit[feature]}')
    plt.title(f"Mean of the {feature_transcriber[feature]} for all teams across 6 seasons")
    plt.legend()
    plt.show()
    plt.savefig(f'./{feature}.eps', format='eps')

    table_data = []
    table_header = ['League', 'Tau', 'P-Value', 'Trend']

    for i, league in enumerate(leagues):
        indices = [j for j in range(len(time_series_values[i]))]
        tau, p_value = kendalltau(indices, time_series_values[i])

        if p_value < 0.05:
            if tau > 0:
                trend = f"Increasing trend (p = {p_value:.4f}, tau = {tau:.4f})"
            else:
                trend = f"Decreasing trend (p = {p_value:.4f}, tau = {tau:.4f})"
        else:
            trend = f"No significant trend (p = {p_value:.4f}, tau = {tau:.4f})"

        table_data.append([league, tau, p_value, trend])

    trends_df = pd.DataFrame(table_data, columns=table_header)
    print(trends_df)