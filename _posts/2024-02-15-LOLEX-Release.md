---
layout: post
title: LOLEX
tags: [Web Develop, League of Legends, LOLEX]
feature-img: "assets/img/0.post/2024-02-15/header2.png"
thumbnail: "assets/img/0.post/2024-02-15/header.png"
categories: LOLEX
---

[**LOLEX**](http://ko-web.com/lolex) is my first own web site and **League of Legends's** game data platform. <br>

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/9a5bb424-e742-4237-bf63-1cfcea5dda23)

## **LOLEX**
**Developer** &nbsp;&nbsp;&nbsp;&nbsp; K.Geonu <br>
**Type of site** &nbsp;&nbsp; Game data platform <br>
**Written in** &nbsp;&nbsp;&nbsp;&nbsp; Flask(Python) <br>
**Languages** &nbsp;&nbsp;&nbsp; Korean(한국어) <br>
**Released** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2024-02-15 <br>

## Main Service

### 1. Match History & Summoner's Info

It is the most basic funtion of this site. Using **RIOT Api**, Showing Summoner's Info, Match data and Graph, etc..

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/6d8d8bf7-526e-42c7-816f-56b00f3d0e48)

### 2. Champion Mastery

It shows the summoner's champion Mastery list you want

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/e9d6f6f0-1dd9-4697-9a3f-55edfb024ff8)

### 3. Process of Running

#### 3.1 Summoner's Info

**Process**<br>
![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/696803ab-cd34-4f53-978e-d5b7859d4817)

```python
import requests

def get_result(table:dict, targets:list) -> list[dict]:
  result = list()
  for target in targets:
    if target in table:
      result.append(table[target])
    else:
      print(f"{target} not in table")
  return result

def get_json(url):
  json = requests.get(url)
  data = json.json()
  return data

def get_data(name, apikey): #riot_id = 최고의피니셔 riot_tag = KR1 lol_name = 최고의피니셔
  name = name[:-1]

  if '-' in name: #최고의피니셔#kr1
    riot_id = name[:name.find('-')]
    riot_tag = name[name.find('-')+1:]
    id_url = 'https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{}/{}?api_key={}'.format(riot_id,riot_tag,apikey)
    id_data = get_json(id_url)

    if len(id_data) == 1:
      return "error"

    puuid = id_data['puuid']
    name_url = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{}?api_key={}'.format(puuid,apikey)
    summoner_info = get_json(name_url)
    lol_name = name
    encrypted_id = summoner_info['id']
    profile_id = summoner_info['profileIconId']
    summoner_level = summoner_info['summonerLevel']
    print(puuid)

    return puuid, lol_name, riot_id, riot_tag, encrypted_id, profile_id, summoner_level
  
  else: #최고의피니셔
    name_url = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{}?api_key={}'.format(name,apikey)
    name_data = get_json(name_url)

    if len(name_data) == 1:
      return "error"

    lol_name = name
    puuid = name_data['puuid']
    encrypted_id = name_data['id']
    profile_id = name_data['profileIconId']
    summoner_level = name_data['summonerLevel']
    id_url = 'https://asia.api.riotgames.com/riot/account/v1/accounts/by-puuid/{}?api_key={}'.format(puuid,apikey)
    id_data = get_json(id_url)
    riot_id = id_data['gameName']
    riot_tag = id_data['tagLine']
    summoner_info = name_data

    return puuid, lol_name, riot_id, riot_tag, encrypted_id, profile_id, summoner_level
```

### 3.2 Match History

**Process**<br>
![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/fcc6ac70-d8bc-4de1-ad0a-c322f8b3e690)

```python
def get_match(puuid, lol_version, start_count, end_count, apikey):
	global regame_count
	global win_count
	global loss_count

	url = 'https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{}/ids?start={}&count={}&api_key={}'.format(puuid, start_count, end_count, apikey)
	match_id_data = get_json(url)

	match_list = list()
	for i in range(len(match_id_data)):
		match_list.append('https://asia.api.riotgames.com/lol/match/v5/matches/{}?api_key={}'.format(match_id_data[i], apikey))

	match_info = [[0 for k in range(1)] for l in range(len(match_list))]

	for i in range(len(match_list)):
		match_data = get_json(match_list[i])
		match_id = match_id_data[i]
			
		player_info = match_data['info']['participants']
		player_metadata = match_data['metadata']['participants']


		player_data = player_info[player_metadata.index(puuid)]
		player_queue = match_data['info']['queueId']

		try:
			queue_type = QUEUE_TABLE[player_queue]
		except:
			queue_type = '특별게임모드'

		if queue_type == '아레나':
			player_match = get_special_match(player_data, match_id, match_data, player_info, queue_type, lol_version)
			match_info[i] = player_match
		else:
			player_match = get_classic_match(player_data, match_id, match_data, player_info, queue_type, lol_version)
			match_info[i] = player_match

	return match_info
```

## History

**Beta Version**

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/70f105c0-33dd-4651-9d20-38f2fd730865)

**Recent Version**

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/6d8d8bf7-526e-42c7-816f-56b00f3d0e48)














