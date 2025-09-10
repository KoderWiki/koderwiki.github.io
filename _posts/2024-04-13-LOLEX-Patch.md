---
layout: post
title: LOLEX BUG FIXES
tags: [Web Develop, League of Legends, LOLEX]
feature-img: "assets/img/0.post/2024-04-13/header2.png"
thumbnail: "assets/img/0.post/2024-04-13/header2.png"
categories: LOLEX
---

This patch is fist patch in [**LOLEX**](https://ko-web.com/lolex). This patch is not added new fuction and deleted some fuctions, also **fixed a fatal BUGs**. "

For more information on **LOLEX**, Please click [**here**](https://koderwiki.github.io/lolex/2024/02/15/LOLEX-Release.html)! <br>

## [BUG FIXES.1] match-v5 returns meaningless data 

I've found that **some of match historys didn't recieved data properly**. After looking at the cause, we can find out that **the cause was RIOT API**.

```json
"info": {
        "endOfGameResult": "Abort_Unexpected",
        "frameInterval": 0,
        "frames": [
            {
                "events": [
                    {
                        "gameId": 6834713231,
                        "realTimestamp": 1709066369872,
                        "timestamp": 0,
                        "type": "GAME_END",
                        "winningTeam": 0
                    }
                ],
                "participantFrames": null,
                "timestamp": 0
            }
        ],
        "gameId": 0,
        "participants": []
    }
```
For some reason, The data sent by Riot API didn't contain anything. 

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/95ccb3d1-e534-4d59-9dd7-7160071ed63f)

## [BUG FIXES.2] Changed structure of SUMMONER-V4

I've found that Summoner's name is inaccurate. It because, structure of SUMMONER-V4 has been changed.

```python
    puuid = id_data['puuid']
    name_url = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{}?api_key={}'.format(puuid,apikey)
    summoner_info = get_json(name_url)
    lol_name = name # Modified
    encrypted_id = summoner_info['id']
    profile_id = summoner_info['profileIconId']
    summoner_level = summoner_info['summonerLevel']
```

We solved this problem by receiving another variable.

## [CHANGES.1] Changes in search format

**The By SummonerName endpoint** in SUMMONER-V$ is deprecated as part of the transition from Summoner Name to **Riot ID**, so this function will be removed on April 22.

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/94007eb3-6505-4639-a045-6e242b995612)

So out search format also changed! <br>
From now on, you won't be able to search with out **RIOT TAG**.

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/6c563352-c313-4cf3-9cef-a8d2389037c7)

```javascript
function searchPost(){
        let searchValue = document.getElementById("search-input").value.trim();
        searchValue = searchValue.replace(/#/g,"-")
        if(searchValue.length > 1){
          if(searchValue.indexOf('-') != -1){
            fetch("/lolex/summoner?name="+searchValue+"/")
            location.href="/lolex/summoner?name="+searchValue+"/";
          }
          else{
            alert('Riot tag가 존재하지않습니다.')
          } 
        }
        else{          
          alert('소환사 이름이 너무 짧습니다.');
        }

        

    }

    document.getElementById('search-input').addEventListener('keyup', function(event)
    {
        if(event.key=='Enter'){
            searchPost();
        }
    });
```

## [UPDATES.2] Delete the Leaderboard

**We've deleted the leaderboard function** being fixed. But we made new function. <br>

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/874303f5-3827-4e6f-bf40-bf428e5f2c66)

You will now be able to see **your Challenges**!

**After change**

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/3a53cb11-5174-46f7-b9b9-fb296b1c33be)





































