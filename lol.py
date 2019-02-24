import requests
import json

for i in range(1, 75, 1):
    with open('./event/event_info' + str(i) + '.json', 'r', encoding="utf-8") as read_file:
        cur_event_data = json.load(read_file)
    if len(cur_event_data) == 4 or len(cur_event_data) == 5:
        if cur_event_data["type"]==3 or cur_event_data["type"]==4: #3 is theater, 4 is tour
            resp = requests.get("https://api.matsurihi.me/mltd/v1/events/" + str(i) + "/rankings/borders")
            ranking_limit_data = resp.json() 
            
            borders = ""
            print(i)
            if len(ranking_limit_data) != 3:
                print("old")
                borders = "100,2000,5000,10000,20000,500000"

            else:
                print("new")
                border_limit = ranking_limit_data["eventPoint"]
                print(border_limit)

                borders = str(border_limit[0]) + "," + str(border_limit[1]) + "," + str(border_limit[2]) + "," + \
                        str(border_limit[3]) + "," + str(border_limit[4]) + "," + str(border_limit[5])

            resp = requests.get("https://api.matsurihi.me/mltd/v1/events/"+str(i)+"/rankings/logs/eventPoint/" + borders)
            borders_json = resp.json()
            with open('./event/event_border' + str(i) + '.json', 'w', encoding="utf-8") as make_file:
                json.dump(borders_json, make_file, ensure_ascii=False, indent="\t")