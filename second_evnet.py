import requests
import json
from time import sleep

for i in range(1, 53, 1):
	print(i)
	sleep(0.5)
	resp = requests.get("https://api.matsurihi.me/mltd/v1/events/92/rankings/logs/idolPoint/" + str(i) + "/10,100,250,500,1000")
	info_json = resp.json()

	with open('./event/second_event_border/idol_num' + str(i) + '.json', 'w', encoding="utf-8") as make_file:
		json.dump(info_json, make_file, ensure_ascii=False, indent="\t")
