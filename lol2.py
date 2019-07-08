import json
import matplotlib.pyplot as plt

tour_point = [[1142099, 628649, 537106, 471605, 849323, 650634, 1164978, 1134796],
              [371237, 207072, 248240, 161014, 305366, 149580, 394587, 394538],
              [137697, 72108, 79122, 68412, 93860, 67752, 114120, 112870],
              [83660, 51044, 55386, 52578, 59695, 52554, 64842, 67280],
              [65680, 40866, 43884, 41312, 50028, 41714, 51326, 53591],
              [51883, 30620, 32622, 28060, 36252, 30018, 40160, 44212]]

theater_point = [[0, 0, 0, 290501, 250399, 491447, 711803, 1032211, 1126660, 493391],
                 [151347, 103398, 135591, 88037, 87912, 123222, 197135, 317690, 362962, 102704],
                 [67477, 54175, 56477, 44374, 41800, 54202, 69542, 96384, 103541, 55400],
                 [45991, 43409, 41397, 36350, 33099, 40630, 51152, 55556, 57721, 46603],
                 [36947, 36992, 36207, 26413, 22928, 32449, 37159, 42724, 44193, 36562],
                 [28172, 25865, 23472, 18029, 14889, 20195, 23018, 30196, 31049, 21042]]

ranking = ["#100", "#2500(2000)", "#5000", "#10000", "#25000(20000)", "#50000"]

for i in range(76, 93, 1):
    print(i)

    with open('./event/event_info' + str(i) + '.json', 'r', encoding="utf-8") as read_file:
        cur_event_data = json.load(read_file)
    if len(cur_event_data) == 4 or len(cur_event_data) == 5:
        if cur_event_data["type"]==3 or cur_event_data["type"]==4: #3 is theater, 4 is tour
            with open('./event/event_border' + str(i) + '.json', 'r', encoding="utf-8") as read_file:
                cur_border_data = json.load(read_file)

                if cur_event_data["type"]==3:
                    for j in range(0,6):
                        theater_point[j].append(cur_border_data[j]["data"][len(cur_border_data[j]["data"]) - 1]["score"])
                elif cur_event_data["type"]==4:
                    for j in range(0,6):
                        tour_point[j].append(cur_border_data[j]["data"][len(cur_border_data[j]["data"]) - 1]["score"])

for i in range(6):
    plt.title("MLTD event " + ranking[i])
    X = range(1,len(tour_point[0]) + 1)
    plt.plot(X, tour_point[i][:len(tour_point[0])], ls=":", label="Tour", marker="o")
    plt.annotate(' ', xy=(9, tour_point[i][8]), arrowprops=dict(facecolor='skyblue', shrink=0.01))

    X = range(1,len(theater_point[0]) + 1)
    plt.plot(X, theater_point[i], ls=":", label="Theater", marker="o")
    plt.annotate(' ', xy=(11, theater_point[i][10]), arrowprops=dict(facecolor='orange', shrink=0.01))

    plt.legend(loc=2)
    plt.xlim(1, 18)
    if i == 0:
        plt.ylim(190000)
    plt.grid(True)
    plt.savefig("./"+ ranking[i] +".png", dpi=600)
    plt.show()
