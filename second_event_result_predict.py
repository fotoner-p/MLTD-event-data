import tensorflow as tf
import json
import datetime

import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt

tf.set_random_seed(765)

seq_length = 10  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
data_dim = 2  # Variable 개수
hidden_dim = 10  # 각 셀의 출력 크기
output_dim = 1  # 결과 분류 총 수
learning_rate = 0.01  # 학습률
epoch_num = 300  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)

point_data = []

def MinMaxScaler(data):
    # 데이터 모든숫자들을 최소 값만큼 뺀다.
    numerator = data - np.min(data, 0)
    # 최대값과 최소 값의 차이(A)를 구한다
    denominator = np.max(data, 0) - np.min(data, 0)
    # 너무 큰 값이 나오지 않도록 나눈다
    return numerator / (denominator + 1e-7)

def lstm_cell():
    # LSTM셀을 생성한다.
    # num_units: 각 Cell 출력 크기
    # forget_bias: The bias added to forget gates.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    # cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.sigmoid)
    # cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=0.8, state_is_tuple=True, activation=tf.tanh)
    return cell

# 텐서플로우 플레이스홀더 생성
# 학습용/테스트용으로 X, Y를 생성한다
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)
predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)

# 몇개의 층으로 쌓인 Stacked RNNs 생성, 여기서는 1개층만
multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)

Y_pred = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_dim, activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE(Root Mean Square Error)
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions))) # 아래 코드와 같다
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1, 53):
    point_data = []
    with open('event/second_event_border/event_border' + str(i) + '.json') as raw_file:
        json_data = json.load(raw_file)
        json_data = json_data[1]["data"]

        temp_list = []
        for j, item in enumerate(json_data):
            #point_data.append([item["score"], (j + 1) * 30, boost_time, end_time])
            point_data.append(item["score"])

    temp1_data = MinMaxScaler(point_data)

    time_data = []

    for j, item in enumerate(temp1_data):
        # point_data.append([item["score"], (j + 1) * 30, boost_time, end_time])
        time_data.append((j + 1) * 30)


    temp2_data = MinMaxScaler(time_data)

    point_data = []
    for j, item in enumerate(temp1_data):
        point_data.append([item, temp2_data[j]])


    dataX = []
    dataY = []

    for j in range(0, len(point_data) - seq_length):
        _x = point_data[j: j + seq_length]
        _y = point_data[j + seq_length]  # 다음 나타날 주가(정답)
        if j is 0:
            print(_x, "->", _y)
        dataX.append(_x)
        dataY.append([_y[0]])

    """
    train_size = int(len(dataY) * 0.7)
    # 나머지(30%)를 테스트용 데이터로 사용
    test_size = len(dataY) - train_size
    
    
    trainX = np.array(dataX[0:train_size])
    trainY = np.array(dataY[0:train_size])
    
    # 데이터를 잘라 테스트용 데이터 생성
    testX = np.array(dataX[train_size:len(dataX)])
    testY = np.array(dataY[train_size:len(dataY)])
    """

    # 학습한다
    for epoch in range(epoch_num):
        _, step_loss = sess.run([train, loss], feed_dict={X: dataX, Y: dataY})
        print("[data: {} step: {} loss: {}".format(i, epoch, step_loss))

# 테스트한다
point_data = []
with open('event/event_border52.json') as raw_file:
    json_data = json.load(raw_file)
    json_data = json_data[1]["data"]

    temp_list = []
    for j, item in enumerate(json_data):
        #point_data.append([item["score"], (j + 1) * 30, boost_time, end_time])
        point_data.append(item["score"])

temp1_data = MinMaxScaler(point_data)

time_data = []

for j, item in enumerate(temp1_data):
    # point_data.append([item["score"], (j + 1) * 30, boost_time, end_time])
    time_data.append((j + 1) * 30)


temp2_data = MinMaxScaler(time_data)

point_data = []
for j, item in enumerate(temp1_data):
     point_data.append([item, temp2_data[j]])

#print(point_data)

dataX = []
dataY = []

for j in range(0, len(point_data) - seq_length):
    _x = point_data[j: j + seq_length]
    _y = point_data[j + seq_length]  # 다음 나타날 주가(정답)
    if j is 0:
        print(_x, "->", _y)
    dataX.append(_x)
    dataY.append([_y[0]])

test_predict = sess.run(Y_pred, feed_dict={X: dataX})

# 테스트용 데이터 기준으로 측정지표 rmse를 산출한다
rmse_val = sess.run(rmse, feed_dict={targets: dataY, predictions: test_predict})
print("rmse: ", rmse_val)

plt.plot(dataY, 'r')
plt.plot(test_predict, 'b')
plt.xlabel("Time Period")
plt.ylabel("Event point")
plt.show()

for z in range(0, 5):
    for i in range(1, 53):
        point_data = []
        with open('event/second_event_border/idol_num' + str(i) + '.json') as raw_file:
            json_data = json.load(raw_file)
            json_data = json_data[z]["data"]

            temp_list = []
            for j, item in enumerate(json_data):
                #point_data.append([item["score"], (j + 1) * 30, boost_time, end_time])
                point_data.append(item["score"])

        temp1_data = point_data

        time_data = []

        for j, item in enumerate(temp1_data):
            # point_data.append([item["score"], (j + 1) * 30, boost_time, end_time])
            time_data.append((j + 1) * 30)


        temp2_data = time_data

        point_data = []
        for j, item in enumerate(temp1_data):
             point_data.append([item, temp2_data[j]])

        #print(point_data)

        dataX = []
        dataY = []

        for j in range(0, len(point_data) - seq_length):
            _x = point_data[j: j + seq_length]
            _y = point_data[j + seq_length]  # 다음 나타날 주가(정답)
            if j is 0:
                print(_x, "->", _y)
            dataX.append(_x)
            dataY.append([_y[0]])

        plt.plot(dataY)

    plt.xlabel("Time Period")
    plt.ylabel("Event point")
    plt.show()