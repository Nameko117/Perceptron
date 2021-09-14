# -*- coding: utf-8 -*-


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as pp
import numpy as np
import random

# constant
FILENAME = ["perceptron1", "perceptron2", "2Ccircle1", "2Circle1", "2Circle2", "2CloseS", "2CloseS2", "2CloseS3", "2cring", "2CS", "2Hcircle1", "2ring"]
COLOR = ['c', 'darkorange', 'm', 'lime', 'r']


# 輸出參考
def find_two_output(d):
    output = [d[0]]
    for i in d:
        if i != output[0]:
            output.append(i)
            break
    return output

# 畫圖
def draw(x, d, w):
    # 畫點
    x1 = []
    x2 = []
    color = []
    for i in range(len(x)):
        x1.append(x[i][1])
        x2.append(x[i][2])
        color.append(COLOR[d[i]])
    pp.scatter(x1, x2, s=10, marker='o', color=color)
    
    # 畫線
    if len(x1) == 1:
        x1.append(x1[0]+1)
        x1[0] -= 1
        x2.append(x2[0]+1)
        x2[0] -= 1
    if w[2] == 0:
        dst = max(x2) - min(x2)
        line_y = np.linspace(min(x2)-0.1*dst, max(x2)+0.1*dst)
        line_x = len(line_y)*[w[0]/w[1]]
    elif w[1] == 0:
        dst = max(x1) - min(x1)
        line_x = np.linspace(min(x1)-0.1*dst, max(x1)+0.1*dst)
        line_y = len(line_x) * [w[0]/w[2]]
    else:
        dst = max(x1) - min(x1)
        line_x = np.linspace(min(x1)-0.1*dst, max(x1)+0.1*dst)
        line_y = (w[0]-w[1]*line_x)/w[2]
    pp.plot(line_x, line_y, 'k')
        
    

# 讀入資料
def read_file(file_name):
    f = open("NN_HW1_DataSet/" + file_name + ".txt")
    x = []
    d = []
    for line in f:
        line = line.replace("\n", "")
        tmp = line.split(" ")
        x.append([float(tmp[0]), float(tmp[1])])
        d.append(int(tmp[2]))
    f.close()
    return [x, d]


def train(x, d, rate, condition):
    # 初始化
    w = [-1, 0, 1]
    OUTPUT = find_two_output(d)
    best_correct_rate = 0
    best_w = []
    # 調整輸入
    for i in range(len(x)):
        x[i] = [-1] + x[i]
    # 訓練
    for i in range(condition):
        for j in range(len(x)):
            result = w[0]*x[j][0] + w[1]*x[j][1] + w[2]*x[j][2]
            if result <= 0:
                output = OUTPUT[0]
                if output != d[j]:
                    for k in range(3): w[k] += rate*x[j][k]
            else:
                output = OUTPUT[1]
                if output != d[j]:
                    for k in range(3): w[k] -= rate*x[j][k]
        # 算正確率
        train_correct = 0
        for j in range(len(x)):
            result = w[0]*x[j][0] + w[1]*x[j][1] + w[2]*x[j][2]
            if result <= 0:
                output = OUTPUT[0]
            else:
                output = OUTPUT[1]
            if output == d[j]:
                train_correct += 1
        correct_rate = train_correct/len(x)
        if correct_rate > best_correct_rate:
            best_correct_rate = correct_rate
            best_w = w
        print("echo {} w = {} correct_rate = {}".format(i+1, w, correct_rate))
        if correct_rate == 1:
            break
    return [best_w, best_correct_rate, OUTPUT]

def test(x, d, w, OUTPUT):
    # 初始化
    test_correct = 0
    # 調整輸入
    for i in range(len(x)):
        x[i] = [-1] + x[i]
    # 測試正確率
    for i in range(len(x)):
        result = w[0]*x[i][0] + w[1]*x[i][1] + w[2]*x[i][2]
        if result < 0:
            output = OUTPUT[0]
        else:
            output = OUTPUT[1]
        if output == d[i]:
            test_correct += 1
    return test_correct/len(x)

# 按下輸入按鈕
def start():
    # 關閉前次訓練結果
    global pic_num
    if pic_num > 0: pp.close()
    pic_num += 1
    
    # 輸入
    rate = float(rate_entry.get())
    condition = int(condition_entry.get())
    file_name = data_combo.get()
    
    # 資料分 2:1 做 訓練 & 測試
    [x, d] = read_file(file_name)
    n = int(len(x)/3)
    test_x = []
    test_d = []
    for i in range(n):
        rnd = random.randint(0, len(x)-1)
        test_x.append(x.pop(rnd))
        test_d.append(d.pop(rnd))

    # 訓練
    [w, train_correct_rate, OUTPUT] = train(x, d, rate, condition)
    pp.subplot(211)
    draw(x, d, w)

    # 測試
    test_correct_rate = test(test_x, test_d, w, OUTPUT)
    pp.subplot(212)
    draw(test_x, test_d, w)

    # 輸出（訓練辨識率、測試辨識率、鍵結值
    for i in range(3): w[i] = round(w[i], 3)
    result = '{}\n訓練辨識率 = {:.3f}\n測試辨識率 = {:.3f}\n鍵結值 = {}'.format(file_name, train_correct_rate, test_correct_rate, w)
    result_label.configure(text=result)
    
    pp.show()


# 開新視窗
window = tk.Tk()
# 設計視窗
window.title('107502508')
window.geometry('300x200')
window.configure(background='white')
# 標題
header_label = tk.Label(window, text='感知機類神經網路')
header_label.pack()

# 學習率 rate 群組
rate_frame = tk.Frame(window)
rate_frame.pack(side=tk.TOP)
rate_label = tk.Label(rate_frame, text='學習率：')
rate_label.pack(side=tk.LEFT)
rate_entry = tk.Entry(rate_frame)
rate_entry.pack(side=tk.LEFT)
# 收斂條件 condition 群組
condition_frame = tk.Frame(window)
condition_frame.pack(side=tk.TOP)
condition_label = tk.Label(condition_frame, text='收斂條件：')
condition_label.pack(side=tk.LEFT)
condition_entry = tk.Entry(condition_frame)
condition_entry.pack(side=tk.LEFT)
# 輸入資料 data 群組
data_frame = tk.Frame(window)
data_frame.pack(side=tk.TOP)
data_label = tk.Label(data_frame, text='輸入檔案：')
data_label.pack(side=tk.LEFT)
data_combo = ttk.Combobox(data_frame, values=FILENAME, state="readonly")
data_combo.current(0)
data_combo.pack(side=tk.LEFT)

# 輸入按鈕
input_btn = tk.Button(window, text='輸入', command=start)
input_btn.pack()

# 輸出
result_label = tk.Label(window)
result_label.pack()

# 畫圖相關
pp.ion()
pic_num = 0

# 執行視窗
window.mainloop()



