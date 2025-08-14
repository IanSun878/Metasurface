import os
import glob
from _pytest.monkeypatch import K
import numpy as np
import matplotlib.pyplot as plt

# ===== 可調整區塊 =====
x_label = "Wavelength ( nm )"
y_label = "Gain ( dB )"
peak_x_range_str = "檔名+-0.1"
peak_y_range = (-2, 0)
title_fontsize = 14    # 標題字體大小
label_fontsize = 14    # X 與 Y 標籤字體大小
tick_fontsize = 10     # X 與 Y 刻度字體大小
vertical_lines = [1270, 1280, 1290, 1300, 1310, 1320, 1330, 1340]
save_new_txt = True  # 若不儲存新檔請設為 False
# =====================

# 刪除上次圖片與新檔案
for file in glob.glob("*_new.txt") + glob.glob("*_Peak.png") + glob.glob("*_nm.png"):
    os.remove(file)

# 處理所有 txt 檔案
for filepath in glob.glob("*.txt"):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 忽略文字與空白列
    data = []
    for line in lines:
        line = line.strip()
        if line == "" or not any(c.isdigit() for c in line):  # 空白或無數字跳過
            continue
        try:
            x, y = map(float, line.split(','))
            data.append([x, y])
        except:
            continue

    data = np.array(data)

    x=data[:,0]
    y=data[:, 1]

    t=0
    k=0
    x_new=[]
    y_new=[]
    for k in range(len(x)-7):
        if x[k]==x[k+1]:
            t+=1;
        elif x[k]!=x[k+1]:
            t+=1
            offsets = [0.01/t * i for i in reversed(range(t))];
            for i in range(t):
                x_new.append(x[k]+offsets[i])
                y_new.append(y[k+1-t+i])
            t=0;


     # 產生新檔名基底（例如 '1270'）
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # 儲存新檔案 (_new.txt)
    new_filename = os.path.splitext(filepath)[0] + "_new.txt"
    if save_new_txt:
        with open(new_filename, 'w') as f:
            for x, y in zip(x_new, y_new):
                f.write(f"{x:.6f}, {y:.4f}\n")  # 儲存範圍：全部
    print(f"{new_filename} 已儲存。")

    # 畫圖：完整範圍
    plt.figure()
    plt.plot(x_new, y_new, linewidth=1)
    # 標題：顯示「1270nm」
    plt.title(f"{base_name}nm", fontsize=title_fontsize)
    for vline in vertical_lines:
        plt.axvline(x=vline, color='red', linestyle='--', linewidth=0.8)
    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel(y_label, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    full_fig = os.path.splitext(filepath)[0] + "_nm.png"
    plt.savefig(full_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{full_fig} 圖已儲存。")

    # 畫圖：Peak 區域圖（使用散點圖）
    center = float(os.path.splitext(filepath)[0])  # 假設檔名為中心波長
    peak_x_min = center - 0.1
    peak_x_max = center + 0.1

    plt.figure()
    plt.scatter(x_new, y_new, s=5)
    # 標題：顯示「1270nm Peak」
    plt.title(f"{base_name}nm Peak", fontsize=title_fontsize)
    plt.xlim([peak_x_min, peak_x_max])
    plt.ylim(peak_y_range)
    for vline in vertical_lines:
        plt.axvline(x=vline, color='red', linestyle='--', linewidth=0.8)
    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel(y_label, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    peak_fig = os.path.splitext(filepath)[0] + "_Peak.png"
    plt.savefig(peak_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{peak_fig} 圖已儲存。")
