import cv2
import os
import random
import glob

import numpy as np
import tensorflow as tf

#任意のフレーム数を切り出すプログラム
def save_frame(video_path,   #切り取る動画が入ったファイルのpath
               data_number,  #データセットの生成数
               stop_frame,   #何フレーム抽出するか   
               cut_height,#保存サイズ
               cut_width,
               ext='jpg'):
    if stop_frame % 2 == 0:
        return

    #データセットのリストを生成
    low_data_list = [[] for _ in range(stop_frame)]
    high_data_list = []
    color_data_list = []
    low_color_data_list  = []

    video_path = video_path + "/*"
    files = glob.glob(video_path)

    ram_mag = 2
    num = 0
    H, W = 720, 1280

    while num < data_number:
        file_num = random.randint(0, len(files)-1)
        photo_files = glob.glob(files[file_num] + "/*")
        photo_num = random.randint(0, len(photo_files) - stop_frame)

        ram_h = random.randint(0, H - cut_height)
        ram_w = random.randint(0, W-cut_width)
        check_list = [[] for _ in range(stop_frame)]
    
        for op in range(stop_frame):
            img = cv2.imread(photo_files[photo_num + op])

            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            gray = color_img[:, :, 0]

            if op == (stop_frame // 2):
                cut_img = gray[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
                color_high_img = color_img[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]

                color1 = cv2.resize(color_img , (int(W * (1 / ram_mag)), int(H * (1 / ram_mag))), interpolation=cv2.INTER_CUBIC)
                color2 = cv2.resize(color1 , (int(W), int(H)), interpolation=cv2.INTER_CUBIC)
                color2 = color2[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]

            img1 = cv2.resize(gray , (int(W * (1 / ram_mag)), int(H * (1 / ram_mag))), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img1 , (int(W), int(H)), interpolation=cv2.INTER_CUBIC)
            img3 = img2[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
            check_list[op].append(img3)
                
        var_list = []
        for pp in range(stop_frame-1):
            check_1 = np.array(check_list[pp]).flatten()
            check_2 = np.array(check_list[pp + 1]).flatten()
            minus = check_1 - check_2
            variance = np.var(minus)
            var_list.append(variance)
                
        if min(var_list) > 0.0035:
            for pp in range(stop_frame):
                low_data_list[pp].append(np.array(check_list[pp]).reshape((cut_height, cut_width)))
                
            high_data_list.append(cut_img)
            color_data_list.append(color_high_img)
            low_color_data_list.append(color2)
            num += 1

            # print(var_list)


    return low_data_list, high_data_list, low_color_data_list, color_data_list