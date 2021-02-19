import model
import data_create
import testdata_create
import vid4_datacreate
import argparse
import os
import cv2
import glob
import keras
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    train_height = 36
    train_width = 36
    test_height = 720
    test_width = 1280

    frame_num = 3
    cut_num = 10

    train_dataset_num = 5000
    test_dataset_num = 100

    train_movie_path = "../rsde/train_sharp"
    test_movie_path = "../rsde/val_sharp"

    BATSH_SIZE = 128
    EPOCHS = 1000

    os.makedirs("model", exist_ok = True)
    epo_path = "model" + "/"
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_model_a', help='datacreate, evaluate')

    args = parser.parse_args()

    if args.mode == 'datacreate':
        train_x, train_y = data_create.save_frame(train_movie_path,   #切り取る動画のpath
                                                 train_dataset_num,  #データセットの生成数
                                                 frame_num, 
                                                 cut_num,
                                                 train_height, #保存サイズ
                                                 train_width)   #倍率
        path = "train_data_list"
        np.savez(path, train_x, train_y)

    elif args.mode == 'test_datacreate':
        test_x, test_y, low_data, cr_data = testdata_create.save_frame(test_movie_path,   #切り取る動画のpath
                                                test_dataset_num,  #データセットの生成数
                                                frame_num,   #何フレーム抽出するか
                                                test_height, #保存サイズ
                                                test_width)   #倍率

        path = "test_data_list"
        np.savez(path, test_x, test_y, low_data, cr_data)

    elif args.mode == "train_model_a":
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"]
        train_y = npz["arr_1"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)
        train_x /= 255
        train_y /= 255

        model_a = model.vsrnet_model_a() 
        model_a.compile(loss = "mean_squared_error",
                        optimizer = opt,
                        metrics = [psnr])

        model_a.fit({"tminus1":train_x[0], "t":train_x[1], "tplus1":train_x[2]},
                    train_y,
                    epochs = EPOCHS,
                    verbose = 2,
                    batch_size = BATSH_SIZE)

        model_a.save(epo_path + "model_a.h5")
        plt.plot(model_a.history.history['psnr'], label="training")
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig(epo_path + "vsrnet_model_a_plot.png")

    elif args.mode == "train_model_b":
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"]
        train_y = npz["arr_1"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)
        train_x /= 255
        train_y /= 255

        model_b = model.vsrnet_model_b() 
        model_b.compile(loss = "mean_squared_error",
                        optimizer = opt,
                        metrics = [psnr])

        model_b.fit({"tminus1":train_x[0], "t":train_x[1], "tplus1":train_x[2]},
                    train_y,
                    epochs = EPOCHS,
                    verbose = 2,
                    batch_size = BATSH_SIZE)

        model_b.save(epo_path + "model_b.h5")

        plt.plot(model_b.history.history['psnr'], label="training")
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig(epo_path + "vsrnet_model_b_plot.png")

    elif args.mode == "train_model_c":
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"]
        train_y = npz["arr_1"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)
        train_x /= 255
        train_y /= 255

        model_c = model.vsrnet_model_c() 
        model_c.compile(loss = "mean_squared_error",
                        optimizer = opt,
                        metrics = [psnr])

        model_c.fit({"tminus1":train_x[0], "t":train_x[1], "tplus1":train_x[2]},
                    train_y,
                    epochs = EPOCHS,
                    verbose = 2,
                    batch_size = BATSH_SIZE)
        model_c.save(epo_path + "model_c.h5")

        plt.plot(model_c.history.history['psnr'], label="training")
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig(epo_path + "vsrnet_model_c_plot.png")

    elif args.mode == "evaluate":
        path = "./test_data_list"

        cube_path = "result"
        os.makedirs(cube_path, exist_ok = True)

        npz = np.load("test_data_list.npz")

        test_x = npz["arr_0"]
        test_y = npz["arr_1"]
        low_list = npz["arr_2"]
        color_list = npz["arr_3"]

        test_x = tf.convert_to_tensor(test_x, np.float32)
        test_y = tf.convert_to_tensor(test_y, np.float32)
        test_x /= 255
        test_y /= 255

        lists = ["a", "b", "c"]
            
        for mark in lists:
            path = "model/model_"+ mark
            exp = ".h5"

            new_path = path + exp
            if os.path.exists(new_path):
                new_model = tf.keras.models.load_model(new_path, custom_objects={'psnr':psnr})
                pred = new_model.predict({"tminus1":test_x[0], "t":test_x[1], "tplus1":test_x[2]},
                batch_size = 1)

                path = cube_path + "/" + "model_" + mark
                os.makedirs(path, exist_ok = True)
                path = path + "/"

                ps_pred_ave = 0
                ps_bicubic_ave = 0
                for p in range(len(test_y)):
                    pred[p][pred[p] > 1] = 1
                    pred[p][pred[p] < 0] = 0
                    ps_pred = psnr(tf.reshape(test_y[p], [test_height, test_width, 1]), pred[p])
                    ps_bicubic = psnr(tf.reshape(test_y[p], [test_height, test_width, 1]), tf.reshape(test_x[1][p], [test_height, test_width, 1]))
                    
                    ps_pred_ave += ps_pred
                    ps_bicubic_ave += ps_bicubic

                    if (ps_pred - ps_bicubic) > 2.3:

                        before_res = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_x[frame_num // 2][p], [test_height, test_width]))
                        low_color_list = low_list[p]
                        low_color_list[:, :, 0] = before_res[:,:,0] * 255
                        fra = cv2.cvtColor(low_color_list, cv2.COLOR_YCrCb2BGR)
                        cv2.imwrite(path + "low_" + mark + "_" +  str(p) + ".jpg", fra)

                        fra = cv2.cvtColor(color_list[p], cv2.COLOR_YCrCb2BGR)
                        cv2.imwrite(path + "high_" + mark + "_" +  str(p) + ".jpg", fra)

                        before_res = tf.keras.preprocessing.image.img_to_array(tf.reshape(pred[p], [test_height, test_width]))
                        pred_color_list = low_list[p]
                        pred_color_list[:, :, 0] = before_res[:,:,0] * 255
                        fra = cv2.cvtColor(pred_color_list, cv2.COLOR_YCrCb2BGR)
                        cv2.imwrite(path + "pred_" + mark + "_" +  str(p) + ".jpg", fra)

                        print("num:{}".format(p))
                        print("psnr_pred:{}".format(ps_pred))
                        print("psnr_bicubic:{}".format(ps_bicubic))
                        # print(ps_pred - ps_bicubic)

                print("psnr_pred_average:{}".format(ps_pred_ave / len(test_y)))
                print("psnr_bicubic_average:{}".format(ps_bicubic_ave / len(test_y)))

  
 