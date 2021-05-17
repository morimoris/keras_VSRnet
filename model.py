from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Concatenate

class VSRnet():
    def __init__(self):
        self.input_channels = 1
        self.input_LR_num = 3

    def model_a(self):
        #input video frames
        input_list = self.input_LR_num * [None]
        for img in range(self.input_LR_num): 
            input_list[img] = Input(shape = (None, None, self.input_channels), name = "input_" + str((img)))

        new_input = Concatenate()(input_list)

        conv2d_0 = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(new_input)
        conv2d_1 = Conv2D(filters = 32, kernel_size = (5, 5), padding = "same", activation = "relu")(conv2d_0)
        conv2d_2 =  Conv2D(filters = self.input_channels, kernel_size = (5, 5), padding = "same")(conv2d_1)

        model = Model(inputs = input_list, outputs = conv2d_2)
        model.summary()

        return model

    def model_b(self):
        #input video frames
        input_list = self.input_LR_num * [None]
        for img in range(self.input_LR_num): 
            input_list[img] = Input(shape = (None, None, self.input_channels), name = "input_" + str((img)))

        #convolution each images
        conv2d_0_tminus1 = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[0])
        conv2d_0_t = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[1])
        conv2d_0_tplus1 = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[2])

        #concatenate each results
        new_input = Concatenate()([conv2d_0_tminus1, conv2d_0_t, conv2d_0_tplus1])

        #convolution
        conv2d_1 = Conv2D(filters = 32, kernel_size = (5, 5), padding = "same", activation = "relu")(new_input)
        conv2d_2 =  Conv2D(filters = self.input_channels, kernel_size = (5, 5), padding = "same")(conv2d_1)

        model = Model(inputs = input_list, outputs = conv2d_2)
        model.summary()

        return model

    def model_c(self):
        #input video frames
        input_list = self.input_LR_num * [None]
        for img in range(self.input_LR_num): 
            input_list[img] = Input(shape = (None, None, self.input_channels), name = "input_" + str((img)))

       #convolution each images
        conv2d_0_tminus1 = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[0])
        conv2d_0_t = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[1])
        conv2d_0_tplus1 = Conv2D(filters = 64, kernel_size = (9, 9), padding = "same", activation = "relu")(input_list[2])

        conv2d_1_tminus1 = Conv2D(filters = 32, kernel_size = (5, 5), padding = "same", activation = "relu")(conv2d_0_tminus1)
        conv2d_1_t = Conv2D(filters = 32, kernel_size = (5, 5), padding = "same", activation = "relu")(conv2d_0_t)
        conv2d_1_tplus1 = Conv2D(filters = 32, kernel_size = (5, 5), padding = "same", activation = "relu")(conv2d_0_tplus1)

        new_input = Concatenate()([conv2d_1_tminus1, conv2d_1_t, conv2d_1_tplus1])

        conv2d_2 =  Conv2D(filters = self.input_channels, kernel_size = (5, 5), padding = "same")(new_input)

        model = Model(inputs = input_list, outputs = conv2d_2)

        model.summary()

        return model

