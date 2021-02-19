from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Add, Concatenate, Dropout

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

def vsrnet_model_a():
    tminus1_input = Input(shape = (None, None, 1), name = "tminus1")
    t_input = Input(shape = (None, None, 1), name = "t")
    tplus1_input = Input(shape = (None, None, 1), name = "tplus1")

    input_shape = Concatenate()([tminus1_input, t_input, tplus1_input])

    conv2d_0 = Conv2D(filters = 64,
                    kernel_size = (9, 9),
                    padding = "same",
                    activation = "relu"
                    # kernel_regularizer=keras.regularizers.l2(0.0005),
                    )(input_shape)

    conv2d_1 = Conv2D(filters = 32,
                    kernel_size = (5, 5),
                    # kernel_regularizer=keras.regularizers.l2(0.0005),
                    padding = "same",
                    activation = "relu"
                    )(conv2d_0)

    conv2d_2 =  Conv2D(filters = 1,
                    kernel_size = (5, 5),
                    # kernel_regularizer=keras.regularizers.l2(0.0005),
                    padding = "same",
                    )(conv2d_1)

    model = Model(inputs = [tminus1_input, t_input, tplus1_input], outputs = [conv2d_2])

    model.summary()

    return model

def vsrnet_model_b():
    tminus1_input = Input(shape = (None, None, 1), name = "tminus1")
    t_input = Input(shape = (None, None, 1), name = "t")
    tplus1_input = Input(shape = (None, None, 1), name = "tplus1")

    conv2d_0_tminus1 = Conv2D(filters = 64,
                            kernel_size = (9, 9),
                            # kernel_regularizer=keras.regularizers.l2(0.0005),
                            padding = "same",
                            activation = "relu",
                            )(tminus1_input)


    conv2d_0_t = Conv2D(filters = 64,
                        kernel_size = (9, 9),
                        # kernel_regularizer=keras.regularizers.l2(0.0005),
                        padding = "same",
                        activation = "relu",
                        )(t_input)


    conv2d_0_tplus1 = Conv2D(filters = 64,
                            kernel_size = (9, 9),
                            # kernel_regularizer=keras.regularizers.l2(0.0005),
                            padding = "same",
                            activation = "relu",
                            )(tplus1_input)


    new_input = Concatenate()([conv2d_0_tminus1, conv2d_0_t, conv2d_0_tplus1])

    conv2d_1 = Conv2D(filters = 32,
                    kernel_size = (5, 5),
                    # kernel_regularizer=keras.regularizers.l2(0.0005),
                    padding = "same",
                    activation = "relu"
                    )(new_input)

    conv2d_2 =  Conv2D(filters = 1,
                    kernel_size = (5, 5),
                    # kernel_regularizer=keras.regularizers.l2(0.0005),
                    padding = "same",
                    )(conv2d_1)

    model = Model(inputs = [tminus1_input, t_input, tplus1_input], outputs = [conv2d_2])

    model.summary()

    return model

def vsrnet_model_c():
    tminus1_input = Input(shape = (None, None, 1), name = "tminus1")
    t_input = Input(shape = (None, None, 1), name = "t")
    tplus1_input = Input(shape = (None, None, 1), name = "tplus1")

    conv2d_0_tminus1 = Conv2D(filters = 64,
                            kernel_size = (9, 9),
                            # kernel_regularizer=keras.regularizers.l2(0.0005),
                            padding = "same",
                            activation = "relu",
                            )(tminus1_input)

    conv2d_1_tminus1 = Conv2D(filters = 32,
                            kernel_size = (5, 5),
                            # kernel_regularizer=keras.regularizers.l2(0.0005),
                            padding = "same",
                            activation = "relu"
                            )(conv2d_0_tminus1)


    conv2d_0_t = Conv2D(filters = 64,
                        kernel_size = (9, 9),
                        # kernel_regularizer=keras.regularizers.l2(0.0005),
                        padding = "same",
                        activation = "relu",
                        )(t_input)
                     
    conv2d_1_t = Conv2D(filters = 32,
                            kernel_size = (5, 5),
                            # kernel_regularizer=keras.regularizers.l2(0.0005),
                            padding = "same",
                            activation = "relu"
                            )(conv2d_0_t)


    conv2d_0_tplus1 = Conv2D(filters = 64,
                            kernel_size = (9, 9),
                            # kernel_regularizer=keras.regularizers.l2(0.0005),
                            padding = "same",
                            activation = "relu",
                            )(tplus1_input)
                      
    conv2d_1_tplus1 = Conv2D(filters = 32,
                            kernel_size = (5, 5),
                            # kernel_regularizer=keras.regularizers.l2(0.0005),
                            padding = "same",
                            activation = "relu"
                            )(conv2d_0_tplus1)


    new_input = Concatenate()([conv2d_1_tminus1, conv2d_1_t, conv2d_1_tplus1])

    conv2d_2 =  Conv2D(filters = 1,
                    kernel_size = (5, 5),
                    # kernel_regularizer=keras.regularizers.l2(0.0005),
                    padding = "same",
                    )(new_input)

    model = Model(inputs = [tminus1_input, t_input, tplus1_input], outputs = [conv2d_2])

    model.summary()

    return model

def srcnn():
    input_shape = Input((None, None, 1))

    conv2d_0 = Conv2D(filters = 64,
                        kernel_size = (9, 9),
                        padding = "same",
                        activation = "relu",
                        kernel_regularizer=keras.regularizers.l2(0.0005)
                        )(input_shape)
    conv2d_1 = Conv2D(filters = 32,
                        kernel_size = (1, 1),
                        padding = "same",
                        activation = "relu",
                        kernel_regularizer=keras.regularizers.l2(0.0005)
                        )(conv2d_0)
    conv2d_2 = Conv2D(filters = 1,
                        kernel_size = (5, 5),
                        padding = "same",
                        kernel_regularizer=keras.regularizers.l2(0.0005)
                        )(conv2d_1)

    model = Model(inputs = input_shape, outputs = [conv2d_2])

    model.summary()

    return model