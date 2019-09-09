from keras.layers import Conv2D, AveragePooling2D, Concatenate

input_layer = None
N_FILTER = 128


# region model
branch_a = Conv2D(N_FILTER, 1, activation='relu', strides=1)(input_layer)

branch_b_1 = Conv2D(N_FILTER, 1, activation='relu')(input_layer)
branch_b_2 = Conv2D(N_FILTER, 3, activation='relu', strides=2)(branch_b_1)

branch_c_1 = AveragePooling2D(3, strides=2)(input_layer)
branch_c_2 = Conv2D(N_FILTER, 3, activation='relu')(branch_c_1)

branch_d_1 = Conv2D(N_FILTER, 1, activation='relu')(input_layer)
branch_d_2 = Conv2D(N_FILTER, 3, activation='relu')(branch_d_1)
branch_d_3 = Conv2D(N_FILTER, 3, activation='relu', strides=2)(branch_d_2)

concatenated = Concatenate([branch_a, branch_b_2, branch_c_2, branch_d_3], axis=-1)
# endregion


# region add a residual connection on branch d_1
residual = Conv2D(N_FILTER, 1, strides=2)(branch_d_1)
concatenated_with_residual = Concatenate([concatenated, residual], axis=-1)
# endregion
