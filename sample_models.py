from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    bn_rnn = BatchNormalization(name='deep_layer_0')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name='distributed')(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='deep_layer_1')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim), name = 'distributed')(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    input_data_copy = input_data
    for charac in range(recur_layers):
        input_data_copy = BatchNormalization(name='bn_rnn_'+str(charac+1))(GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn_'+str(charac+1))(input_data_copy))
        
    time_dense = TimeDistributed(Dense(output_dim), name = 'distributed')(input_data_copy)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn = Bidirectional(GRU(units, activation="relu", return_sequences=True, implementation=2), name="deep_layer_2")(input_data)
    time_dense = TimeDistributed(Dense(output_dim), name='distributed')(bidir_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    deep_layer_5 = TimeDistributed(Dense(output_dim), name='deep_layer_4')(Dropout(0.3, name='deep_layer_3')(BatchNormalization(name='deep_layer_2')(Bidirectional(GRU(units, activation="relu", return_sequences=True, implementation=2), name="deep_layer")(BatchNormalization(name='after_norm')(Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='cnn_1')(input_data))))))
    y_pred = Activation('softmax', name='softmax')(deep_layer_5)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda k: cnn_output_length(k, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
