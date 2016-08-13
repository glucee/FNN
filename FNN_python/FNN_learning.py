import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from FNN_params import parse_args
import FNN_model


OUTPUT_LAYER_SIZE = 2
INPUT_LAYER_SIZE = 75
BatchSize = 8
training_epoch = 10

def import_data(file_name):
    mat_contents = sio.loadmat(file_name)
    """
    Extract two matrices from mat file:
    1. Input data
    2. Output data
    These are returned as numpy arrays
    """
    Input_data  = mat_contents['input']
    Output_data  = mat_contents['output']
    return Input_data, Output_data

    # return NumberOfInputs, input_data, output_data


def save_nn_model(FNN, SAVING_PATH):
    FNN.save_model(SAVING_PATH+ 'model.ckpt')

def restore_nn_model(FNN, SAVING_PATH):
    ckpt = tf.train.get_checkpoint_state(SAVING_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        FNN.restore_model(ckpt.model_checkpoint_path)
        FNN.init_t()
        # print('length of mem:{}'.format(len(self.memory)))
        # print('weight averg {}'.format(length(self.memory)))
        print('saved model found and restored')
    else:
        print('No saved model, creating new one...')

def write_to_log(FNN, t):
    FNN.train_writer.add_summary(FNN.summary,t)

def train():
    """load the parameters"""
    training_params, network_params = parse_args()

    SAVING_PATH = training_params['saver_path']
    SAVING_RATE = training_params['saving_rate']
    input_data, output_data = import_data('data.mat')
    print (np.shape(output_data))
    FNN = FNN_model.FNN(OUTPUT_LAYER_SIZE, INPUT_LAYER_SIZE, network_params)
    restore_nn_model(FNN, SAVING_PATH)
    for epoch in range(training_epoch):
        avg_cost = 0
        NumberOfInputs, _ = input_data.shape
        for i in range(NumberOfInputs):
            FNN.train_step(input_data,output_data)
        # total_batch = int(NumberOfInputs / BatchSize)
        # for i in range(total_batch):
        #     Input_batch = input_data[i*BatchSize:(i+1)]
        #     Output_batch = output_data
        #     FNN.train_step(Input_batch,Output_batch)
    #     if epoch % SAVING_RATE == 0:
    #         save_nn_model(FNN, SAVING_PATH)
    #

if __name__=='__main__':
    train()
