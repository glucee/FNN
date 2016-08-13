import os
import argparse

DEFAULT_EPI = 10000000
DEFAULT_STEPS = 500
DEFAULT_MEM_WIDTH = 100000
DEFAULT_INITIAL_EPSILON = 1
DEFAULT_FINAL_EPSILON = 0.1
DEFAULT_GAMMA = 0.95
DEFAULT_MINI_BATCH = 16
DEFAULT_OBSERVATION = 10000

DEFAULT_LEARNING_RATE = 0.00005
DEFAULT_REGULARIZATION = 0.001
DEFAULT_HIDDEN_LAYER = 300
DEFAULT_HIDDEN_LAYER_NUM = 2

DEFAULT_SAVER_PATH = "dqn_tmp/"
DEFAULT_LOG_PATH = "dqn_log/log1/data1"
DEFAULT_SAVING_FREQUENCY = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-episodes', default = DEFAULT_EPI, type = int)
    parser.add_argument('-steps', default = DEFAULT_STEPS, type = int )
    parser.add_argument('-MemSize', default = DEFAULT_MEM_WIDTH, type = int )
    parser.add_argument('-initial_epsilon', default = DEFAULT_INITIAL_EPSILON, type = int )
    parser.add_argument('-final_epsilon', default = DEFAULT_FINAL_EPSILON, type = int )
    parser.add_argument('-gamma', default = DEFAULT_GAMMA, type = int )
    parser.add_argument('-BatchSize', default = DEFAULT_MINI_BATCH, type = int )
    parser.add_argument('-reg', default = DEFAULT_REGULARIZATION, type = int )
    parser.add_argument('-LearningRate', default = DEFAULT_LEARNING_RATE, type = int )
    parser.add_argument('-HiddenLayerNum', default = DEFAULT_HIDDEN_LAYER_NUM, type = int )
    parser.add_argument('-HiddenLayerSize', default = DEFAULT_HIDDEN_LAYER, type = int )
    parser.add_argument('-observation', default = DEFAULT_OBSERVATION, type = int )
    parser.add_argument('-saver_path', default = DEFAULT_SAVER_PATH, type = str )
    parser.add_argument('-saving_rate', default = DEFAULT_SAVING_FREQUENCY, type = int)
    parser.add_argument('-log_path', default = DEFAULT_LOG_PATH, type = str)

    args = parser.parse_args()
    training_params = {
        'saver_path' : args.saver_path,
        'saving_rate' : args.saving_rate
    }
    network_params = {
        'HiddenLayerSize': args.HiddenLayerSize,
        'reg':args.reg,
        'LearningRate': args.LearningRate,
        'HiddenLayerNum': args.HiddenLayerNum,
        'mini_batch_size': args.BatchSize,
        'log_file_path': args.log_path
        }

    return agent_params, dqn_params, network_params
