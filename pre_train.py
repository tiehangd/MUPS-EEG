##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Generate commands for pre-train phase. """
import os

def run_exp(lr=0.05, gamma=0.5, step_size=20):
    max_epoch = 10
    shot = 20
    query = 10
    way = 4
    gpu = 2
    base_lr = 0.005
    
    the_command = 'python3 main.py' \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --pre_step_size=' + str(step_size) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --pre_lr=' + str(lr) \
        + ' --phase=pre_train' 

    os.system(the_command)

run_exp(lr=0.05, gamma=0.5, step_size=20) 


