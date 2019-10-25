import pandas as pd
import numpy as np
import csv
from os import path
import MAIN.Basics as basics
import MAIN.Reinforcement as RL
import CONFIG as configuration
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from UTIL import FileIO
from STRATEGY.Cointegration import EGCointegration


save_model = True
restore_model = False

# Read config
config_path  = 'CONFIG\config_train.yml'
config_train = FileIO.read_yaml(config_path)

# Read prices
x = pd.read_csv('STATICS/FX/2019/EURJPY.csv')
y = pd.read_csv('STATICS/FX/2019/GBPJPY.csv')
divider = round(len(x) * 0.7)
x_divider = list(range(divider, len(x)))
y_divider = list(range(divider, len(y)))
x = x.iloc[x_divider, :]
y = y.iloc[y_divider, :]
x, y = EGCointegration.clean_data(x, y, 'date', 'close')

# Separate training and testing sets
train_pct = 0.7
train_len = round(len(x) * 0.7)
idx_train = list(range(0, train_len))
idx_test  = list(range(train_len, len(x)))
EG_Train = EGCointegration(x.iloc[idx_train, :], y.iloc[idx_train, :], 'date', 'close')
EG_Test  = EGCointegration(x.iloc[idx_test,  :], y.iloc[idx_test,  :], 'date', 'close')

# Create action space
n_hist    = list(np.arange(60, 601, 60))
n_forward = list(np.arange(120, 1201, 120))
trade_th  = list(np.arange(1, 5.1, 1))  # 1, 5.1, 1 # 5steps -- 1,  10.1, 2
stop_loss = list(np.arange(1, 2.1, 0.5))  # 1, 2.1, 0.5 # 3steps -- 2,  11.1, 3
cl        = list(np.arange(0.05, 0.11, 0.05))  # 0.05, 0.11, 0.05 # 2steps -- 0.3,  0.71, 0.2
actions   = {'n_hist':    n_hist,
             'n_forward': n_forward,
             'trade_th':  trade_th,
             'stop_loss': stop_loss,
             'cl':        cl}
n_action  = int(np.product([len(actions[key]) for key in actions.keys()]))

# Create state space
transaction_cost = [0.000]
states  = {'transaction_cost': transaction_cost}
n_state = len(states)

# Assign state and action spaces to config
config_train['StateSpaceState'] = states
config_train['ActionSpaceAction'] = actions

# Create and build network
one_hot  = {'one_hot': {'func_name':  'one_hot',
                        'input_arg':  'indices',
                         'layer_para': {'indices': None,
                                        'depth': n_state}}}
output_layer = {'final': {'func_name':  'fully_connected',
                          'input_arg':  'inputs',
                          'layer_para': {'inputs': None,
                                         'num_outputs': n_action,
                                         'biases_initializer': None,
                                         'activation_fn': tf.nn.relu,
                                         'weights_initializer': tf.ones_initializer()}}}

state_in = tf.placeholder(shape=[1], dtype=tf.int32)

N = basics.Network(state_in)
N.build_layers(one_hot)
N.add_layer_duplicates(output_layer, 1)

# Create learning object and perform training
RL_Train = RL.ContextualBandit(N, config_train, EG_Train)

sess = tf.Session()

RL_Train.process(sess, save=save_model, restore=restore_model)

if restore_model is False:
    # Extract training results
    action = RL_Train.recorder.record['NETWORK_ACTION']
    reward = RL_Train.recorder.record['ENGINE_REWARD']
    print(np.mean(reward))

    df1 = pd.DataFrame()
    df1['action'] = action
    df1['reward'] = reward
    mean_reward = df1.groupby('action').mean()
    sns.distplot(mean_reward)
    # Test by trading continuously
    [opt_action] = sess.run([RL_Train.output], feed_dict=RL_Train.feed_dict)
    opt_action = np.argmax(opt_action)
    action_dict = RL_Train.action_space.convert(opt_action, 'index_to_dict')
    folder_path = config_train['AgentModelSaverRestorePath']
    name = 'action_dict.csv'
    file_path = path.join(folder_path, name).replace('\\', '/')
    with open(file_path, 'w', newline='') as f:
        w = csv.DictWriter(f, action_dict.keys())
        w.writeheader()
        w.writerow(action_dict)

if restore_model is True:
    folder_path = config_train['AgentModelSaverRestorePath']
    name = 'action_dict.csv'
    file_path = path.join(folder_path, name).replace('\\', '/')
    with open(file_path, 'r') as f:
        action_dict_temp = []
        for i, row in enumerate(csv.DictReader(f)):
            row_ = FileIO.Record.transform(row)
        action_dict = dict(row_)
        print(action_dict)

indices = range(601, len(EG_Test.x) - 1200)

pnl = pd.DataFrame()
pnl['Time'] = EG_Test.timestamp
pnl['Trade_Profit'] = 0
pnl['Cost'] = 0
pnl['N_Trade'] = 0

import warnings
warnings.filterwarnings('ignore')
for i in indices:
    if i % 100 == 0:
        print(i)
    EG_Test.process(index=i, transaction_cost=0.000, **action_dict)
    trade_record = EG_Test.record
    if (trade_record is not None) and (len(trade_record) > 0):
        # print('value at {}'.format(i))
        trade_record = pd.DataFrame(trade_record)
        trade_cost   = trade_record.groupby('trade_time')['trade_cost'].sum()
        close_cost   = trade_record.groupby('close_time')['close_cost'].sum()
        profit       = trade_record.groupby('close_time')['profit'].sum()
        open_pos     = trade_record.groupby('trade_time')['long_short'].sum()
        close_pos    = trade_record.groupby('close_time')['long_short'].sum() * -1

        pnl['Cost'].loc[pnl['Time'].isin(trade_cost.index)] += trade_cost.values
        pnl['Cost'].loc[pnl['Time'].isin(close_cost.index)] += close_cost.values
        pnl['Trade_Profit'].loc[pnl['Time'].isin(close_cost.index)] += profit.values
        pnl['N_Trade'].loc[pnl['Time'].isin(trade_cost.index)] += open_pos.values
        pnl['N_Trade'].loc[pnl['Time'].isin(close_cost.index)] += close_pos.values

warnings.filterwarnings(action='once')


# Plot the testing result
pnl['PnL'] = (pnl['Trade_Profit'] - pnl['Cost']).cumsum()
plt.plot(pnl['PnL'])
plt.plot(pnl['N_Trade'])
plt.plot(pnl['Time'], pnl['PnL'])

plt.plot(pnl['Time'], pnl['N_Trade'])

plt.show()

sess.close()
