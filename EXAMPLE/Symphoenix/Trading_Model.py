from datetime import *
from dateutil import parser
import multiprocessing as mp
import oandapyV20
from oandapyV20 import API
from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails
from oandapyV20.contrib.requests import TrailingStopLossOrderRequest
from oandapyV20.endpoints.pricing import PricingInfo
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
import telebot
import pandas as pd
import numpy as np
import csv
from os import path
import MAIN.Basics as basics
import MAIN.Reinforcement as RL
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from UTIL import FileIO
from STRATEGY.Cointegration import EGCointegration

# Bot Parameters
save_model = False
restore_model = True
spread_limit = 3.5
Multi_Threading = True
Tgr_Verbose = True

# Read config
config_path    = 'CONFIG\config_train.yml'
config_train   = FileIO.read_yaml(config_path)
config_path    = 'CONFIG\config_trading.yml'
config_trading = FileIO.read_yaml(config_path)
instrument_1   = config_trading['Instrument_A']
instrument_2   = config_trading['Instrument_B']

# OANDA Config
accountID = config_trading['DataOandaAccount']
access_token = config_trading['DataOandaToken']

# Telegram Config
TOKEN = config_trading['DataTgrChatID']
chatid = config_trading['DataTgrToken']
tb = telebot.TeleBot(TOKEN)

# Do Not Touch
pairs_traded = '%s,%s' % (str(instrument_1), str(instrument_2))
pairs_traded_dict = {instrument_1, instrument_2}

api = API(access_token=access_token, environment="practice")

stream = PricingStream(accountID=accountID, params={"instruments": pairs_traded})
orders_list = orders.OrderList(accountID)
trades_list = trades.TradesList(accountID)

candle_1 = instruments.InstrumentsCandles(instrument=instrument_1,
                                          params={"count": 1200, "granularity": "M1", "price": "B", "smooth": True})
candle_2 = instruments.InstrumentsCandles(instrument=instrument_2,
                                          params={"count": 1200, "granularity": "M1", "price": "B", "smooth": True})

actual_minute_ej = 0
actual_minute_gj = 0
minute_cached = None
go = False
first = True
restored = False
trained = False
print('It began in Africa')
sess = tf.Session()

try:
    R = api.request(stream)
    for i in R:
        if i['type'] == 'PRICE':
            if i['instrument'] == instrument_1:
                actual_minute_ej = parser.parse(i['time']).minute
                # print(actual_minute_ej)
            if i['instrument'] == instrument_2:
                actual_minute_gj = parser.parse(i['time']).minute
                # print(actual_minute_gj)
        if actual_minute_ej == actual_minute_gj and minute_cached != actual_minute_ej:
            minute_cached = actual_minute_ej
            print('Minute update...')
            if first is False:
                go = True
            first = False
        if go is True:
            print('Ready to analyze history...')
            candles_1 = api.request(candle_1)
            candles_2 = api.request(candle_2)
            rows_c1 = []
            rows_c2 = []
            mid_point = datetime.now() - timedelta(minutes=500)

            time_start = time_end = mid_point
            for c1 in candles_1['candles']:
                if c1['complete'] is True:
                    close = c1['bid']['c']
                    time = datetime.strptime(c1['time'][:19], '%Y-%m-%dT%H:%M:%S')
                    if time < time_start:
                        time_start = time
                    if time > time_end:
                        time_end = time
                    row_c1 = time, close
                    rows_c1.append(row_c1)
            print(instrument_1)
            print('C1:', len(rows_c1))
            print('TS:', time_start, ' /TE:', time_end)
            index = pd.date_range(time_start, time_end, freq='min')
            ts_c1 = pd.DataFrame(rows_c1, columns=['date', 'close'])
            ts_c1.set_index('date', drop=True, inplace=True)
            ts_c1 = ts_c1.reindex(index=index, method='pad')
            ts_c1 = ts_c1[ts_c1.index.dayofweek < 5]

            time_start = time_end = mid_point
            for c2 in candles_2['candles']:
                if c2['complete'] is True:
                    close = c2['bid']['c']
                    time = datetime.strptime(c2['time'][:19], '%Y-%m-%dT%H:%M:%S')
                    if time < time_start:
                        time_start = time
                    if time > time_end:
                        time_end = time
                    row_c2 = time, close
                    rows_c2.append(row_c2)
            print(instrument_2)
            print('C2:', len(rows_c2))
            print('TS:', time_start, ' /TE:', time_end)
            index = pd.date_range(time_start, time_end, freq='min')
            ts_c2 = pd.DataFrame(rows_c2, columns=['date', 'close'])
            ts_c2.set_index('date', drop=True, inplace=True)
            ts_c2 = ts_c2.reindex(index=index, method='pad')
            ts_c2 = ts_c2[ts_c2.index.dayofweek < 5]

            if len(ts_c1) > len(ts_c2):
                ts_c1 = ts_c1.tail(len(ts_c2))
            elif len(ts_c1) < len(ts_c2):
                ts_c2 = ts_c2.tail(len(ts_c1))

            ts_c1.to_csv(path_or_buf='STATICS/FX/2019RT/candles_1.csv', index=True, index_label='date')
            ts_c2.to_csv(path_or_buf='STATICS/FX/2019RT/candles_2.csv', index=True, index_label='date')

            # Read prices
            x = pd.read_csv('STATICS/FX/2019RT/candles_1.csv')
            y = pd.read_csv('STATICS/FX/2019RT/candles_2.csv')
            x, y = EGCointegration.clean_data(x, y, 'date', 'close')
            EG_Train = []
            EG_Test = EGCointegration(x, y, 'date', 'close')

            if trained is False:
                # Create action space
                n_hist = list(np.arange(60, 601, 60))
                n_forward = list(np.arange(120, 1201, 120))
                trade_th = list(np.arange(1, 5.1, 1))  # 1, 5.1, 1 # 5steps -- 1,  10.1, 2
                stop_loss = list(np.arange(1, 2.1, 0.5))  # 1, 2.1, 0.5 # 3steps -- 2,  11.1, 3
                cl = list(np.arange(0.05, 0.11, 0.05))  # 0.05, 0.11, 0.05 # 2steps -- 0.3,  0.71, 0.2
                actions = {'n_hist': n_hist,
                           'n_forward': n_forward,
                           'trade_th': trade_th,
                           'stop_loss': stop_loss,
                           'cl': cl}
                n_action = int(np.product([len(actions[key]) for key in actions.keys()]))

                # Create state space
                transaction_cost = [0.000]
                states = {'transaction_cost': transaction_cost}
                n_state = len(states)

                # Assign state and action spaces to config
                config_train['StateSpaceState'] = states
                config_train['ActionSpaceAction'] = actions

                # Create and build network
                one_hot = {'one_hot': {'func_name': 'one_hot',
                                       'input_arg': 'indices',
                                       'layer_para': {'indices': None,
                                                      'depth': n_state}}}
                output_layer = {'final': {'func_name': 'fully_connected',
                                          'input_arg': 'inputs',
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
                RL_Train.process(sess, save=save_model, restore=restore_model)
                trained = True

            if restore_model is False and restored is False:
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
                restored = True

            if restore_model is True and restored is False:
                folder_path = config_train['AgentModelSaverRestorePath']
                name = 'action_dict.csv'
                file_path = path.join(folder_path, name).replace('\\', '/')
                with open(file_path, 'r') as f:
                    action_dict_temp = []
                    for k, row in enumerate(csv.DictReader(f)):
                        row_ = FileIO.Record.transform(row)
                    action_dict = dict(row_)
                    # print(action_dict)
                restored = True

            indices = range(0, 1)

            import warnings

            warnings.filterwarnings('ignore')
            for j in indices:
                EG_Test.process_trading(index=j, transaction_cost=0.000, **action_dict)

            warnings.filterwarnings(action='once')

            go = False

except V20Error as e:
    sess.close()
    print("Error: {}".format(e))

except KeyboardInterrupt:
    sess.close()
    print('  ---This is the end !---')
