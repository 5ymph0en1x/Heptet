import tensorflow as tf
import itertools
import random
import numpy as np
import abc
from os import path


class Agent(metaclass=abc.ABCMeta):

    def __init__(self, network, config):
        self.session   = None
        self.network   = network
        self.config    = config
        self.data      = dict()
        self.feed_dict = dict()
        self.saver     = tf.train.Saver()
        self.counters  = dict()
        self.input_layer  = None
        self.output_layer = None
        self.get_counter()
        self.docking()

    def docking(self):
        self.input_layer  = getattr(self.network, list(self.network.__dict__.keys())[0])
        self.output_layer = getattr(self.network, list(self.network.__dict__.keys())[-1])

    def assign_network(self, network):
        self.network = network
        self.docking()

    def set_session(self, session):
        self.session = session

    def initialize_global(self):
        init = tf.global_variables_initializer()
        self.session.run(init)

    def get_counter(self):
        for key in self.config['Counter'].keys():
            self.counters[key] = StepCounter(**self.config['Counter'][key])

    def save_model(self, folder=None, name=None, session=None):
        folder_path = self.config['AgentModelSaverSavePath'] if folder is None else folder
        name        = self.config['AgentModelSaverSaveName'] if name is None else name
        file_path   = path.join(folder_path, name).replace('\\', '/')
        session     = self.session if session is None else session
        self.saver.save(session, file_path)

    def restore_model(self, folder=None, name=None, session=None):
        folder_path = self.config['AgentModelSaverRestorePath'] if folder is None else folder
        name        = self.config['AgentModelSaverRestoreName'] if name is None else name
        file_path   = path.join(folder_path, name).replace('\\', '/')
        session     = self.session if session is None else session
        # self.saver.restore(session, file_path + '.meta')
        saving = tf.train.import_meta_graph(file_path + '.meta')
        saving.restore(session, save_path=tf.train.latest_checkpoint(folder_path))

    def close(self):
        self.session.close()

    @abc.abstractmethod
    def process(self, **kwargs):
        pass


class Processor(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def process(self, **kwargs):
        pass


class Strategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def process(self, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def reward(self):
        return

    @property
    @abc.abstractmethod
    def record(self):
        return

    @reward.setter
    @abc.abstractmethod
    def reward(self, value):
        return

    @record.setter
    @abc.abstractmethod
    def record(self, value):
        return


class Network(object):

    def __init__(self, input_layer):
        self.input_layer = input_layer

    @property
    def num_layer(self):
        return self.layer_names

    @property
    def layer_names(self):
        return list(self.__dict__.keys())

    def build_layers(self, layer_dict):
        layer_names = list(layer_dict.keys())
        for name in layer_names:
            current_name = list(self.__dict__.keys())
            assert name not in current_name, 'Error: Duplicated layer names.'

            func_name    = layer_dict[name]['func_name']
            input_arg    = layer_dict[name]['input_arg']
            input_name   = current_name[-1]
            layer_para   = layer_dict[name]['layer_para']

            layer_para[input_arg] = getattr(self, input_name)
            layer_func = TFLayer.get_func(func_name)
            setattr(self, name, layer_func()(**layer_para))

    def add_layer_duplicates(self, layer_dict, n_copy):
        num_layer     = 0
        layer_names   = list(layer_dict.keys())
        for i in range(n_copy):
            num_layer += 1
            for name in layer_names:
                current_names = list(self.__dict__.keys())
                input_name    = current_names[-1]
                new_name      = name + '_' + str(num_layer)

                assert new_name not in current_names, 'Error: Duplicated layer names.'
                new_layer_dict = {new_name: layer_dict[name]}
                new_layer_dict[new_name]['input_name'] = input_name
                self.build_layers(new_layer_dict)


class TFLayer(object):

    @classmethod
    def get_func(cls, method):
        return getattr(cls, method)

    @staticmethod
    def fully_connected():
        return tf.contrib.layers.fully_connected

    @staticmethod
    def dense():
        return tf.layers.dense

    @staticmethod
    def flatten():
        return tf.layers.flatten

    @staticmethod
    def dropout():
        return tf.layers.dropout

    @staticmethod
    def softmax():
        return tf.contrib.layers.softmax

    @staticmethod
    def one_hot():
        return tf.one_hot


class Space(object):

    def __init__(self, space):
        self.check_space(space)
        self.space = space
        self.n_combination, self.indices, self.multipliers = Space.get_attribute(space)
        self.idx_range = range(len(self.indices))

    @classmethod
    def check_space(cls, space):
        assert isinstance(space, dict), 'Error:Input space should be a dictionary.'
        for value in space.values():
            assert isinstance(value, list), 'Error:Space value should be a list.'

    @classmethod
    def get_attribute(cls, space):
        n_element  = [len(space[key]) for key in space.keys()]
        multiplier = [1]
        for i in range(-1, -len(n_element), -1):
            prod = multiplier[-1] * n_element[i]
            multiplier.append(prod)
        multiplier.reverse()

        multiplier  = tuple(multiplier)
        space_index = tuple([list(range(n)) for n in n_element])
        n_comb      = np.product(n_element)
        return n_comb, space_index, multiplier

    def get_combinations(self):
        space_keys   = list(self.space.keys())
        space_sets   = list(map(list, self.space.values()))
        combinations = list(itertools.product(*space_sets))
        comb_list    = [dict(zip(space_keys, element)) for element in combinations]
        return comb_list

    def get_random_sample(self, method):
        indices = [random.choice(idx) for idx in self.indices]
        if   method == 'indices':
            return indices
        elif method == 'index':
            return self._indices_to_index(indices)
        elif method == 'one_hot':
            return self._indices_to_one_hot(indices)
        elif method == 'dict':
            return self._indices_to_dict(indices)
        else:
            raise ValueError('Error: Method should be indices/index/one_hot/dict.')

    def convert(self, sample, method):
        method = '_' + method
        return getattr(self, method)(sample)

    def _indices_to_index(self, indices):
        index = sum([indices[i] * self.multipliers[i] for i in self.idx_range])
        return index

    def _indices_to_one_hot(self, indices):
        index  = self._indices_to_index(indices)
        output = self._index_to_one_hot(index)
        return output

    def _indices_to_dict(self, indices):
        output = dict()
        keys   = list(self.space.keys())
        for i in self.idx_range:
            output[keys[i]] = self.space[keys[i]][indices[i]]
        return output

    def _index_to_indices(self, index):
        mod = index
        output = list(np.zeros(self.idx_range[-1] + 1, dtype=int))
        for i in self.idx_range:
            div, mod = divmod(mod, self.multipliers[i])
            output[i] = div
            if mod == 0:
                break
        return output

    def _index_to_one_hot(self, index):
        output = np.zeros((1, self.n_combination), dtype=int)
        output[0][index] = 1
        return output

    def _index_to_dict(self, index):
        indices = self._index_to_indices(index)
        output  = self._indices_to_dict(indices)
        return output

    def _one_hot_to_index(self, one_hot, axis=None):
        index = np.argmax(one_hot, axis=axis)
        return index

    def _one_hot_to_indices(self, one_hot):
        index  = self._one_hot_to_index(one_hot)
        output = self._index_to_indices(index)
        return output

    def _one_hot_to_dict(self, one_hot):
        index  = self._one_hot_to_index(one_hot)
        output = self._index_to_dict(index)
        return output

    def _dict_to_indices(self, dict_in):
        output = [self.space[key].index(value) for key, value in dict_in.items()]
        return output

    def _dict_to_index(self, dict_in):
        indices = self._dict_to_indices(dict_in)
        index   = self._indices_to_index(indices)
        return index

    def _dict_to_one_hot(self, dict_in):
        index  = self._dict_to_index(dict_in)
        output = self._index_to_one_hot(index)
        return output

    def _no_conversion(self, sample_in):
        return sample_in


class StepCounter(object):

    def __init__(self, name, start_num, end_num, step_size, n_buffer=0, is_descend=True, print_freq=0):
        self.name        = name
        self.start_num   = start_num
        self.end_num     = end_num
        self.step_size   = abs(step_size)
        self.n_buffer    = n_buffer
        self.is_descend  = is_descend
        self.print_freq  = print_freq
        self.value       = start_num
        self.n_step      = 0
        self.is_buffered = False
        self.is_ended    = True if start_num == end_num else False

    def reset(self, reset_buffer=False, reset_n_step=False):
        self.value = self.start_num
        if reset_n_step is True: self.n_step = 0
        if reset_buffer is True: self.is_buffered = False
        self.is_ended = True if self.start_num == self.end_num else False

    def step(self):
        if self.is_ended is False:
            self.n_step += 1
            self._check_is_buffered()
            if self.print_freq is not None:
                if self.n_step % self.print_freq == 0:
                    print('Counter [{name}]: {n_step} steps processed...'.format(name=self.name, n_step=self.n_step))
            if (self.value != self.end_num) & (self.is_buffered is True):
                if self.is_descend is True:
                    self._step_down()
                elif self.is_descend is False:
                    self._step_up()
                else:
                    raise ValueError("Error: Boolean value required for input is_descend.")

    def _check_is_buffered(self):
        if (self.is_buffered is False) & (self.n_step > self.n_buffer):
            self.is_buffered = True

    def _step_down(self):
        self.value -= self.step_size
        if self.value <= self.end_num:
            self.value = self.end_num
            self.is_ended = True
            print('Counter [{name}]: Process completed.'.format(name=self.name))

    def _step_up(self):
        self.value += self.step_size
        if self.value == self.end_num:
            self.value = self.end_num
            self.is_ended = True
            print('Counter [{name}]: Process completed.'.format(name=self.name))
