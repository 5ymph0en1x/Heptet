import random
import numpy as np
from MAIN.Basics import Processor, Space
from operator import itemgetter


class StateSpace(Processor, Space):

    def __init__(self, agent):
        self.agent = agent
        super().__init__(agent.config['StateSpaceState'])

    def process(self):
        self.agent.data['NETWORK_STATE'] = self._get_network_input()
        self.agent.data['ENGINE_STATE' ] = self._get_engine_input()

    def _get_network_input(self):
        method = self.agent.config['StateSpaceNetworkSampleType']
        state  = self.get_random_sample(method)
        return state

    def _get_engine_input(self):
        method = self.agent.config['StateSpaceEngineSampleConversion']
        state  = self.agent.data['NETWORK_STATE']
        state  = self.convert(state, method)
        return state


class ActionSpace(Processor, Space):

    def __init__(self, agent):
        self.agent = agent
        super().__init__(agent.config['ActionSpaceAction'])

    def process(self):
        self.agent.data['NETWORK_ACTION'] = self._get_network_input()
        self.agent.data['ENGINE_ACTION' ] = self._get_engine_input()

    def _get_network_input(self):
        method = self.agent.config['ActionSpaceNetworkSampleType']
        if method == 'exploration':
            self.agent.exploration.process()
            action = self.agent.data['EXPLORATION_ACTION']
        else:
            action = self.get_random_sample(method)
        return action

    def _get_engine_input(self):
        method = self.agent.config['ActionSpaceEngineSampleConversion']
        index  = self.agent.data['EXPLORATION_ACTION']
        action = self.convert(index, method)
        return action


class RewardEngine(Processor):

    def __init__(self, agent, engine):

        self.engine = engine
        self.agent  = agent

    def process(self):
        reward, record = self._get_reward()
        self.agent.data['ENGINE_REWARD'] = reward
        self.agent.data['ENGINE_RECORD'] = record

    def _get_reward(self):
        state  = self.agent.data['ENGINE_STATE']
        action = self.agent.data['ENGINE_ACTION']
        self.engine.process(**state, **action)
        return self.engine.reward, self.engine.record


class Exploration(Processor):

    def __init__(self, agent):
        self.agent   = agent
        self.method  = agent.config['ExplorationMethod']
        self.counter = agent.counters[agent.config['ExplorationCounter']]
        self.func    = self.get_func(self.method)
        if self.method == 'boltzmann':
            self.target_attr = getattr(self.agent, self.agent.config['ExplorationBoltzmannProbAttribute'])

    def process(self):
        self.agent.data['EXPLORATION_ACTION'] = self.func()

    def get_func(self, method):
        method = '_' + method
        return getattr(self, method)

    def _random(self):
        n_action = self.agent.action_space.n_combination
        action_idx = random.randrange(n_action)
        return action_idx

    def _greedy(self):
        self.agent.feed_dict[self.agent.input_layer] = [self.agent.data['NETWORK_STATE']]
        q_value = self.agent.session.run(self.agent.output_layer, feed_dict=self.agent.feed_dict)
        q_value = q_value.reshape(-1,)
        action_idx = np.argmax(q_value)
        return action_idx

    def _e_greedy(self):
        e = self.counter.value
        action_idx = self._random() if random.random() < e else self._greedy()
        self.counter.step()
        return action_idx

    def _boltzmann(self):
        self.agent.data['BOLTZMANN_TEMP'] = self.counter.value
        self.agent.feed_dict[self.agent.input_layer] = [self.agent.data['NETWORK_STATE']]
        self.agent.feed_dict[self.agent.temp       ] = [self.agent.data['BOLTZMANN_TEMP']]
        prob = self.agent.session.run(self.target_attr, feed_dict=self.agent.feed_dict)
        action_idx = np.random.choice(self.agent.action_space.n_combination, p=prob)
        self.counter.step()
        return action_idx


class ExperienceBuffer(Processor):

    def __init__(self, agent):
        buffer_size  = int(agent.config['ExperienceBufferBufferSize'])
        self.agent   = agent
        self.buffer  = []
        self.buffer_size = buffer_size

    def process(self, method):
        if method == 'add':
            self._add_sample(self.agent.data['SAMPLE'])
        elif method == 'get':
            self.agent.data['EXPERIENCE_BUFFER_SAMPLE'] = self._get_sample()
        else:
            raise ValueError("Error: method name should be add/get.")

    def _add_sample(self, sample):
        sample_length = len(sample)
        buffer_length = len(self.buffer)
        is_single_sample = True if sample_length == 1 else False
        if is_single_sample is True:
            total_length = buffer_length
        elif is_single_sample is False:
            total_length = buffer_length + sample_length
        else:
            raise ValueError("Error: Boolean value required for input is_single_sample.")

        if total_length > buffer_length:
            idx_start = total_length - buffer_length
            self.buffer = self.buffer[idx_start:]
            self.buffer.extend(sample)
        else:
            self.buffer.extend(sample)

    def _get_sample(self):
        size   = int(self.agent.config['ExperienceBufferSamplingSize'])
        sample = itemgetter(*np.random.randint(len(self.buffer), size=size))(self.buffer)
        return sample


class Recorder(Processor):

    def __init__(self, agent):
        self.data_field  = agent.config['RecorderDataField']
        self.record_freq = agent.config['RecorderRecordFreq']
        self.agent = agent
        if self.data_field is not None:
            self.record = {key: [] for key in self.data_field}

    def process(self):
        if self.data_field is not None:
            if (self.agent.epoch_counter.n_step % self.record_freq) == 0:
                for key in self.record.keys():
                    self.record[key].append(self.agent.data[key])
