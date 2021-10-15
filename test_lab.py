import os
# NOTE increase if needed. Pytorch thread overusage https://github.com/pytorch/pytorch/issues/975
from datetime import datetime

import numpy as np

from drawing import plot_result_frames

os.environ['OMP_NUM_THREADS'] = '1'
from slm_lab.spec import spec_util
from slm_lab.lib import logger, util
from slm_lab.experiment import analysis
from slm_lab.env.openai import OpenAIEnv
from slm_lab.agent import Agent, Body
import torch
import random

seed = 75  #np.random.randint(low=0, high=2**10)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Session:

    def __init__(self, spec):
        self.spec = spec
        self.env = OpenAIEnv(self.spec)
        self.env.seed(seed)

        body = Body(self.env, self.spec)
        self.agent = Agent(self.spec, body=body)
        logger.info(f'Initialized session')

        weights = sum(p.numel() for p in self.agent.algorithm.net.parameters())  # test
        print(f'{weights} weights, model: {self.agent.algorithm.net}')           # test
        print(f'Using {self.agent.algorithm.net.device} device')                 # test

    def run_rl(self):
        current_score = 0       # test
        prev_score = 0          # test
        score = np.zeros(self.env.max_frame+1)      # test
        epsilon = np.zeros(self.env.max_frame+1)    # test
        learning_rate = np.zeros(self.env.max_frame+1)    # test

        clock = self.env.clock
        state = self.env.reset()
        done = False
        while clock.get('frame') <= self.env.max_frame:
            if done:  # reset when episode is done
                prev_score = current_score      # test
                current_score = 0               # test
                clock.tick('epi')
                state = self.env.reset()
                done = False
            clock.tick('t')
            with torch.no_grad():
                action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state = next_state

            current_score += 1                              # test
            score[clock.get('frame')-1] = prev_score        # test
            epsilon[clock.get('frame')-1] = self.agent.body.explore_var  # test
            learning_rate[clock.get('frame')-1] = self.agent.algorithm.optim.param_groups[0]['lr']  # test

            if clock.get('frame') % self.env.log_frequency == 0:
                self.agent.body.ckpt(self.env, 'train')
                self.agent.body.log_summary('train')

        # test:
        title = f'SLM DQN seed: {seed}'
        info = f''
        time = datetime.now().strftime("%Y.%m.%d %H-%M")
        filename = f'./tmp/{time}_training_SLM.png'
        plot_result_frames([score], epsilon, title, info, filename, learning_rate=learning_rate)

    def close(self):
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self):
        self.run_rl()
        # this will run SLM Lab's built-in analysis module and plot graphs
        self.data = None  # analysis.analyze_session(self.spec, self.agent.body.train_df, 'train')
        self.close()
        return self.data


# spec = spec_util.get(spec_file='dqn_lunarlander.json', spec_name='dqn_lunarlander')
# spec = spec_util.get(spec_file='dqn_cartpole.json', spec_name='dqn_cartpole')
# spec = spec_util.get(spec_file='test_lab_sac.json', spec_name='sac_cartpole')
spec = spec_util.get(spec_file='demo.json', spec_name='dqn_cartpole')
os.environ['lab_mode'] = 'train'  # set to 'dev' for rendering

# update the tracking indices
spec_util.tick(spec, 'trial')
spec_util.tick(spec, 'session')

# initialize and run session
session = Session(spec)
session_metrics = session.run()