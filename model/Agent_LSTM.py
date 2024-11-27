#!/usr/bin/env python
# coding: utf-8

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random


transition = namedtuple('transition', ('state', 'action', 'reward', 'done'))

class ReplayMemory():
    def __init__(self, MEMORY_SIZE, BATCH_SIZE):
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.memory = []
        self.position = 0

    def push(self, *args):
        # a ring buffer
        if len(self.memory) < self.MEMORY_SIZE:
            self.memory.append(None)
            
        self.memory[self.position] = transition(*args)
        self.position = (self.position+1) % self.MEMORY_SIZE
        
    def sample(self):
        batch = random.sample(self.memory, self.BATCH_SIZE)
        batch = transition(*zip(*batch))
        return batch
        
    def load(self, memory):          
        self.memory, self.position = memory
        
    def reset(self):
        self.memory = []
        self.position = 0


class BeliefStep(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.STATE_DIM = arg.STATE_DIM
        self.OBS_DIM = arg.OBS_DIM
        self.obs_noise_range = arg.obs_noise_range
        
        self.H = torch.zeros(self.OBS_DIM, self.STATE_DIM)
        self.H[0, -2] = 1
        self.H[1, -1] = 1
        
    @property
    def obs_noise_range(self):
        return self._obs_noise_range
    
    @obs_noise_range.setter
    def obs_noise_range(self, value):
        self._obs_noise_range = [0, 0] if value is None else value

    def reset(self, pro_gains, obs_noise_std=None):
        self.obs_noise_std = obs_noise_std
        
        if self.obs_noise_std is None:
            self.obs_noise_std = torch.zeros(1).uniform_(self.obs_noise_range[0], self.obs_noise_range[1]) * pro_gains

    def forward(self, x):
        zita = (self.obs_noise_std * torch.randn(self.OBS_DIM)).view([-1, 1])
        o_t = self.H @ x + zita
        
        return o_t


class ActionNoise():
    def __init__(self, ACTION_DIM, mean, std):
        self.mu = torch.ones(ACTION_DIM) * mean
        self.std = std
        self.ACTION_DIM = ACTION_DIM

    def reset(self, mean, std):
        self.mu = torch.ones(self.ACTION_DIM) * mean
        self.std = std

    def noise(self):
        n = torch.randn(self.ACTION_DIM)
        return self.mu + self.std * n


class Agent():
    def __init__(self, arg, Actor, Critic):
        self.__dict__ .update(arg.__dict__)

        self.actor = Actor(self.OBS_DIM, self.ACTION_DIM, self.TARGET_DIM, 
                           self.RNN_SIZE, self.FC_SIZE, self.RNN).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()
        self.actor_optim = self.optimizer(self.actor.parameters(), lr=self.lr, eps=self.eps)
        
        self.critic = Critic(self.OBS_DIM, self.ACTION_DIM, self.TARGET_DIM, 
                             self.RNN_SIZE, self.FC_SIZE, self.RNN).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.eval()
        self.critic_optim = self.optimizer(self.critic.parameters(), lr=self.lr, eps=self.eps)
        
        self.memory = ReplayMemory(arg.MEMORY_SIZE, arg.BATCH_SIZE)
        self.bstep = BeliefStep(arg)
        
        self.initial_episode = 0
        self.it = 0
        self.episodic = True


    def select_action(self, state, hidden_in, action_noise=None):            
        with torch.no_grad():
            action, hidden_out = self.actor(state, hidden_in, return_hidden=True, critic=self.critic)
            
        action = action.cpu()
        action_raw = action.clone()
        if (action_noise is not None) and (action_raw.abs() > self.TERMINAL_ACTION).any():
            action += action_noise.noise().view_as(action)

        return action.clamp(-1, 1), action_raw, hidden_out
    
    def target_smoothing(self, next_actions):
        mask_stop = (next_actions.view(-1, self.ACTION_DIM).abs().max(dim=1).values < self.TERMINAL_ACTION
                        ).view(-1, 1).repeat(1, self.ACTION_DIM).view_as(next_actions)
        mask_nonstop_pos = (next_actions > self.TERMINAL_ACTION) & (~mask_stop)
        mask_nonstop_neg = (next_actions < -self.TERMINAL_ACTION) & (~mask_stop)
        mask_nonstop_other = (next_actions.abs() < self.TERMINAL_ACTION) & (~mask_stop)

        next_actions[mask_stop] = (next_actions[mask_stop]                         + torch.zeros_like(next_actions[mask_stop]).normal_(
                                                mean=0, std=self.policy_noise)
                        ).clamp(-self.TERMINAL_ACTION, self.TERMINAL_ACTION)

        next_actions[mask_nonstop_pos] = (next_actions[mask_nonstop_pos]                         + torch.zeros_like(next_actions[mask_nonstop_pos]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(self.TERMINAL_ACTION, 1)

        next_actions[mask_nonstop_neg] = (next_actions[mask_nonstop_neg]                         + torch.zeros_like(next_actions[mask_nonstop_neg]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(-1, -self.TERMINAL_ACTION)

        next_actions[mask_nonstop_other] = (next_actions[mask_nonstop_other]                         + torch.zeros_like(next_actions[mask_nonstop_other]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(-1, 1)
        
        return next_actions

    def update_parameters(self, batch):
        states = torch.cat(batch.state, dim=1)
        actions =  torch.cat(batch.action, dim=1)
        rewards = torch.cat(batch.reward, dim=1)
        if self.episodic:
            dones = torch.cat(batch.done, dim=1)
        else:
            dones = torch.zeros_like(rewards)
        
        with torch.no_grad():
            # get next action and apply target policy smoothing
            #next_states = torch.zeros_like(states)
            #next_states[:-1] = states[1:]
            next_states = states[1:]
            states = states[:-1]
                       
            _, t1_hidden = self.target_actor(states[:1], hidden_in=None, return_hidden=True, critic=self.target_critic)
            next_actions = self.target_actor(next_states, hidden_in=t1_hidden, return_hidden=False, critic=self.target_critic)
            next_actions = self.target_smoothing(next_actions)

            # compute the target Q
            _, _, t1_hidden1, t1_hidden2 = self.target_critic(states[:1], actions[:1], return_hidden=True)
            target_Q1, target_Q2 = self.target_critic(next_states, next_actions, hidden_in1=t1_hidden1, hidden_in2=t1_hidden2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1-dones) * self.GAMMA * target_Q

        # current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # delay policy updates
        if self.it % self.POLICY_FREQ == 0:
            # define actor loss
            actor_loss = - self.critic.Q1(states, self.actor(states, hidden_in=None, return_hidden=False, critic=self.critic)).mean()
            
            # optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)
        else:
            actor_loss = torch.tensor([0])

        return actor_loss.detach().item(), critic_loss.detach().item()

    def learn(self):
        batch = self.memory.sample()
        loss_logs = self.update_parameters(batch)
        self.it += 1
        return loss_logs

    def save(self, save_memory, episode, pre_phase=False, full_param=True):
        if pre_phase:
            file = self.data_path / f'{self.filename}-{episode}_pre.pth.tar'
        else:
            file = self.data_path / f'{self.filename}-{episode}.pth.tar'
            
        state = {'actor_dict': self.actor.state_dict(),
                 'critic_dict': self.critic.state_dict()}
        if full_param:
            state.update({'target_actor_dict': self.target_actor.state_dict(),
                          'target_critic_dict': self.target_critic.state_dict(),
                          'actor_optimizer_dict': self.actor_optim.state_dict(),
                          'critic_optimizer_dict': self.critic_optim.state_dict(),
                          'episode': episode})
        if save_memory:
            state['memory'] = (self.memory.memory, self.memory.position)

        torch.save(state, file)

    def load(self, filename, load_memory, load_optimzer, full_param=False, load_name=True):
        if load_name:
            self.filename = filename
        file = self.data_path / f'{filename}.pth.tar'
        state = torch.load(file)

        self.actor.load_state_dict(state['actor_dict'])
        self.critic.load_state_dict(state['critic_dict'])
        if full_param:
            self.target_actor.load_state_dict(state['target_actor_dict'])
            self.target_critic.load_state_dict(state['target_critic_dict'])
            #self.initial_episode = state['episode']
        
        if load_memory is True:
            self.memory.load(state['memory'])
        if load_optimzer is True:
            self.actor_optim.load_state_dict(state['actor_optimizer_dict'])
            self.critic_optim.load_state_dict(state['critic_optimizer_dict'])

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.TAU) + param.data * self.TAU)
            
    def mirror_traj(self, states, actions, mirrored_index=(1, 3, 4)):
        # state index 1: w; 3: action aw; 4: target x
        states_ = states.clone()
        states_[..., mirrored_index] = - states_[..., mirrored_index]
        # 1 of action indexes angular action aw
        actions_ = actions.clone()
        actions_[..., 1] = - actions_[..., 1]
        
        return states_, actions_

