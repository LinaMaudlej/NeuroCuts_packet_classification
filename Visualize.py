#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOAgent
from neurocuts_env import NeuroCutsEnv
from run_neurocuts import on_episode_end
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from mask import PartitionMaskModel
from tree import Node
from tree import load_rules_from_file

#%load_ext autoreload
#%autoreload 2

# In[2]:


### Init ###
ray.init()

# In[3]:


rules = os.path.abspath("classbench/{}".format("fw5_1k"))
dump_dir = "/tmp/neurocuts_out"
reward_shape = "linear"

# In[4]:


### Setup ###
register_env(
    "tree_env", lambda env_config: NeuroCutsEnv(
        env_config["rules"],
        max_depth=env_config["max_depth"],
        max_actions_per_episode=env_config["max_actions"],
        dump_dir=env_config["dump_dir"],
        depth_weight=env_config["depth_weight"],
        reward_shape=env_config["reward_shape"],
        partition_mode=env_config["partition_mode"]))
ModelCatalog.register_custom_model("mask", PartitionMaskModel)

config =  {
        "num_gpus": 0,
        "num_workers": 3,
        "sgd_minibatch_size": 100,
        "sample_batch_size": 200,
        "train_batch_size": 1000,
        "batch_mode": "complete_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "custom_model": "mask",
            "fcnet_hiddens": [512, 512],
        },
        "vf_share_layers": True,
        "entropy_coeff": 0.01,
        "callbacks": {
            "on_episode_end": tune.function(on_episode_end),
        },
        "env_config": {
            "dump_dir": dump_dir,
            "partition_mode": None,
            "reward_shape": reward_shape,
            "max_depth": 100,
            "max_actions": 1000,
            "depth_weight": 1.0,
            "rules": rules,
        },
    }

# In[5]:


# file = "/Users/yitianzou/ray_results/neurocuts_None/PPO_tree_env_1_rules=_Users_yitianzou_neurocuts-code_classbench_acl5_1k,sample_batch_size=400_2019-02-20_15-30-42uewkqv_l/checkpoint_1/checkpoint-1"
# env = NeuroCutsEnv(rules_file="classbench/acl5_1k")
# agent = PPOAgent(env="tree_env", config=config)
# agent.restore(file)

# In[34]:


# directory = "/Users/yitianzou/ray_results/neurocuts_None/PPO_tree_env_0_2019-03-08_23-48-42pb2k3nsd"
# directory = "/Users/yitianzou/ray_results/neurocuts_None/PPO_tree_env_0_2019-03-20_20-38-09ixikyqq8/"
directory = "/Users/yitianzou/ray_results/neurocuts_None/PPO_tree_env_0_2019-03-27_16-29-19mwsnjoc7"
env = NeuroCutsEnv(rules_file=rules)

agent_1 = PPOAgent(env="tree_env", config=config)
agent_1.restore(directory+"/checkpoint_1/checkpoint-1")
agent_1M = PPOAgent(env="tree_env", config=config)
agent_1M.restore(directory+"/checkpoint_23/checkpoint-23")
agent_2M = PPOAgent(env="tree_env", config=config)
agent_2M.restore(directory+"/checkpoint_54/checkpoint-54")
agent_4M = PPOAgent(env="tree_env", config=config)
agent_4M.restore(directory+"/checkpoint_136/checkpoint-136")
agent_8M = PPOAgent(env="tree_env", config=config)
agent_8M.restore(directory+"/checkpoint_318/checkpoint-318")

# In[35]:


meta = directory + "/checkpoint_318/checkpoint-318.tune_metadata"
with open(meta, 'rb') as f:
    text = pickle.load(f)
print(text)

# ## TODOS
# 1. Verify that after num-fast # iterations, reward (memory_access) has converged to around 10 - DONE
# 2. Generate distribution of (src, dst) - DONE
# 3. Make contour for 1st checkpoint - DONE
# 4. Vectorize _value - DONE
# 5. Get checkpoints + run for different ones - DONE
# 6. Try different initializations of other bits in real_obs
# 7. Random vector projection?

# ### Fix
# 
# Construct Rule(id=0, ranges=[src_ip_start, src_ip_end, dst_ip_start, dst_ip_end, src_port_start, src_port_end, dst_port_start, dst_port_end, proto_start, proto_end])
# 
# Call rule.get_state()

# In[36]:


rules = load_rules_from_file("classbench/fw5_1k")
# rules = load_rules_from_file("classbench/acl5_1k")
X = np.arange(0, 2**32, 2**28)
Y = np.arange(0, 2**32, 2**28)
N = len(X)
r = rules[0]
print(len(rules))
print(X[1])
# print(rules[0].ranges)
# print(rules[100].ranges)
# print(rules[200].ranges)
# print(rules[300].ranges)
# print(rules[400].ranges)
# for i in range(N - 1):
#     for j in range(N - 1):
#         src_ip_range = [X[i], X[i + 1]]
#         dst_ip_range = [Y[i], Y[i + 1]]
#         if src_ip_range[0] < r.ranges[0] < src_ip_

# x_pts = [r.ranges[0], r.ranges[1]]
# y_pts = [r.ranges[2], r.ranges[3]]
# plt.scatter(x_pts, y_pts)

# In[37]:


def intersect(sx1, sx2, sy1, sy2, ox1, ox2, oy1, oy2):
    return (sx1 < ox2 and sx2 > ox1) and (sy1 < oy2 and sy2 > oy1)

# In[38]:


src_diff = [r.ranges[1] - r.ranges[0] for r in rules]
dst_diff = [r.ranges[3] - r.ranges[2] for r in rules]
print(max(src_diff))
print(max(dst_diff))
print(2**28)

# In[39]:


num_rules = np.zeros((N-1, N-1))
for r in rules:
    for i in range(N-1):
        for j in range(N-1):
            if intersect(r.ranges[0], r.ranges[1], r.ranges[2], r.ranges[3], X[i], X[i+1], Y[j], Y[j+1]):
                num_rules[i, j] += 1

print(num_rules)

# In[40]:


def make_binary_list(num):
    b = "{0:b}".format(int(num))
    lst = [float(d) for d in b]
    diff = 32 - len(lst)
    lst = [0.0]*diff + lst
    return lst
def make_dec(ip):
    ip_str = ''.join(str(int(e)) for e in ip)
    ip_dec = int(ip_str, 2)
    return ip_dec
i = 2**8 + 1
b = make_binary_list(i)
print(b)
print(make_dec(b))
b = list(bin(int(i))[2:])
l = [0.0] * (32 - len(b)) + [float(i) for i in b]
print(l)

# In[41]:


### Get values ###
def value(policy, obs):
    feed_dict = {
       policy.observations: obs,
    }
    vf = policy.sess.run(policy.value_function, feed_dict)
    return vf

# values_lst = value(agent.get_policy(), obs_lst)
# values = values_lst.reshape((N, N))
# print(values[30][30])

# In[ ]:




# In[50]:


def generate_obs_and_vals(agent):
    max_ip = 2**32
    max_port = 2**16
    max_proto = 2**8
    interval = 2**28
    X = np.arange(0, max_ip, interval)
    Y = np.arange(0, max_ip, interval)
    N = len(X)
    obs_lst = []

    for i in range(N-1): 
        for j in range(N-1):
#             [src_ip_start, src_ip_end, dst_ip_start, dst_ip_end, src_port_start, src_port_end, dst_port_start, dst_port_end, proto_start, proto_end]
#             src = make_binary_list(X[i])
#             dst = make_binary_list(Y[j])
#             size = 279 - 64
#             if mask is None:
#                 mask = [0.0] * size
#             real_obs = src + dst + mask
            ranges = [X[i], X[i+1], Y[j], Y[j+1], 0, max_port, 0, max_port, 0, 2**8]
            node = Node(0, ranges, [], 0, None, None)
            real_obs = node.get_state()
            action_mask = [0.0] * 10
            obs = {"real_obs": real_obs, "action_mask": action_mask}
            _obs = agent.local_evaluator.preprocessors["default"].transform(obs)
            obs_lst.append(_obs)
#     print(obs_lst)
    values_lst = value(agent.get_policy(), obs_lst)
    values = values_lst.reshape((N-1, N-1))
    return X, Y, values

# In[53]:


### Visualize ###
def visualize(agent, num):
    X, Y, values = generate_obs_and_vals(agent)
    plt.imshow(values, cmap='hot', interpolation='nearest', origin='lower')
#     _X, _Y = np.meshgrid(X, Y)
#     plt.contour(_X, _Y, values)
    plt.title("Timesteps {}".format(num))
    plt.xlabel("Dst ip")
    plt.ylabel("Src ip")
    plt.show()
    
visualize(agent_1, "1")
visualize(agent_1M, "1M")
visualize(agent_2M, "2M")
visualize(agent_4M, "4M")
visualize(agent_8M, "8M")
# Yellow = high
# Green = medium
# Blue = low

# In[45]:


plt.imshow(num_rules, cmap='hot', interpolation='nearest', origin='lower')

# In[ ]:




# In[ ]:



