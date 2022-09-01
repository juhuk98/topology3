import numpy as np
import pandas as pd

from collections import deque
from matplotlib import pyplot as plt
from node import Source, Destination, Source_3
from node import Node


def states_shape(state, a):
  states = np.empty(len(a), np.ndarray)
  for i in range(len(a)):
    if i==0:
      states[i] = state[:a[i]]
    else:
      states[i] = state[sum(a[:i]):sum(a[:i+1])]
  return states

def number_to_action(action_id, num_agent):  # number -> binary gcl code
    s = '0' + str(num_agent) + 'b'
    b_id = format(int(action_id), s)
    action_ = np.array(list(map(int, b_id)))
    return action_

class PortBased0(object):
	
  def __init__(self, n_agents=1, topology=0, node_type=0, random=0):
    self.time = 0
    self.n_agents = n_agents
    self.node_type = node_type
    self.random = random
    
    self.topology = topology
    if topology == 0:
      self.links, self.srcs, self.dsts, self.nodes = self.topology0()
      self.state_dim = np.array([8, 8], np.int32)
      self.action_bound = np.array(self.state_dim/4, np.int32)
      self.n_RL_node = 2
      self.max_output_port_num = 2
    elif topology == 1:
      self.links, self.srcs, self.dsts, self.nodes = self.topology1()
      self.state_dim = np.array([8, 8, 8, 8], np.int32)
      self.action_bound = np.array(self.state_dim/4, np.int32)
      self.n_RL_node = 4
      self.max_output_port_num = 2
    elif topology == 2:
      self.links, self.srcs, self.dsts, self.nodes = self.topology2()
      self.state_dim = np.array([8, 8, 8, 8, 8], np.int32)
      self.action_bound = np.array(self.state_dim/4, np.int32)
      self.n_RL_node = 5
      self.max_output_port_num = 2
    elif topology == 3:
      self.links, self.srcs, self.dsts, self.nodes = self.topology3()
      self.state_dim = np.array([8], np.int32)
      self.action_bound = np.array(self.state_dim//4, np.int32)
      self.n_RL_node = 1
      self.max_output_port_num = 1
    elif topology == 4:
      self.links, self.srcs, self.dsts, self.nodes = self.topology4()
      self.state_dim = np.array([8, 16, 8, 16, 16, 16, 8, 16, 8], np.int32)
      self.action_bound = np.array(self.state_dim//4, np.int32)
      self.n_RL_node = 9
      self.max_output_port_num = 2
    self.num_src = len(self.srcs)
    self.num_dst = len(self.dsts)
    self.num_node = len(self.nodes)

    try:
        if n_agents<=self.n_RL_node:
            pass
    except OSError:
        print('Error: Too many agents.')

  def reset(self) -> np.ndarray:
    self.time = 0
    states = np.empty((self.num_node), np.ndarray)

    if self.topology == 0:
      self.links, self.srcs, self.dsts, self.nodes = self.topology0()
    elif self.topology == 1:
      self.links, self.srcs, self.dsts, self.nodes = self.topology1()
    elif self.topology == 2:
      self.links, self.srcs, self.dsts, self.nodes = self.topology2()
    elif self.topology == 3:
      self.links, self.srcs, self.dsts, self.nodes = self.topology3()
    elif self.topology == 4:
      self.links, self.srcs, self.dsts, self.nodes = self.topology4()

    states = np.empty(self.num_node, np.ndarray)
    for i in range(self.num_src):
      self.srcs[i].send(self.time)
    for i in range(self.num_node):
      self.nodes[i].receive(self.time)
    for i in range(self.num_node):
      if sum(self.nodes[i].module_type):
        states[i] = self.nodes[i].get_state(self.time)
    states = [elem for elem in states if elem is not None]
    #states = np.array(s_, dtype=object)
    if self.n_agents==1:
      states = np.hstack(states)
      states = states.reshape(1,-1)
    return states

  def step(self, actions):
    next_states = np.empty((self.num_node), np.ndarray)
    rewards = np.zeros(0)#np.empty(self.num_node)
    dones = np.zeros(self.num_node)
    if self.n_agents==1:
      actions = number_to_action(actions[0], self.n_RL_node)

    c = np.zeros((self.num_node,self.max_output_port_num))
    a_idx = 0
    for i in range(self.num_node):
      n = sum(self.nodes[i].module_type)
      for j in range(n):
        if self.nodes[i].empty[j]:
          c[i, j] = self.nodes[i].count[j]
      if n:
        self.nodes[i].send(actions[a_idx], self.time)
        a_idx += 1
      else:
        self.nodes[i].send(0, self.time)
    ##문제: actions[i]에서 i가 인덱스를 벗어나는 경우 고려해야 함. node에도 send함수에 동일한 문제 있음
    self.time+=1

    for i in range(self.num_src):
      self.srcs[i].send(self.time)
    for i in range(self.num_node):
      self.nodes[i].receive(self.time)
    
    for i in range(self.num_dst):
      self.dsts[i].receive(self.time)

    for i in range(self.num_node):
      mt = self.nodes[i].module_type
      if sum(mt):
        next_states[i] = self.nodes[i].get_state(self.time)
        for j in range(len(mt)):
          if mt[j]:
            if c[i, j] != 0 and c[i, j] == self.nodes[i].count[j]:
              #rewards = np.append(rewards, -20)
              rewards = np.append(rewards, -2*max(next_states[i].flatten()))
            elif c[i, j] > self.nodes[i].count[j]:
              rewards = np.append(rewards, self.nodes[i].eds[j] - max(next_states[i].flatten()))
            else:
              rewards = np.append(rewards, 0)
      if sum(self.nodes[i].count) == 0:
        dones[i] = 1
    next_states = [elem for elem in next_states if elem is not None]
    #next_states = np.array([elem for elem in next_states if elem is not None])
    if self.n_agents==1:
      next_states = np.hstack(next_states)
      next_states = next_states.reshape(1,1,-1)
      rewards = sum(rewards)
      rewards = np.array([rewards])
      dones = [all(dones)]
    #print(next_states)
    return [next_states, rewards, dones, 0]

  def get_max_delay(self):
    max_delay = 0
    for i in range(self.num_dst):
      if len(self.dsts[i].delay):
        max_delay_ = max(self.dsts[i].delay)
        if max_delay_ > max_delay:
          max_delay = max_delay_
      else:
        max_delay = 100

    return max_delay

  def topology0(self):
    if self.random:
      scenario=np.full((3,1),None)
      n_packets = 10
    else:
      scenario = [[1], [1], [0,1]]
      n_packets = 1

    links = {"src0_to_node0": deque(),
            "src1_to_node0": deque(),
            "src2_to_node1": deque(),
            "node0_to_node1": deque(),
            "node1_to_node2": deque(),
            "node2_to_dst0": deque(),
            "node1_to_dst1": deque(),
            "node2_to_dst2": deque()}

    srcs = {0: Source(src_id = 0, output = links["src0_to_node0"], 
            slot_length = 20, num = n_packets, hop = 4, route = [[0,0], [0,0], [0,0]], scenario=scenario[0]),
            1: Source(1, links["src1_to_node0"], 20, n_packets, 3, [[0,1], [1,0]], scenario[1]),
            2: Source(2, links["src2_to_node1"], 20, n_packets, 3, [[0,1], [1,0]], scenario[2])}
    dsts = {0: Destination(0, links["node2_to_dst0"]),
            1: Destination(1, links["node1_to_dst1"]),
            2: Destination(2, links["node2_to_dst2"])}

    nodes = {0: Node(0, [2*n_packets],
            [links["src0_to_node0"], links["src1_to_node0"]],
            [links["node0_to_node1"]], [1], self.node_type),
            1: Node(1, [2*n_packets, n_packets],
            [links["node0_to_node1"], links["src2_to_node1"]],
            [links["node1_to_node2"], links["node1_to_dst1"]], [1, 0], self.node_type),
            2: Node(2, [n_packets, n_packets],
            [links["node1_to_node2"]],
            [links["node2_to_dst0"], links["node2_to_dst2"]], [0, 0], self.node_type)}
    
    return links, srcs, dsts, nodes

  def topology1(self):
    if self.random:
      scenario=np.full((5,1),None)
      n_packets = 10
    else:
      scenario = [[1], [1], [0,1], [0,0,1], [0,0,0,1]]
      n_packets = 1

    links = {"src0_to_node0": deque(),
              "src1_to_node0": deque(),
              "src2_to_node1": deque(),
              "src3_to_node2": deque(),
              "src4_to_node3": deque(),
              "node0_to_node1": deque(),
              "node1_to_node2": deque(),
              "node2_to_node3": deque(),
              "node3_to_node4": deque(),
              "node4_to_dst0": deque(),
              "node4_to_dst4": deque(),
              "node3_to_dst3": deque(),
              "node2_to_dst2": deque(),
              "node1_to_dst1": deque()}
    srcs = {0: Source(src_id = 0, output = links["src0_to_node0"], 
                      slot_length = 20, num = n_packets, hop = 6, 
                      route = [[0,0], [0,0], [0,0], [0,0], [0,0]], scenario=scenario[0]),
            1: Source(1, links["src1_to_node0"], 20, n_packets, 3, [[0,1], [1,0]], scenario[1]),
            2: Source(2, links["src2_to_node1"], 20, n_packets, 3, [[0,1], [1,0]], scenario[2]),
            3: Source(3, links["src3_to_node2"], 20, n_packets, 3, [[0,1], [1,0]], scenario[3]),
            4: Source(4, links["src4_to_node3"], 20, n_packets, 3, [[0,1], [1,0]], scenario[4])}
    dsts = {0: Destination(0, links["node4_to_dst0"]),
            1: Destination(1, links["node1_to_dst1"]),
            2: Destination(2, links["node2_to_dst2"]),
            3: Destination(3, links["node3_to_dst3"]),
            4: Destination(4, links["node4_to_dst4"])}
    nodes = {0: Node(0, [2*n_packets],
                [links["src0_to_node0"], links["src1_to_node0"]],
                [links["node0_to_node1"]], [1], self.node_type),
            1: Node(1, [2*n_packets, n_packets],
                [links["node0_to_node1"], links["src2_to_node1"]],
                [links["node1_to_node2"], links["node1_to_dst1"]], [1,0], self.node_type),
            2: Node(2, [2*n_packets, n_packets],
                [links["node1_to_node2"], links["src3_to_node2"]],
                [links["node2_to_node3"], links["node2_to_dst2"]], [1,0], self.node_type),
            3: Node(3, [2*n_packets, n_packets],
                [links["node2_to_node3"], links["src4_to_node3"]],
                [links["node3_to_node4"], links["node3_to_dst3"]], [1,0], self.node_type),
            4: Node(4, [n_packets, n_packets],
                [links["node3_to_node4"]],
                [links["node4_to_dst0"], links["node4_to_dst4"]], [0,0], self.node_type)}

    return links, srcs, dsts, nodes

  
  def topology2(self):
    if self.random:
      scenario=np.full((6,1),None)
      n_packets = 10
    else:
      scenario = [[1], [1], [1], [1], [0,1], [0,1]]
      n_packets = 1

    links = {"src0_to_node0": deque(),
              "src1_to_node0": deque(),
              "src2_to_node1": deque(),
              "src3_to_node1": deque(),
              "src4_to_node2": deque(),
              "src5_to_node3": deque(),
              "node0_to_node2": deque(),
              "node1_to_node3": deque(),
              "node2_to_node4": deque(),
              "node3_to_node4": deque(),
              "node4_to_node5": deque(),
              "node5_to_dst0": deque(),
              "node5_to_dst1": deque()}
    srcs = {0: Source(src_id = 0, output = links["src0_to_node0"], 
                      slot_length = 20, num = n_packets, hop = 5, 
                      route = [[0,0], [0,0], [0,0], [0,0]], scenario=scenario[0]),
            1: Source(1, links["src1_to_node0"], 20, n_packets, 5, [[0,1], [0,0], [0,0], [0,0]], scenario[1]),
            2: Source(2, links["src2_to_node1"], 20, n_packets, 5, [[0,0], [0,0], [0,1], [1,0]], scenario[2]),
            3: Source(3, links["src3_to_node1"], 20, n_packets, 5, [[0,1], [0,0], [0,1], [1,0]], scenario[3]),
            4: Source(4, links["src4_to_node2"], 20, n_packets, 4, [[0,1], [0,0], [0,0]], scenario[4]),
            5: Source(5, links["src5_to_node3"], 20, n_packets, 4, [[0,1], [0,1], [1,0]], scenario[5])}
    dsts = {0: Destination(0, links["node5_to_dst0"]),
            1: Destination(1, links["node5_to_dst1"])}
    nodes = {0: Node(0, [2*n_packets],
                [links["src0_to_node0"], links["src1_to_node0"]],
                [links["node0_to_node2"]], [1], self.node_type),
            1: Node(1, [2*n_packets],
                [links["src2_to_node1"], links["src3_to_node1"]],
                [links["node1_to_node3"]], [1], self.node_type),
            2: Node(2, [3*n_packets],
                [links["node0_to_node2"], links["src4_to_node2"]],
                [links["node2_to_node4"]], [1], self.node_type),
            3: Node(3, [3*n_packets],
                [links["node1_to_node3"], links["src5_to_node3"]],
                [links["node3_to_node4"]], [1], self.node_type),
            4: Node(4, [6*n_packets],
                [links["node2_to_node4"], links["node3_to_node4"]],
                [links["node4_to_node5"]], [1], self.node_type),
            5: Node(5, [3*n_packets,3*n_packets],
                [links["node4_to_node5"], links["node4_to_node5"]],
                [links["node5_to_dst0"], links["node5_to_dst1"]], [0, 0], self.node_type)}

    return links, srcs, dsts, nodes

  def topology3(self):
    links = {"src0_to_node0": deque(),
              "src2_to_node0": deque(),
              "node0_to_dst0": deque()}
    srcs = {0: Source_3(src_id = 0, output = links["src0_to_node0"], 
                      hop = 2, route = [[0,0]], scenario = [1], delay = 1),
            1: Source_3(1, links["src0_to_node0"], 2, [[0,0]], [0, 1], 7),
            2: Source_3(2, links["src2_to_node0"], 2, [[0,1]], [1], 3)}
    dsts = {0: Destination(0, links["node0_to_dst0"])}
    nodes = {0: Node(0, [3],
                [links["src0_to_node0"], links["src2_to_node0"]],
                [links["node0_to_dst0"]], [1], self.node_type)}

    return links, srcs, dsts, nodes

  def topology4(self):
    links = {"src0_to_node0": deque(),
                  "src1_to_node1": deque(),
                  "src2_to_node1": deque(),
                  "src3_to_node2": deque(),
                  "src4_to_node6": deque(),
                  "src5_to_node7": deque(),
                  "src6_to_node7": deque(),
                  "src7_to_node8": deque(),
                  "node0_to_node3": deque(),
                  "node1_to_node0": deque(),
                  "node1_to_node2": deque(),
                  "node2_to_node5": deque(),
                  "node3_to_node4": deque(),
                  "node4_to_node1": deque(),
                  "node4_to_node7": deque(),
                  "node5_to_node4": deque(),
                  "node6_to_node3": deque(),
                  "node7_to_node6": deque(),
                  "node7_to_node8": deque(),
                  "node8_to_node5": deque(),
                  "node8_to_dst0": deque(),
                  "node3_to_dst1": deque(),
                  "node5_to_dst2": deque(),
                  "node6_to_dst3": deque(),
                  "node2_to_dst4": deque(),
                  "node3_to_dst5": deque(),
                  "node5_to_dst6": deque(),
                  "node0_to_dst7": deque()}

    srcs = {0: Source(src_id = 0, output = links["src0_to_node0"], 
                  slot_length = 20, num = 10, hop = 6, route = [[0,0], [0,0], [1,0], [1,1], [1,1]]),
                1: Source(1, links["src1_to_node1"], 20, 10, 4, [[0,0], [0,1], [1,0]]),
                2: Source(2, links["src1_to_node1"], 20, 10, 4, [[1,0], [0,1], [1,0]]),
                3: Source(3, links["src3_to_node2"], 20, 10, 6, [[0,0], [0,0], [1,1], [0,1], [1,1]]),
                4: Source(4, links["src4_to_node6"], 20, 10, 6, [[0,0], [0,1], [0,0], [1,1], [1,1]]),
                5: Source(5, links["src5_to_node7"], 20, 10, 4, [[0,0], [0,1], [1,1]]),
                6: Source(6, links["src5_to_node7"], 20, 10, 4, [[1,0], [0,1], [1,1]]),
                7: Source(7, links["src7_to_node8"], 20, 10, 6, [[0,0], [0,1], [0,1], [0,1], [1,1]])}
    dsts = {0: Destination(0, links["node8_to_dst0"]),
            1: Destination(1, links["node3_to_dst1"]),
            2: Destination(2, links["node5_to_dst2"]),
            3: Destination(3, links["node6_to_dst3"]),
            4: Destination(4, links["node2_to_dst4"]),
            5: Destination(5, links["node3_to_dst5"]),
            6: Destination(6, links["node5_to_dst6"]),
            7: Destination(7, links["node0_to_dst7"])}

    nodes = {0: Node(0, [20, 10],
              [links["src0_to_node0"], links["node1_to_node0"]],
              [links["node0_to_node3"], links["node0_to_dst7"]], [1,0], self.node_type),
            1: Node(1, [20, 20],
              [links["src1_to_node1"], links["node4_to_node1"]],
              [links["node1_to_node0"], links["node1_to_node2"]], [1,1], self.node_type),
            2: Node(2, [20, 10],
              [links["src3_to_node2"], links["node1_to_node2"]],
              [links["node2_to_node5"], links["node2_to_dst4"]], [1,0], self.node_type),
            3: Node(3, [20, 20],
              [links["node0_to_node3"], links["node6_to_node3"]],
              [links["node3_to_node4"], links["node3_to_dst1"]], [1,1], self.node_type),
            4: Node(4, [20, 20],
              [links["node3_to_node4"], links["node5_to_node4"]],
              [links["node4_to_node1"], links["node4_to_node7"]], [1,1], self.node_type),
            5: Node(5, [20, 20],
              [links["node2_to_node5"], links["node8_to_node5"]],
              [links["node5_to_node4"], links["node5_to_dst2"]], [1,1], self.node_type),
            6: Node(6, [20, 10],
              [links["src4_to_node6"], links["node7_to_node6"]],
              [links["node6_to_node3"], links["node6_to_dst3"]], [1,0], self.node_type),
            7: Node(7, [20, 20],
              [links["src5_to_node7"], links["node4_to_node7"]],
              [links["node7_to_node6"], links["node7_to_node8"]], [1,1], self.node_type),
            8: Node(8, [20, 10],
              [links["src7_to_node8"], links["node7_to_node8"]],
              [links["node8_to_node5"], links["node8_to_dst0"]], [1,0], self.node_type)}
        
    return links, srcs, dsts, nodes