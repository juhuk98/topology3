from typing import Any
import numpy as np
from dataclasses import dataclass
import warnings
from collections import deque

warnings.filterwarnings('ignore')


def number_to_action(action_id, n):  # number -> binary gcl code
  bit = '0'+str(int(n))+'b'
  b_id = format(action_id, bit)
  action_ = np.array(list(map(int, b_id)))
  return action_


@dataclass
class Packet:
  type: int = None # 몇번째 flow
  num: int = None # 몇번째 packet
  generated_time: float = None # 생성시간
  hop: int = None # 남아있는 hop 수 - 계속 변하는 값
  route: np.ndarray = None # 환경
  node_arrival_time: int = None # FIFO - 강화학습 X

class Node:
  def __init__(self, node_id, count, input, output, module_type, node_type=0):
    self.node_id = node_id
    self.input = input
    self.output = output
    self.num_port = len(output)
    self.num_queue = len(input)
    self.queues = [deque() for _ in range(self.num_port*self.num_queue)]
    self.queues[0].append(0)
    self.queues = np.reshape(self.queues, [self.num_port, self.num_queue])
    self.queues[0,0].popleft()
    self.count_init = count
    self.count = count
    self.eds = np.zeros(self.num_port) # 추정지연 시간
    self.empty = np.zeros(self.num_port) # 비어있는지
    self.module_type = module_type
    self.node_type = node_type

# state를 가져옴 - state_size : 큐 개수 * 4 * port 수
# 행 : 포트, 열 : 큐
  def get_state(self, time):
    state = np.zeros((sum(self.module_type), self.num_queue * 4))
    for i in range(self.num_port):
      if self.module_type[i]:
        for j in range(self.num_queue):
          ql = len(self.queues[i, j])
          state[i, j*4] = ql
          if ql:
            packet = self.queues[i, j][0]
            state[i, j*4+1] = time - packet.generated_time
            state[i, j*4+2] = packet.hop
            max_ed = 0
            for k in range(ql):
              packet_ = self.queues[i, j][k]
              ed = (time - packet_.generated_time
                    + packet_.hop + k)
              if max_ed < ed:
                max_ed = ed
            state[i, j*4+3] = max_ed
    return np.reshape(state, (1,-1))

# print해서 제대로 보냈는지 확인
  def receive(self, time):
    for i in range(self.num_queue):
      while len(self.input[i]):
        packet = self.input[i].popleft()
        packet.node_arrival_time = time
        packet.hop -= 1
        r_ = packet.route[::-1][packet.hop-1]
        self.queues[r_[0], r_[1]].append(packet)
        self.empty[r_[0]] += 1
  
  def send(self, action, time):
    if self.node_type == 0:
      self.rl_node(action, time)
    elif self.node_type == 1:
      self.fifo(time)
    elif self.node_type == 2:
      self.roundrobin(time)
    elif self.node_type == 3:
      self.heuristic(time)
    elif self.node_type == 4:
      self.n_fifo(time)

  def rl_node(self, action, time):
    a_list = number_to_action(int(action), sum(self.module_type))
    a_idx = 0
    for i in range(self.num_port):
      if self.module_type[i]:
        a = a_list[a_idx]
        if len(self.queues[i, a]):
          packet = self.queues[i, a].popleft()
          self.output[i].append(packet)
          self.count[i] -= 1
          self.empty[i] -= 1
          recent_ed = time - packet.generated_time + packet.hop
          self.eds[i] = (recent_ed)
        a_idx += 1
      else:
        for j in range(self.num_queue):
          if len(self.queues[i, j]):
            packet = self.queues[i, j].popleft()
            self.output[i].append(packet)
            self.count[i] -= 1
            self.empty[i] -= 1

  def fifo(self, time):
    for i in range(self.num_port):
      if self.module_type[i]:
        first = 0
        first_t = 1000
        for j in range(self.num_queue):
          if len(self.queues[i, j]):
            t_ = self.queues[i, j][0].node_arrival_time
            if first_t > t_:
              first_t = t_
              first = j
            if first_t == t_:
              if np.random.rand()<0.5:
                first_t = t_
                first = j
        if len(self.queues[i, first]):
          packet = self.queues[i, first].popleft()
          self.output[i].append(packet)
          self.count[i] -= 1
          self.empty[i] -= 1
          recent_ed = time - packet.generated_time + packet.hop
          self.eds[i] = (recent_ed)
      else:
        for j in range(self.num_queue):
          if len(self.queues[i, j]):
            packet = self.queues[i, j].popleft()
            self.output[i].append(packet)
            self.count[i] -= 1
            self.empty[i] -= 1

  def roundrobin(self, time):
    for i in range(self.num_port):
      if self.module_type[i]:
        for j in range(self.num_queue):
          self.order[i] = (self.order[i]+1)
          if self.order[i] == self.num_queue: self.order[i]=0
          if len(self.queues[i, self.order[i]]):
            packet = self.queues[i, self.order[i]].popleft()
            self.output[i].append(packet)
            self.count[i] -= 1
            self.empty[i] -= 1
            recent_ed = time - packet.generated_time + packet.hop
            self.eds[i] = (recent_ed)
            break
      else:
        for j in range(self.num_queue):
          if len(self.queues[i, j]):
            packet = self.queues[i, j].popleft()
            self.output[i].append(packet)
            self.count[i] -= 1
            self.empty[i] -= 1

  def heuristic(self, time):
    for i in range(self.num_port):
      if self.module_type[i]:
        max_ed = 0
        q_index = 0
        for j in range(self.num_queue):
          ql = len(self.queues[i, j])
          if ql:
            for k in range(ql):
              packet_ = self.queues[i, j][k]
              ed = (time - packet_.generated_time
                    + packet_.hop + k)
              if max_ed < ed:
                max_ed = ed
                q_index = j
        if len(self.queues[i, q_index]):
          packet = self.queues[i, q_index].popleft()
          self.output[i].append(packet)
          self.count[i] -= 1
          self.empty[i] -= 1
          recent_ed = time - packet.generated_time + packet.hop
          self.eds[i] = (recent_ed)
      else:
        for j in range(self.num_queue):
          if len(self.queues[i, j]):
            packet = self.queues[i, j].popleft()
            self.output[i].append(packet)
            self.count[i] -= 1
            self.empty[i] -= 1

  def n_fifo(self, time):
    for i in range(self.num_port):
      if self.module_type[i]:
        first = 0
        first_t = 1000
        for j in range(self.num_queue):
          if len(self.queues[i, j]):
            t_ = self.queues[i, j][0].generated_time
            if first_t > t_:
              first_t = t_
              first = j
            if first_t == t_:
              if np.random.rand()<0.5:
                first_t = t_
                first = j
        if len(self.queues[i, first]):
          packet = self.queues[i, first].popleft()
          self.output[i].append(packet)
          self.count[i] -= 1
          self.empty[i] -= 1
          recent_ed = time - packet.generated_time + packet.hop
          self.eds[i] = (recent_ed)
      else:
        for j in range(self.num_queue):
          if len(self.queues[i, j]):
            packet = self.queues[i, j].popleft()
            self.output[i].append(packet)
            self.count[i] -= 1
            self.empty[i] -= 1


class Source:

  def __init__(self, src_id, output, slot_length, num, hop, route, scenario=None):
    self.src_id = src_id
    self.output = output
    self.hop = hop
    self.num = num
    #np.random.seed(seed)
    self.scenario = np.zeros(slot_length)
    self.scenario[np.random.choice(slot_length, num, replace = False)] = 1
    if not scenario == None:
      self.scenario = scenario
    #print(self.scenario)
    self.route = route
        
  def generate_packet(self, time):
    p = Packet()
    p.type = self.src_id
    p.generated_time = time
    p.hop = self.hop
    p.route = self.route
    return p

# 패킷 보냄
  def send(self, time):
    if time<len(self.scenario):
      if self.scenario[time]:
        packet = self.generate_packet(time)
        self.output.append(packet)
            
# 패킷받고 delay 계산
class Destination:

    def __init__(self, dst_id, input):
      self.dst_id = dst_id
      self.input = input
      self.delay = np.array([])
        
    def receive(self, time):
      if len(self.input):
        packet = self.input.popleft()
        self.delay = np.append(self.delay, time - packet.generated_time)

#topology 3 전용 source
class Source_3:

  def __init__(self, src_id, output, hop, route, scenario, delay):
    self.src_id = src_id
    self.output = output
    self.hop = hop
    self.route = route
    self.scenario = scenario
    self.delay = delay

  def generate_packet(self, time):
    p = Packet()
    p.type = self.src_id
    p.generated_time = time-self.delay
    p.hop = self.hop
    p.route = self.route
    return p
    
  def send(self, time):
    if time<len(self.scenario):
      if self.scenario[time]:
        packet = self.generate_packet(time)
        self.output.append(packet)