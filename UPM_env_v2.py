# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 23:41:29 2021

@author: 廖淯舜
"""

import simpy
import numpy as np
import pandas as pd
import scipy.stats
import copy
from GanttPlot import Gantt
import sys


NUM_MACHINES = 3

#job info and setup time dataframe
#job_data = pd.read_csv("instance\\job_info_1.csv",index_col = 0)
#st_data = pd.read_csv("instance\\setup_time_1.csv",index_col = 0)

#debug data
job_data = pd.read_csv("instance\\debug_data\\job_info_1.csv",index_col = 0)
st_data = pd.read_csv("instance\\debug_data\\setup_time_1.csv",index_col = 0)

#|Type AT PT DD| Transform df to np.array then to list
ORDER_DATA = np.array(job_data).tolist()
#Sorting ORDER_DATA by AT
ORDER_DATA.sort(key=lambda x:x[1])


SETUP_TIME = np.array(st_data).tolist()

print(ORDER_DATA)

NUM_ACTIONS = 5 # SPT EDD MST CR LST

#np.random.seed(0)
class Order:
    def __init__(self, fac, ID, Type, AT, PT, DD):
        self.fac = fac
        #attribute
        self.ID = ID
        self.Type = Type
        self.PT = PT
        self.enter_system = AT #arrival time
        self.ST = None #start time
        self.leave_system = None #finish time
        self.DD = DD
 

class Source:
    def __init__(self, fac):
        self.fac = fac
        self.env  = fac.env
        
        #statistics
        self.output = 0
        
        #initial process.
        self.env.process(self.arrival())
        
    def set_port(self):
        #reference
        self.queue = self.fac.queue   
        
    def arrival(self):
        for i in range(len(ORDER_DATA)):
            #generate order
            inter_AT = ORDER_DATA[i][1] - ORDER_DATA[i-1][1] if i >= 1 else ORDER_DATA[i][1]
            
            if inter_AT != 0:
                yield self.env.timeout(inter_AT)
                
            
            order = Order(self.fac, ORDER_DATA[i][0]-1, ORDER_DATA[i][0], ORDER_DATA[i][1], ORDER_DATA[i][2], ORDER_DATA[i][3])
            self.output += 1

            print("{} : order {} arrive. Type: {}|DD: {}".format(round(self.env.now,2), order.ID, order.Type, order.DD))
            
            #change order state to 1-waiting
            self.fac.update_state_m1(order.ID, 1, self.env.now)
            
            #send order to queue
            self.queue.append(order)
            
            if i+1 != len(ORDER_DATA):
                if (ORDER_DATA[i][1] == ORDER_DATA[i+1][1]):
                    continue
                
            
            self.queue.check_machine()

        

class Queue:
    def __init__(self, fac):
        self.fac = fac
        #attribute
        self.space = []
        self.entity_type_now = np.zeros((NUM_MACHINES,), dtype=np.int32)
        
    def set_port(self, machine_list):
        #reference
        self.env = self.fac.env
        self.machines = machine_list
        
        
        
    def append(self, order):
        self.space.append(order)
        
    def check_machine(self):
        for i in range(len(self.machines)):
            if self.machines[i].state == "idle":
                self.get_order(i)
                
        
    def get_order(self, i):
        if len(self.space) > 0:
            if len(self.space) == 1:
                 order = self.space[0]
                 self.space.pop(0)
                 
                 #record current order type 
                 self.entity_type_now[i] = order.Type
                 self.machines[i].process_order(order)
            else:
                self.fac.decision_point.succeed()
                self.fac.decision_point = self.env.event()
                self.env.process(self.wait_for_action(i))
            
    def wait_for_action(self, i):
        yield self.fac.get_action
        if len(self.space) > 0:
            #sorting orders in queue by given action
            self.sort_queue(self.fac.dispatcher.action, i)
            
            order = self.space[0]
            self.space.pop(0)
            
            # compute_reward
            self.fac.compute_reward(self.env.now, order.ID)
            
            #record current order type 
            self.entity_type_now[i] = order.Type
            self.machines[i].process_order(order)
        
    def sort_queue(self, dp_rule, machine_id):
        #dp_rule: dispatching rule
        
        if   dp_rule == 0:   #SPT
            self.space.sort(key = lambda entity : entity.PT)
        elif dp_rule == 1: #EDD
            self.space.sort(key = lambda entity : entity.DD)
        elif dp_rule == 2: #MST
            a = [i for i in self.space if i.Type == self.entity_type_now[machine_id]]
            b = [i for i in self.space if i.Type != self.entity_type_now[machine_id]]
            self.space = np.concatenate((a, b), axis=None).tolist()
        elif dp_rule == 3: #LST
            self.space.sort(key = lambda entity : entity.DD - entity.PT)
        elif dp_rule == 4: #CR
            self.space.sort(key = lambda entity : entity.DD / entity.PT)
        
            
        print('action:{}, queue:{}'.format(dp_rule, [o.ID for o in self.space]))


class Machine:
    def __init__(self, fac, ID):
        self.fac = fac
        #attribute
        self.state = "idle"
        self.ID = ID
        self.prev_order_type = 0 #previous order type. default=0
        self.MAT = 0             #Machine Available Time
        self.last_st = 0         #last setup time
        self.using_time = 0

        
    def set_port(self):
        #reference
        self.env = self.fac.env
        self.source = self.fac.source
        self.queue = self.fac.queue
        self.sink = self.fac.sink
        
    def check_arrival(self):
        '''
        This function aims to solve confliction of 
        MC start event and arrival event.
        The priority of events:
        MC completion > Job(Order) arrival > MC get orders
        '''
        cur_num = self.source.output - 1
        
        while(cur_num < len(ORDER_DATA)):
            if ORDER_DATA[cur_num][1] == self.env.now:
                self.fac.arrival_over = False
                break
            elif ORDER_DATA[cur_num][1] > self.env.now:
                self.fac.arrival_over = True
            cur_num += 1
        
    def process_order(self, order):
        self.state = order
        
        print("{} : order {} start processing at machine {}".format(round(self.env.now,2), order.ID, self.ID))
        order.ST = self.env.now
        self.process = self.env.process(self._process_order_callback(order))
        
    def _process_order_callback(self, order):
        process_time = order.PT
        #need setup?
        if self.prev_order_type != order.Type:
            if self.prev_order_type == 0 or self.prev_order_type == order.Type:
                self.fac.gantt_plot.update_gantt(self.ID, self.env.now, order.PT, order.ID)
            else:
                process_time = SETUP_TIME[self.prev_order_type - 1][order.Type - 1] + order.PT
                self.fac.gantt_plot.update_gantt(self.ID, self.env.now, SETUP_TIME[self.prev_order_type - 1][order.Type - 1], 100)
                self.fac.gantt_plot.update_gantt(self.ID, self.env.now + SETUP_TIME[self.prev_order_type - 1][order.Type - 1], order.PT, order.ID)
                
        
#        self.fac.gantt_plot.draw_gantt(self.env.now)
        
        self.MAT = self.env.now + process_time
        
        #update state t
        self.fac.update_state_m1(order.ID, 2, self.env.now) 
        self.fac.update_state_m2(self.ID, self.prev_order_type, order.Type, self.MAT)
        
        #processing order for PT or PT+SETUP_TIME mins
        yield self.env.timeout(process_time)
        
        print("{} : order {} finish processing".format(round(self.env.now,2), order.ID))
        
        #change state
        self.state = "idle"
        #send order to sink
        self.sink.complete_order(order)
        #update previous order type to current one
        self.prev_order_type = order.Type
        
        self.MAT = 0
        self.using_time += order.PT
        #update state t+1
        self.fac.update_state_m1(order.ID, -1, self.env.now)
        self.fac.update_state_m2(self.ID, self.prev_order_type, 0, self.MAT)
        self.fac.update_state_mat()
        
        # check if any arrival event generate when MC complete
        self.check_arrival()
        
        if self.fac.arrival_over:
            #get next order in queue
            yield self.env.timeout(0)
            self.queue.check_machine()
        else:
            pass

        
        

class Sink:
    def __init__(self, fac):
        self.fac = fac
        self.orders = []
        
    def set_port(self):
        #reference
        self.env = self.fac.env
        
    def complete_order(self, order):
        self.orders.append(order)
        order.leave_system = self.env.now
        self.fac.throughput += 1
        
        
        if len(self.orders) >= len(ORDER_DATA):
            #the time of the last order leaving system equals to makespan
            self.fac.makespan = self.env.now
            self.fac.decision_point.succeed()
            self.fac.terminal.succeed()

class Dispatcher:
    def __init__(self, fac):
        self.action = None
        self.fac = fac
    
    def assign_action(self, action):
        self.fac.get_action.succeed()
        self.fac.get_action = self.fac.env.event()
        assert action in np.arange(NUM_ACTIONS)
        self.action = action

class Factory:
    def __init__(self):
        #statistics
        self.makespan = float('inf')
        self.throughput = 0
        
        self.current_util = 0
        self.last_util    = 0
        self.arrival_over = False
        
    def build(self):      
        #build
        self.env        = simpy.Environment()
        self.dispatcher = Dispatcher(self)
        self.source     = Source(self)
        self.queue      = Queue(self)
        self.machines   = {"M1-1" : Machine(self, ID=1),
                           "M1-2" : Machine(self, ID=2),
                           "M1-3" : Machine(self, ID=3)}

        self.sink       = Sink(self)
        
        #set_port
        self.source.set_port()
        self.queue.set_port([self.machines["M1-1" ],self.machines["M1-2" ],self.machines["M1-3" ]])
        self.machines["M1-1" ].set_port()
        self.machines["M1-2" ].set_port()
        self.machines["M1-3" ].set_port()
        self.sink.set_port()
        
        #decision event
        self.decision_point = self.env.event()
        
        #terminal event
        self.terminal       = self.env.event()
        
        #get action event
        self.get_action     = self.env.event()
        
        self.observation    = self.get_initial_state()
        
        self.reward = 0
        
        #Gantt chart
        self.gantt_plot = Gantt()
        
    def get_utilization(self):
        #compute average utiliztion of machines
        total_using_time = 0
        for _, machine in self.machines.items():
            total_using_time += machine.using_time

        if self.env.now != 0:
            avg_using_time = total_using_time / NUM_MACHINES
            return avg_using_time / self.env.now
        else:
            return 0
    
    def get_state(self):
        return copy.deepcopy(self.observation)
        
    
    def get_reward(self):
        self.current_util = self.get_utilization()
        last_util    = self.last_util
        reward       = (self.current_util - last_util)
        
         ###TO DO:###
#        Compare result to optimal makespan
         ############
        
        
        #record current utilization as last utilization
        self.last_util = self.current_util
        
        #final state
        if self.terminal:
            makespan     = self.makespan
            lower_bound  = np.sum(ORDER_DATA, axis = 0)[2] / NUM_MACHINES
            
            if makespan == lower_bound:
                reward += 100
            else:
                reward += 100 / (makespan - lower_bound)
                
        return reward
    
    def reset(self):
        self.build()
        self.env.run(until = self.decision_point)

        #reset statistics
        self.throughput = 0
        self.makespan   = float('inf')
        
        self.current_util = 0
        self.last_util  = 0

        initial_state = self.get_state()
        
        return initial_state
    
    def step(self, action):
        self.dispatcher.assign_action(action)
        self.env.run(until = self.decision_point)
        
        state = self.get_state()
        reward = self.get_reward()
        done = self.terminal.triggered
        
        info = 0
        if done:
            info = self.env.now
        
        self.reset_reward()
        return state, reward, done, info
    
    def get_initial_state(self):
        # Job Processing Status (n jobs * 6)
        matrix_1 = np.zeros((len(ORDER_DATA), 6), dtype = np.float32)
        origin_ORDER_DATA = np.array(job_data).tolist()
        # Machine Status (m machines * 4)
        matrix_2 = np.zeros((NUM_MACHINES, 4), dtype = np.float32)
        # Setup Time (num_types * num_types)
        matrix_3 = np.array(SETUP_TIME, dtype = np.float32)
        
        
        for i in range(len(matrix_1)):
            #Type, AT, PT, DD
            matrix_1[i][:len(ORDER_DATA[i])] = origin_ORDER_DATA[i]

        return [matrix_1, matrix_2, matrix_3]
    
    def update_state_m1(self, order_id, order_state, time):
        #update order state: 1-waiting|-1-processed|0-otherwise
        self.observation[0][order_id,4] = order_state
        #update T_now
        self.observation[0][:,5] = time
        
#        self.print_state()
        
              
    def update_state_m2(self, machine_id, prev_order_type, cur_order_type, MAT):
        self.observation[1][machine_id-1,:-1] = [prev_order_type, cur_order_type, MAT]
        
#        self.print_state()
        
    def print_state(self):
        print("state matrix 1:\n{}".format(self.observation[0]))
        print("state matrix 2:\n{}".format(self.observation[1]))
    
    #compute last setup time til now
    def update_state_mat(self):
        for m in self.machines:
            self.observation[1][self.machines[m].ID-1,-1] = self.env.now - self.machines[m].last_st
            
    #reward method
    def compute_reward(self, start_process_t, job_id):
        current_util = self.get_utilization()
        last_util    = self.last_util
        reward       = (current_util - last_util)
        
        #final state
        if self.terminal:
            makespan     = self.makespan
            lower_bound  = np.sum(ORDER_DATA, axis = 0)[2] / NUM_MACHINES
            
            if makespan == lower_bound:
                reward += 100
            else:
                reward += 100 / (makespan - lower_bound)

        
        #record current utilization as last utilization
        self.last_util = current_util
#        print("Reward = {}".format(round(reward,2)))
        return reward

    def reset_reward(self):
        self.reward = 0
        
         

def mean_confidence_interval(a, confidence=0.95):
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return  round(m-h,1), round(m+h,1)

if __name__ == "__main__":  
#    sys.stdout = open("console_output.txt", "w")
    
    fac = Factory()
    state = fac.reset()
    
    while True:
        action = 0 #0:SPT
        next_state, reward, done, info = fac.step(action)
#        print("state:\n{}\naction:\n{}\nreward:\n{}".format(state, action, reward))
        state = next_state
        
        if done:
            break
    fac.gantt_plot.draw_gantt(fac.makespan)
    print("-----Makespan = {}-----".format(fac.makespan))
    
#    sys.stdout.close()
    