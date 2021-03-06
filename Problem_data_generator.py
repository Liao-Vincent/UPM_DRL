# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 03:41:14 2021

@author: vincent_liao

This program is designed to generator RL problem data.

------------------------Problem data design------------------------
Suppose π is the number of jobs; π is the number of machines
 Setup times: π _(π_1,π_2 ) (1β€π_1,π_2β€π)
 Processing times: π_(π,π) (1β€πβ€π, 1β€πβ€π)
 Release times: π_π (1β€πβ€π)
are generated from uniform distributions: π[1,20], π[1, 100], π[1, 50π/π]. 

 Due dates: π_π (1β€πβ€π) are computed by πΈπ.1 
π_π=π_π+πΎ_π π_π  (1β€πβ€π)     (1)
where πΎ_π is a tightness factor for the due date allowance.
The problem π(12, 200, 3) involves 12 machines, 200 jobs and πΎ_π=3.
--------------------------------------------------------------------

"""

import numpy as np
import pandas as pd


def generate_jobs(n_jobs, n_mcs, K):
    #columns
    col = ["Type", "AT", "PT", "DD"]
    
    #set columns
    df1 = pd.DataFrame(columns = col)
    
    for i in range(n_jobs):
        Type            = i + 1
        Arrival_time    = np.random.randint(1, 10*n_jobs/n_mcs)
        Processing_time = np.random.randint(10, 101)
        # DD = Arrival_time + Kd * Processing_time
        due_date        = Arrival_time +  K * Processing_time
        
        job_i = [Type, Arrival_time, Processing_time, due_date]

        df1.loc[i] = job_i

        
    Setup_time      = np.random.randint(1, 21, size=(n_jobs, n_jobs))
    df2 = pd.DataFrame(Setup_time)
    
    return df1, df2

    

if __name__ == "__main__":
    #number of instances
    num_instances = 1
    #number of machines
    num_machines = 3
    #number of jobs
    num_jobs = 15
    #tightness factor for due date allowance
    K = 1
    
    for i in range(num_instances):
        job_info, setup_time = generate_jobs(num_jobs, num_machines, K)
        
        #check
        print("-----Instance {}-----".format(i+1))
        print("Job info:\n{}\nSetup time table:\n{}\n".format(job_info, setup_time))
        
        #output to excel
        out_path = "D:\\Vincent\\2021summer\\HW\\UPM_DRL\\instance\\"
        name1 = out_path + "job_info_" + str(i+1) + ".csv"
        name2 = out_path + "setup_time_" + str(i+1) + ".csv"
        job_info.to_csv(name1)
        setup_time.to_csv(name2)
    
    
    