from numpy import matlib
from Global_vars import Global_vars
from Model_DQN import Model_DQN
from BWK import BWK
from FLO import FLO
from GOA import GOA
from NRO import NRO
from Obj import Obj_fun_RL
from Plot_Results import plot_Results, plot_Fitness
from Proposed import Proposed
import numpy as np


# Initialization
an = 0
if an == 1:
    No_of_Server = [100, 120, 140, 160, 180]
    No_of_VM = [500, 600, 700, 800, 900]
    no_of_tasks = [50, 100, 150, 200, 250]
    Pmax = 215  # maximum power(W)
    VC_limit = [1, 98]  # CPU
    VM_limit = [0, 80]  # GB Ram
    PC_Value = 500  # CPU
    PM_Value = 500  # GB Ram
    Var = []  # List of dictionaries
    for i in range(len(no_of_tasks)):
        VC = np.random.randint(low=VC_limit[0], high=VC_limit[1], size=[1, No_of_VM[i]])  # VM - CPU
        VM = np.random.randint(low=VM_limit[0], high=VM_limit[1], size=[1, No_of_VM[i]])  # VM - Memory
        PC = PC_Value * np.ones(No_of_Server[i])  # Server - CPU
        PM = PM_Value * np.ones(No_of_Server[i])  # Server - Memory
        BW = np.random.randint(5, 50, size=No_of_VM[i])  # VM - Bandwidth
        Active_Servers = np.random.randint(No_of_Server[i], size=[No_of_VM[i]])

        # Use dictionary for each task group
        Var.append({
            'Active_Servers': Active_Servers,
            'No_of_Server': No_of_Server[i],
            'No_of_VM': No_of_VM[i],
            'no_of_tasks': no_of_tasks[i],
            'VC_limit': VC_limit,
            'VM_limit': VM_limit,
            'PC_Value': PC_Value,
            'PM_Value': PM_Value,
            'VC': VC,
            'VM': VM,
            'PC': PC,
            'PM': PM,
            'Pmax': Pmax,
            'BW': BW
        })
    # Save to file
    np.save('Var.npy', Var)



# Optimization for Resource Allocation
an = 0
if an == 1:
    Bestsol = []
    Fitness = []
    Var = np.load('Var.npy', allow_pickle=True)
    for n in range(len(Var)):
        Info = Var[n]
        C = Var[n].Active_Servers  # active  server  after placement and used for migration
        Npop = 10
        Chlen = Var[n].no_of_tasks + 1 # No.of Task
        xmin = matlib.repmat([np.ones((Npop, Chlen-1))], Npop, 100)
        xmax = matlib.repmat([np.ones((Npop, Chlen-1))* Var[n].no_of_tasks], Npop, 1000)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = Obj_fun_RL
        max_iter = 250

        Global_vars.Chlen = Chlen
        Global_vars.Info = Var[n]
        Global_vars.C = C

        print('NRO....')
        [bestfit1, fitness1, bestsol1, Time1] = NRO(initsol, fname, xmin, xmax, max_iter)

        print('FLO....')
        [bestfit2, fitness2, bestsol2, Time2] = FLO(initsol, fname, xmin, xmax, max_iter)

        print('GOA....')
        [bestfit3, fitness3, bestsol3, Time3] = GOA(initsol, fname, xmin, xmax, max_iter)

        print('BWK....')
        [bestfit4, fitness4, bestsol4, Time4] = BWK(initsol, fname, xmin, xmax, max_iter)

        print('Proposed....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)

        Bestsol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        Fitness.append([fitness1, fitness2, fitness3, fitness4, fitness5])
    np.save('Bestsol.npy', Bestsol)
    np.save('Fitness.npy', Fitness)

# Allocated Resource
an = 0
if an == 1:
    TDMA_Rewards = []
    Agent_Rewards = []
    No_of_Task =[50, 100, 150, 200, 250]
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    for n in range(len(No_of_Task)):
        Episode_Reward = []
        Maximum_Reward = []
        an.no_of_tasks = No_of_Task[n]
        Bestsol = BestSol[n]
        for i in range(Bestsol.shape[0]):
            bestsol = Bestsol[i]
            C.memory_size = bestsol[0]  # learning rate for actor
            C.replace_target_iter = bestsol[1]  # learning rate for critic
            C.batch_size = bestsol[2]  # reward discount
            C.learning_rate = bestsol[3]  # soft replacement
            sol = np.reshape(bestsol[4:], (10, len(bestsol[4:]) // 10))
            max_iter = sol[1]
            agent_reward_list, TDMA_reward_list, Total_reward = Model_DQN(max_iter, sol)
            Episode_Reward.append(agent_reward_list)
            Maximum_Reward.append(TDMA_reward_list)
        TDMA_Rewards.append(Episode_Reward)
        Agent_Rewards.append(Maximum_Reward)
    np.save('Reward.npy', Agent_Rewards)

plot_Results()
plot_Fitness()