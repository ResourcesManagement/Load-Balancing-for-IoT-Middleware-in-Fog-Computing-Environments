import numpy as np


def Obj_fun_RL(Soln):
    Datas = Glob_Vars.Info  # Assuming this is your global dataset
    Soln = np.array(Soln)

    if Soln.ndim == 2:
        Fitn = np.zeros(Soln.shape[0])
        for i in range(Soln.shape[0]):
            sol = Soln[i]

            # Assign each variable from sol
            Scalability = np.mean(Datas) * sol[0]
            Security = np.var(Datas) * sol[1] + 1  # +1 to avoid division by zero
            ServingDelay = (np.mean(Datas) * 0.3) * sol[2]
            Time = (np.mean(Datas) * 0.5) * sol[3]
            EnergyConsumption = (np.mean(Datas) * 0.4) * sol[4]  # you can adjust coefficient
            Penalty = abs(np.max(Datas) - np.min(Datas)) * sol[5]  # if 6th parameter exists

            Fitness = (1 / (Scalability + Security)) + ServingDelay + Time + EnergyConsumption + Penalty
            Fitn[i] = Fitness

        return Fitn

    else:
        sol = Soln
        Scalability = np.mean(Datas) * sol[0]
        Security = np.var(Datas) * sol[1] + 1
        ServingDelay = (np.mean(Datas) * 0.3) * sol[2]
        Time = (np.mean(Datas) * 0.5) * sol[3]
        EnergyConsumption = (np.mean(Datas) * 0.4) * sol[4]
        Penalty = abs(np.max(Datas) - np.min(Datas)) * sol[5]

        Fitness = (1 / (Scalability + Security)) + ServingDelay + Time + EnergyConsumption + Penalty
        return Fitness
