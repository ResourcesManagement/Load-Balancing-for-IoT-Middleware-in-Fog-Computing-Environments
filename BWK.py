import time

import numpy as np


def BWK(XPos, fobj, lb, ub, T):
    """
    pop : population size
    T   : maximum iterations
    lb  : lower bound (scalar or array)
    ub  : upper bound (scalar or array)
    dim : problem dimension
    fobj: objective function
    """
    pop, dim = XPos.shape[0], XPos.shape[1]
    # Ensure bounds are arrays
    lb = np.array(lb) * np.ones(dim)
    ub = np.array(ub) * np.ones(dim)

    # ---------------- Initialization ----------------
    p = 0.9
    r = np.random.rand()

    XFit = np.array([fobj(XPos[i, :]) for i in range(pop)])

    Convergence_curve = np.zeros(T)
    ct = time.time()
    # ---------------- Main loop ----------------
    for t in range(T):

        # Sort population
        sorted_indexes = np.argsort(XFit)
        XLeader_Pos = XPos[sorted_indexes[0], :]
        XLeader_Fit = XFit[sorted_indexes[0]]

        # ---------------- Attacking & Migration ----------------
        for i in range(pop):

            # -------- Attacking behavior --------
            n = 0.05 * np.exp(-2 * (t / T) ** 2)

            if p < r:
                XPosNew = XPos[i, :] + n * (1 + np.sin(r)) * XPos[i, :]
            else:
                XPosNew = XPos[i, :] * (n * (2 * np.random.rand(dim) - 1) + 1)

            # Boundary check
            XPosNew = np.clip(XPosNew, lb, ub)

            # Fitness evaluation
            XFit_New = fobj(XPosNew)
            if XFit_New < XFit[i]:
                XPos[i, :] = XPosNew
                XFit[i] = XFit_New

            # -------- Migration behavior --------
            m = 2 * np.sin(r + np.pi / 2)
            s = np.random.randint(0, pop)
            r_XFitness = XFit[s]

            ori_value = np.random.rand(dim)
            cauchy_value = np.tan((ori_value - 0.5) * np.pi)

            if XFit[i] < r_XFitness:
                XPosNew = XPos[i, :] + cauchy_value * (XPos[i, :] - XLeader_Pos)
            else:
                XPosNew = XPos[i, :] + cauchy_value * (XLeader_Pos - m * XPos[i, :])

            # Boundary check
            XPosNew = np.clip(XPosNew, lb, ub)

            # Fitness evaluation
            XFit_New = fobj(XPosNew)
            if XFit_New < XFit[i]:
                XPos[i, :] = XPosNew
                XFit[i] = XFit_New

        # ---------------- Update best solution ----------------
        best_idx = np.argmin(XFit)
        Best_Fitness_BKA = XFit[best_idx]
        Best_Pos_BKA = XPos[best_idx, :]

        Convergence_curve[t] = Best_Fitness_BKA
    ct = time.time() - ct
    return Best_Fitness_BKA, Best_Pos_BKA, Convergence_curve, ct
