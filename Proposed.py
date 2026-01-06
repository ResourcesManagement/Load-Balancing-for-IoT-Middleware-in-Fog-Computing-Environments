
import time
import numpy as np


def Proposed(XPos, fobj, lb, ub, T):
    pop, dim = XPos.shape

    lb = np.array(lb) * np.ones(dim)
    ub = np.array(ub) * np.ones(dim)

    p = 0.9
    XFit = np.array([fobj(XPos[i, :]) for i in range(pop)])
    Convergence_curve = np.zeros(T)

    ct = time.time()

    # ---------------- Main loop ----------------
    for t in range(T):  # t = 1 to maxiter

        # Sort population
        sorted_indexes = np.argsort(XFit)
        XLeader_Pos = XPos[sorted_indexes[0], :]
        XLeader_Fit = XFit[sorted_indexes[0]]

        WorstFit = np.max(XFit)

        # ---------------- Population loop ----------------
        for i in range(pop):  # i = 1 to Npop

            CurrentFit = XFit[i]
            r = CurrentFit / (CurrentFit + WorstFit + 1e-10)                # PROPOSED Updation

            # -------- Attacking behavior --------
            n = 0.05 * np.exp(-2 * (t / T) ** 2)

            if p < r:
                XPosNew = XPos[i, :] + n * (1 + np.sin(r)) * XPos[i, :]
            else:
                XPosNew = XPos[i, :] * (n * (2 * np.random.rand(dim) - 1) + 1)

            XPosNew = np.clip(XPosNew, lb, ub)

            XFit_New = fobj(XPosNew)
            if XFit_New < XFit[i]:
                XPos[i, :] = XPosNew
                XFit[i] = XFit_New

            # -------- Migration behavior --------
            m = 2 * np.sin(r + np.pi / 2)
            s = np.random.randint(0, pop)

            ori_value = np.random.rand(dim)
            cauchy_value = np.tan((ori_value - 0.5) * np.pi)

            if XFit[i] < XFit[s]:
                XPosNew = XPos[i, :] + cauchy_value * (XPos[i, :] - XLeader_Pos)
            else:
                XPosNew = XPos[i, :] + cauchy_value * (XLeader_Pos - m * XPos[i, :])

            XPosNew = np.clip(XPosNew, lb, ub)

            XFit_New = fobj(XPosNew)
            if XFit_New < XFit[i]:
                XPos[i, :] = XPosNew
                XFit[i] = XFit_New

        # ---------------- Best solution ----------------
        best_idx = np.argmin(XFit)
        Convergence_curve[t] = XFit[best_idx]

    ct = time.time() - ct

    return XFit[best_idx], XPos[best_idx, :], Convergence_curve, ct
