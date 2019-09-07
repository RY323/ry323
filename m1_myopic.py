

import random
import matplotlib.pyplot as plt
import datetime
from math import floor
# from random import randrange

from random import random
from random import randint
from random import sample
from random import seed
import numpy as np
import pandas as pd
from gurobipy import *

import multiprocessing as mp


"""Function to calculate preference unmet and overtime"""
def func_pu(vars_n, solution, st, et):
    sum_p = np.array([solution[t][p][w][d] for t in range(st, et) for p in range(P) for w in range(0, len_Wa) for d in DP[p]]).sum()
    sum_t = np.array([vars_n[t]for t in range(st, et)]).sum()
    # sum_t = np.array([vars_n[t] for t in range(st, et)]).sum()
    return 1 - sum_p / sum_t

def func_o(vars_num_tw, st, et):
    return np.array([((vars_num_tw[t][d] - N)>0) * (vars_num_tw[t][d] - N) for t in range(st, et) for d in range(len_D)]).sum()

def dynamic_num(vars_x, vars_num, st):
    sum_x_xu = np.sum(vars_x, axis=0)
    for w in range(len_Wa):
        for wc in Wc:
            for d in range(0, len_D):
                try:
                    vars_num[wc - w + st][d] = vars_num[wc - w + st][d] + sum_x_xu[w][d]
                except:
                    break
    return vars_num


def optimize_tw(vars_num, vars_n):

    model = Model()

    # Add determining variables to model
    vars_x = [[[model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="x[%d,%d,%d]" % (p, w, d)) for d in range(len_D)] for w in range(len_Wa)] for p in range(P)]
    vars_o = [[model.addVar(lb=-100000, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="vars_o[%d,%d]" % (d, t)) for t in range(T)] for d in range(len_D)]
    vars_o_adj = np.array([[model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="vars_o_adj[%d,%d]" % (d, t)) for t in range(T)] for d in range(len_D)])

    # Populate A matrix
    # add constraint xu
    model.addConstrs((quicksum(vars_x[p][w]) == vars_n[p][w]) for p in range(P) for w in range(len_Wa))

    # dynamic_num
    vars_num0 = [[vars_num[t][d] for d in range(len_D)]for t in range(T)]
    vars_num0 = dynamic_num(vars_x, vars_num0, 0)

    # Populate objective

    """preference unmet"""
    pu = func_pu([vars_n], [vars_x], 0, 1)
    model.addConstr(pu <= bound_pu)


    """overtime penalty set constraint and objective function"""
    model.addConstrs((vars_o[d][t] == (vars_num0[t][d] - N)) for d in range(len_D) for t in range(T))
    model.addConstrs((vars_o_adj[d][t] == max_(vars_o[d][t], 0))for d in range(len_D) for t in range(T))
    o = (vars_o_adj.sum(axis=0)).sum()

    model.setObjective(o*10 + pu*10, 1)

    model.update()

    # Solve
    model.optimize()
    if model.status == GRB.Status.OPTIMAL:
        vars = model.getVars()
        vars_x = [[[vars[p * len_Wa * len_D + w * len_D + d].getAttr("x") for d in range(len_D)] for w in range(len_Wa)] for p in range(P)]
        return vars_x
    else:
        return False





# Put model data into dense matrices
# Wa - 11 Wc - 11 W -11

P = 5
N = 220
T = 30
Wa = [0, 1, 2, 3]
Wc = [5, 9, 13, 17, 19, 21, 23, 25, 26, 27, 28, 29]
D = [0, 1, 2, 3, 4]
DP = [[0], [1], [2], [3], [4]]
# DistributionP =[0.15, 0.25, 0.25, 0.2, 0.15]
DistributionP =[0.1, 0.32, 0.32, 0.16, 0.1]
W = list(range(0, T))



len_Wa = len(Wa)
len_Wc = len(Wc)
len_D = len(D)
len_W = len(W)


TW = 130
bound_pu= 0.2
ar_list = [30]


writer = pd.ExcelWriter('myopic.xlsx')

ST = datetime.datetime.now()

o_all = []
pu_all = []
ar_all = []
runtime_all =[]

# Optimize
for ar in ar_list:
    np.random.seed(100)
    vars_n_all = np.floor(np.array([np.random.poisson(lam=ar * 5 * DistributionP[p] / len_Wa, size=(TW, len_Wa)) for p in range(P)]).transpose((1, 0, 2)))

    tw = 0

    vars_num_tw = [[0 for d in range(len_D)] for t in range(TW + T)]
    vars_x_tw = []


    pu_tw = [0 for s in range(TW)]
    o_tw = [0 for s in range(TW)]
    pu_tw_single = [0 for s in range(TW)]
    o_tw_single = [0 for s in range(TW)]

    ST1 = datetime.datetime.now()
    while (tw <= TW - 1):

        vars_n =vars_n_all[tw]
        vars_num = vars_num_tw[tw:tw + T].copy()

        vars_x= optimize_tw(vars_num, vars_n)
        vars_x_tw.append(vars_x)

        vars_num_tw = dynamic_num(vars_x, vars_num_tw, tw)

        pu_tw_single[tw] = func_pu(vars_n_all, vars_x_tw, tw, tw + 1)
        o_tw_single[tw] = func_o(vars_num_tw, tw, tw + 1)
        tw = tw + 1


    ET1 = datetime.datetime.now()
    runtime = ET1.second - ST1.second

    pu_all.append(sum(pu_tw_single[100:])/(TW-100))
    o_all.append(sum(o_tw_single[100:])/(TW-100))
    ar_all.append(ar)
    runtime_all.append(runtime)


print(runtime_all)


ET = datetime.datetime.now()
runtime = ET - ST
print("Run time = ", ET - ST)

df = pd.DataFrame([ar_all,pu_all,o_all,runtime_all])
df.to_excel(writer,header=False, index=False)


writer.save()