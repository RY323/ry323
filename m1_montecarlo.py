
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

# 满足所有偏好来完成xu的初始化
def ini_xu(vars_nu):
    vars_xu = np.zeros((P,len_D, T - 1, len_Wa))
    vars_nu0 = np.transpose(vars_nu,(1,0,2))
    for p in range(P):
        vars_xu[p][p] = np.copy(vars_nu0[p])
    return np.transpose(vars_xu,(2,0,3,1))

# 计算st到达的每类病人导致的加班时间
def func_o_xu(vars_num_tw,st):
    o_xu = np.zeros((len_Wa,len_D))
    o_tw = np.where(vars_num_tw>N,1,0)
    tw_list = np.array([st - w + np.array(Wc) for w in Wa])
    for w in range(len_Wa):
        try:
            o_xu[w] = np.sum(np.array([o_tw[tw] for tw in tw_list[w][np.where(T>tw_list[w])]]),axis=0)
        except:
            o_xu[w] = np.zeros(len_D)
    return o_xu

# 选择导致加班时间最长的一类病人重新安排日期
def select_xu(o_xu,vars_xu,st):
    i = 0
    while i < len_Wa * len_D:
        index = np.argsort(o_xu.flatten())[i]
        w = index // len_D
        d = index % len_D
        p = d
        if vars_xu[st][p][w][d] != 0:
            return w,d
        else:
            i+=1
    return False



# 选择导致加班时间最长的一类病人重新安排日期
def adj_xu(vars_xu,vars_num,st,w,d0):
    p = d0
    o = np.zeros(5)
    vars_xu[st][p][w][d0] -= 1
    vars_num0 = np.copy(vars_num)
    vars_num0 -= dynamic_num(vars_xu[st],np.zeros((T,len_D)),st+1)
    vars_num_all = np.array([np.copy(vars_num0) for i in range(len_D)])
    vars_xu_all = np.array([np.copy(vars_xu) for i in range(len_D)])
    for d in range(len_D):
        vars_xu0 = np.copy(vars_xu_all[d])
        vars_num0 = np.copy(vars_num_all[d])
        vars_xu0[st][p][w][d] += 1
        vars_num0 = dynamic_num(vars_xu0[st],vars_num0,st+1)
        o[d] = func_o(vars_num0,st+1,T)
        vars_xu_all[d] = vars_xu0
        vars_num_all[d] = vars_num0
    index = np.argmax(o)
    return index, vars_xu_all[index],vars_num_all[index]

def fitness_xu(vars_x,vars_n,vars_nu,vars_num):

    vars_xu = ini_xu(vars_nu)
    vars_x_xu = np.append([vars_x],vars_xu,axis=0)
    for t in range(T):
        vars_num = dynamic_num(vars_x_xu[t],vars_num,t)
    pu_num_tw = np.floor(np.sum(vars_nu,axis = (1,2)) * bound_pu)

    for t in range(T-1):
        pu_num = 0
        adj_time = 0
        while pu_num < pu_num_tw[t] and adj_time < P*len_D*len_Wa:
            o_xu = func_o_xu(vars_num,t+1)
            index_w,index_d = select_xu(o_xu,vars_xu,t)
            index,vars_xu,vars_num = adj_xu(vars_xu,vars_num,t,index_w,index_d)
            if index != index_d:
                pu_num += 1
            adj_time += 1

    vars_n_nu = np.append([vars_n],vars_nu,axis=0)
    vars_x_xu = np.append([vars_x],vars_xu,axis=0)

    return func_pu(vars_n_nu, vars_x_xu, 0, T), func_o(vars_num, 0, T)

"""Function to calculate preference unmet and overtime"""
def func_pu(vars_n, solution, st, et):
    sum_p = np.array([solution[t][p][w][d] for t in range(st, et) for p in range(P) for w in range(0, len_Wa) for d in DP[p]]).sum()
    sum_t = np.array([vars_n[t]for t in range(st, et)]).sum()
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


def optimize_tw(vars_num, vars_n,vars_nu_s,l):

    model = Model()

    # Add determining variables to model
    vars_x = [[[model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="x[%d,%d,%d]" % (p, w, d)) for d in range(len_D)] for w in range(len_Wa)] for p in range(P)]
    vars_xu_s = [[[[[model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="xu[%s,%d,%d,%d,%d]" % (s,t, p, w, d)) for d in range(len_D)] for w in range(len_Wa)] for p in range(P)] for t in range(T - 1)] for s in range(senario_nu)]
    vars_o_s = [[[model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="vars_o[%d,%d,%d]" % (s,d, t)) for t in range(T)] for d in range(len_D)] for s in range(senario_nu)]

    vars_n_nu_s = [[vars_n] + list(vars_nu_s[s]) for s in range(senario_nu)]
    vars_x_xu_s = [[vars_x] + list(vars_xu_s[s]) for s in range(senario_nu)]

    # Populate A matrix
    # add constraint xu
    model.addConstrs((quicksum(vars_x[p][w]) == vars_n[p][w]) for p in range(P) for w in range(len_Wa))
    model.addConstrs((quicksum(vars_xu_s[s][t][p][w]) == vars_nu_s[s][t][p][w]) for s in range(senario_nu) for t in range(T - 1) for p in range(P) for w in range(len_Wa))

    model.setParam('MIPGap',0.0006)
    model.setParam(GRB.Param.TimeLimit, 100.0)

    # dynamic_num
    vars_num0_s = [[[vars_num[t][d] for d in range(len_D)]for t in range(T)] for s in range(senario_nu)]
    for s in range(senario_nu):
        vars_num0 = vars_num0_s[s].copy()
        for t in range(T):
            vars_num0 = dynamic_num(vars_x_xu_s[s][t], vars_num0, t)
        vars_num0_s[s] = vars_num0.copy()

    # Populate objective
    """preference unmet"""
    pu_s = np.array([[func_pu(vars_n_nu_s[s], vars_x_xu_s[s], i, i+1) for i in range(T)]for s in range(senario_nu)])
    model.addConstrs((pu_s[s][i] <= bound_pu) for s in range(senario_nu) for i in range(T))


    """overtime penalty set constraint and objective function"""
    model.addConstrs((vars_o_s[s][d][t] >= (vars_num0_s[s][t][d] - N)) for s in range(senario_nu) for d in range(len_D) for t in range(T))
    o = np.sum(vars_o_s)

    model.setObjective(o+ np.sum(pu_s)/T/senario_nu, 1)

    model.update()
    # Solve
    model.optimize()
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.TIME_LIMIT:
        vars = model.getVars()
        vars_x = [[[vars[p * len_Wa * len_D + w * len_D + d].getAttr("x") for d in range(len_D)] for w in range(len_Wa)] for p in range(P)]
        return vars_x

    else:
        return False





# Wa = Wa - 11 Wc = Wc - 11 W = W -11
P = 5
N = 220
T = 30
Wa = [0, 1, 2, 3]
Wc = [5, 9, 13, 17, 19, 21, 23, 25, 26, 27, 28, 29]
D = [0, 1, 2, 3, 4]
DP = [[0], [1], [2], [3], [4]]
DistributionP =[0.15, 0.25, 0.25, 0.2, 0.15]


W = list(range(0, T))



len_Wa = len(Wa)
len_Wc = len(Wc)
len_D = len(D)
len_W = len(W)


TW = 130
bound_pu = 0.2
senario_nu = 35
ar_list = [10]


col = 0
o = 0
pu = 0

writer = pd.ExcelWriter('montecarlo.xlsx')

ST = datetime.datetime.now()

o_all = []
pu_all = []
ar_all = []
runtime_all =[]


for l in range(len(ar_list)):
    tw = 0
    np.random.seed(100)

    vars_n_all = np.floor(np.array(
        [np.random.poisson(lam=ar_list[l] * 5 * DistributionP[p] / len_Wa, size=(TW, len_Wa)) for p in range(P)]).transpose(
        (1, 0, 2)))

    np.random.seed(1000)
    vars_nu_all = np.floor(np.array(
        [np.random.poisson(lam=ar_list[l] * 5 * DistributionP[p] / len_Wa, size=(TW, senario_nu, T - 1, len_Wa)) for p in
         range(P)]).transpose((1, 2, 3, 0, 4)))

    vars_num_tw = [[0 for d in range(len_D)] for t in range(TW + T)]
    vars_x_tw = []
    ar = ar_list[l]

    pu_tw = [0 for s in range(TW)]
    o_tw = [0 for s in range(TW)]
    pu_tw_single = [0 for s in range(TW)]
    o_tw_single = [0 for s in range(TW)]
    ST1 = datetime.datetime.now()

    # vars_num0
    while (tw <= TW - 1):
        vars_num = vars_num_tw[tw:tw + T].copy()

        vars_n = vars_n_all[tw]
        vars_nu_s = vars_nu_all[tw]
        vars_x = optimize_tw(vars_num, vars_n,vars_nu_s,l)
        vars_x_tw.append(vars_x)

        vars_num_tw = dynamic_num(vars_x, vars_num_tw, tw)

        pu_tw_single[tw] = func_pu(vars_n_all, vars_x_tw, tw, tw + 1)
        o_tw_single[tw] = func_o(vars_num_tw, tw, tw + 1)


        tw = tw + 1

    ET1 = datetime.datetime.now()
    runtime = ET1- ST1

    pu_all.append(sum(pu_tw_single[100:])/(TW-100))
    o_all.append(sum(o_tw_single[100:])/(TW-100))
    runtime_all.append(runtime)


ET = datetime.datetime.now()
runtime = ET - ST
print("Run time = ", ET - ST)

df = pd.DataFrame([ar_list,pu_all,o_all])
df.to_excel(writer,header=False, index=False)


print(runtime_all)

writer.save()