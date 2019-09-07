
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



def ini_pu(sum, solution):
    sum_p = np.array([solution[p][w][d] for p in range(P) for w in range(0, len_Wa) for d in DP[p]]).sum()
    return 1 - sum_p / sum

# initialiation by LP
def ini_x(vars_n, vars_e, vars_num,bound_r_num):

    model = Model()

    # Add determining variables to mode
    vars_x = [[[model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="x[%d,%d,%d]" % (p, w, d)) for d
                  in range(len_D)] for w in range(len_Wa)] for p in range(P)]
    vars_a = np.array([[model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="a[%d,%d]" % (p, w))  for w in range(len_Wa)] for p in range(P)])
    vars_o = np.array([[model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="o[%d,%d]" % (d, t)) for t in range(T)]
              for d in range(len_D)])

    # Populate A matrix
    model.addConstrs((quicksum(vars_x[p][w]) == vars_a[p][w]) for p in range(P) for w in range(len_Wa) )
    model.addConstrs((vars_a[p][w] <= vars_n[p][w]) for p in range(P) for w in range(len_Wa))
    model.addConstr(np.sum(vars_a) == np.sum(vars_n)-bound_r_num)

    # dynamic_num
    vars_num0 = np.copy(vars_num)
    vars_num0 = dynamic_num(vars_x, vars_num0, 0)

    # Populate objective
    """preference unmet"""
    pu = ini_pu(np.sum(vars_n)-bound_r_num,vars_x)
    model.addConstr(pu <= bound_pu)
    model.addConstrs((vars_o[d][t] >= (vars_num0[t][d]+vars_e[t][d] - N)) for d in range(len_D) for t in range(T))
    o = np.sum(vars_o)

    model.setObjective(o*10 + pu*10, 1)

    model.setParam('MIPGap', 0.0003)

    model.update()


    # Solve
    model.optimize()
    if model.status == GRB.Status.OPTIMAL:
        # return pu, o
        vars = model.getVars()
        vars_x = [[[vars[p * len_Wa * len_D + w * len_D + d].getAttr("Xn") for d in range(len_D)] for w in range(len_Wa)] for p in range(P) ]

        return vars_x
    else:
        return False


# Fitness calculation
def func_pu(solution, st, et):
    sum_p = np.array([solution[t][p][w][p] for t in range(st, et) for p in range(P) for w in range(0, len_Wa) ]).sum()
    sum_a = np.array(solution[st:et]).sum()

    return 1 - sum_p / sum_a


def func_o(vars_num_tw, vars_e,st, et):
    vars_num_tw0 = np.copy(vars_num_tw[st:et]+vars_e[st:et]-N)
    return np.where(vars_num_tw0>0,vars_num_tw0,0).sum()

def func_r(vars_n, solution, st, et):
    sum_a = np.sum(np.array(solution[st:et]))
    sum_t = np.sum(vars_n[st:et])
    return (sum_t-sum_a)/sum_t

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

def fitness(solution,vars_num,vars_n,vars_e_s):
    vars_num0 = np.copy(vars_num)
    vars_num0 = dynamic_num(solution, vars_num0, 0)
    pu = func_pu([solution],0,1)
    o = np.array([func_o(vars_num0, vars_e_s[i] ,0, T) for i in range(senario_e)]).sum()/senario_e
    r = func_r([vars_n], [solution],0,1)
    return pu,o,r


"""Non dominated sort"""

def func_values(func123):
    function123_values = func123.transpose((1,0))

    # 将所有个体分为两个部分，满足bound_pu约束(pu_low)和不满足的(pu_high)
    index1_low = np.where(function123_values[0] <= bound_pu)[0]
    index3_low = np.where(function123_values[2] <= bound_r)[0]
    index_low = np.array([index3_low[np.where(index3_low == i)] for i in index1_low])
    index_low = [j for i in index_low for j in i]

    func123_low = func123[index_low]
    f02 = function123_values[0] + function123_values[2]
    index_high = np.delete(np.arange(len(func123)),index_low,axis=0)
    sorted_high = index_high[np.argsort(f02[index_high])]
    return sorted_high, func123_low,index_low

def compare(i,values):
    c = values[i]
    compare0 = np.sum(np.where(c <= values, 1, 0), axis=1)
    compare1 = np.sum(np.where(c >= values, 1, 0), axis=1)
    S1 = np.where(compare0 == 3)[0]
    S2 = np.where(compare1 == 3)[0]
    S1_adj = S1[np.where(np.sum(c) != np.sum(values[S1], axis=1))[0]]
    S2_adj = S2[np.where(np.sum(c) != np.sum(values[S2], axis=1))[0]]
    return list(S1_adj),len(S2_adj)

# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values123):
    # S represent index

    len_v1 = len(values123)
    S = [[] for i in range(0, len_v1)]
    n = [0 for i in range(0, len_v1)]
    rank = [0 for i in range(0, len_v1)]
    front = [[]]

    for p in range(0, len_v1):
        S[p],n[p] = compare(p,values123)
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)


    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[-1]

    return front

def func_cd(func_values,index_list):
    distance = np.zeros(len(func_values))
    values = func_values[index_list]
    sorted = index_list[np.argsort(values)]

    distance[sorted[0]] = 444
    distance[sorted[len(index_list)-1]] = 444

    if (func_values[sorted[len(index_list)-1]] != func_values[sorted[0]]):
        range0 = 1 / (func_values[sorted[len(index_list)-1]] - func_values[sorted[0]])

    else:
        range0 = 0

    if (len(index_list) - 1)>2:
        for k in range(1, (len(index_list) - 1)):
            distance[sorted[k]] = range0 * (func_values[sorted[k + 1]] - func_values[sorted[k - 1]])

    return distance

def crowding_distance(func123,front_index_list,index_low):
    sort_list = [[] for i in range(len(front_index_list))]
    distance_list = [[] for i in range(len(front_index_list))]
    func_values123 = func123.transpose((1,0))

    for l in range(len(front_index_list)):
        index_list = np.array(front_index_list[l])
        distance = np.array([func_cd(func_values123[i],index_list) for i in range(3)])
        distance0 = 0 - np.sum(distance,axis=0)[index_list]
        distance_list[l] = list(np.sort(distance0))
        sort_list[l] = [index_low[i] for i in index_list[np.argsort(distance0)]]

    return distance_list,sort_list

def rank(cd_index, sorted_high):
    rank_list0 = np.array([j for i in cd_index for j in i])
    rank_list = np.append(rank_list0,sorted_high)
    return rank_list


def non_dominated_rank(func123):
    sorted_high, func123_low,index_low= func_values(func123)
    non_dominated_sorted_index = fast_non_dominated_sort(func123_low)
    crowding_distance_values, cd_non_dominated_sorted_index = crowding_distance(func123_low,non_dominated_sorted_index,index_low)
    rank_list = rank(cd_non_dominated_sorted_index, sorted_high)
    return rank_list,non_dominated_sorted_index


"""GA operator"""
def selection(rank_list):
    prob = 0
    selection_parent = [-1, -1]
    len_rank = np.size(rank_list)

    selection_prob = len_rank * (len_rank + 1)/2 * np.random.random(size=2)

    for i in range(0, len_rank):
        lb = prob
        prob = (len_rank-i) + prob
        ub = prob
        for j in range(2):
            if (lb <= (selection_prob[j]) <= ub):
                selection_parent[j] = i

        if (-1 not in selection_parent):
            break
    return selection_parent[0],selection_parent[1]


# Multiple cross points - > generate single child
def crossover(pop_x, rank_list,rank1,rank2,num):
    new_list = []
    parent12_list = np.array([pop_x[int(rank_list[int(rank1)])]]+[pop_x[int(rank_list[int(rank2)])]])
    list0 = np.reshape(parent12_list,(2,P*len_Wa,len_D))
    crossover_point = sample(range(1, P * len_Wa), num) + [0, P*len_Wa]
    crossover_point.sort()
    prob = (rank1 + 1) / (rank1 + 1 + rank2 + 1)
    r = [int(random() < prob) for i in range(num + 1)]
    for i in range(num + 1):
        new_list = np.append(new_list,list0[r[i]][crossover_point[i]:crossover_point[i + 1]])

    return np.reshape(new_list,(P,len_Wa,len_D))


def mutation(list,vars_n,bound_r_num):
    vars_x = np.copy(list)
    mutation_index = np.random.randint(P * len_Wa)
    r = np.random.random(size=len_D)
    p = mutation_index // len_Wa
    w = mutation_index % len_Wa
    low = np.sum(vars_n[p][w])-(bound_r_num - (np.sum(vars_n)-np.sum(vars_x)))
    high = vars_n[p][w] + 1
    sum = np.random.randint(min(max(low,0),vars_n[p][w]), high)
    vars_x[p][w] = np.floor( sum * r / np.sum(r))
    index = randint(0, 4)
    vars_x[p][w][index] += sum - np.sum(vars_x[p][w])
    return vars_x


def ga_operator(pop_x, rank_list, num_cp,vars_n,bound_r_num):
    rank1, rank2 = selection(rank_list)
    if np.random.random()<0.3:
        individual = crossover(pop_x, rank_list, rank1, rank2, num_cp)
    else:
        individual = pop_x[int(rank_list[int(min(rank1,rank2))])]
    if np.random.random()<0.8:
        individual = mutation(individual,vars_n,bound_r_num)
    return individual


"""GA structure"""
def ga_structure(vars_n, vars_e_s, vars_num, bound_r_num,q):
    gen_no = 1
    pop_x, func123, rank_list, non_dominated_sorted_index = ga_ini(vars_n, vars_e_s, vars_num, bound_r_num)


    pu_all0 = []
    o_all0 = []
    r_all0 = []
    a_all0 = []
    while (gen_no < max_gen):
        vars_num0 = np.copy(vars_num)

        """elite strategy"""
        pop_x2, func123_2, rank_list2, non_dominated_sorted_index2 = ga_generate(pop_x, func123, rank_list, vars_num0, vars_n, vars_e_s, bound_r_num)
        vars_x, index = ga_final(non_dominated_sorted_index2, func123_2, pop_x2, q)
        pop_x, func123, rank_list, non_dominated_sorted_index = ga_sift(pop_x2, func123_2, rank_list2)
        pop_x.append(vars_x)
        func123 = np.insert(func123,pop_size-1,func123_2[index],axis = 0)
        vars_x, index = ga_final(non_dominated_sorted_index, func123, pop_x, q)
        gen_no += 1

    pu_all0.append(func123[index][0])
    o_all0.append(func123[index][1])
    r_all0.append(func123[index][2])
    a0 = np.sum(np.array(vars_x), axis=(0, 1))
    a = np.std(a0)
    a_all0.append(a)
    df = pd.DataFrame([pu_all0, r_all0, o_all0,a_all0])
    df.to_excel(writer, startcol=q,header=False, index=False,sheet_name="func")
    df = pd.DataFrame(np.array(vars_x).reshape(P*len_Wa,len_D))
    df.to_excel(writer, startcol=q*6,header=False, index=False,sheet_name="vars_x")

    return vars_x,func123[index]

def ga_ini(vars_n, vars_e_s,vars_num,bound_r_num):
    pop_x = [ini_x(vars_n, vars_e_s[np.random.randint(senario_e)], vars_num, np.random.randint(bound_r_num + 1)) for i in range(pop_size)]
    func123 = np.array([fitness(pop_x[i], vars_num, vars_n, vars_e_s) for i in range(pop_size)])
    rank_list, non_dominated_sorted_index = non_dominated_rank(func123)
    return pop_x,func123,rank_list, non_dominated_sorted_index

def ga_generate(pop_x,func123,rank_list, vars_num,vars_n, vars_e_s,bound_r_num):
    pop_x2 = pop_x + [ga_operator(pop_x, rank_list, num_cp, vars_n, bound_r_num) for i in range(pop_size)]
    func123_2 = np.array([fitness(pop_x2[i + pop_size], vars_num, vars_n, vars_e_s) for i in range(pop_size)])
    func123_2 = np.concatenate((func123, func123_2), axis=0)
    rank_list2, non_dominated_sorted_index2 = non_dominated_rank(func123_2)
    return pop_x2,func123_2, rank_list2, non_dominated_sorted_index2

def ga_sift(pop_x2, func123_2, rank_list2):
    pop_x = [pop_x2[int(i)] for i in rank_list2[:pop_size-1]]
    func123 = func123_2[ rank_list2[:pop_size-1]]
    rank_list, non_dominated_sorted_index = non_dominated_rank(func123)
    return pop_x, func123, rank_list, non_dominated_sorted_index

def ga_final(non_dominated_sorted_index,func123,pop_x,q):
    func = func123[non_dominated_sorted_index[0]].transpose((1, 0))

    o_list = func[1]
    r_list = func[2]


    a0 = np.sum(np.array(pop_x)[non_dominated_sorted_index[0]], axis=(1, 2))
    a = np.std(a0,axis=1)
    func_list = o_list * senario_e+a/10000+r_list

    vars_x = pop_x[non_dominated_sorted_index[0][np.argmin(func_list)]]



    return vars_x,non_dominated_sorted_index[0][np.argmin(func_list)]

# Main program starts here

P = 5
DistributionP =[0.15, 0.25, 0.25, 0.2, 0.15]
# DistributionP =[0.1, 0.32, 0.32, 0.16, 0.1]
N = 220
T = 30
Wa = [0, 1, 2, 3]
Wc = [5, 9, 13, 17, 19, 21, 23, 25, 26, 27, 28, 29]
D = [0, 1, 2, 3, 4]
DP = [[0], [1], [2], [3], [4]]
W = list(range(0, T))

len_Wa = len(Wa)
len_Wc = len(Wc)
len_D = len(D)
len_W = len(W)


TW = 1
bound_r = 0.1
bound_pu = 0.1
# senario_e_list = [1,/10,50,100]
senario_e_list = [2,2]
pop_size = 20
max_gen = 1000
num_cp = 2
senario_e2 = 1

ST = datetime.datetime.now()

np.random.seed(100)
vars_n_all = np.floor(np.array([np.random.poisson(lam=19 * 5 * DistributionP[p] / len_Wa, size=(TW, len_Wa)) for p in range(P)]).transpose((1, 0, 2)))

np.random.seed(3000)
vars_e_all = np.floor(np.random.poisson(lam=5, size=(senario_e2, TW, len_D)))



writer = pd.ExcelWriter('m2_NSGA_myopic106.xlsx')
df = pd.read_excel(io='C:\\Users\\dell\\Desktop\\m2_NSGA_myopic.xlsx', sheet_name='vars_num_tw', header=None, index_col=None)

o_all =[]
pu_all =[]
r_all = []
r_num_all = []
senario_all=[]
runtime_all=[]
o_max = []
o_min = []
o_std = []
o_mean = []
a_all = []
func_all= []

# Main function

q=0
for senario_e in senario_e_list:
    np.random.seed(10)
    vars_e_s0 = np.floor(np.random.poisson(lam=5, size=(TW, senario_e, T, len_D)))


    solution = []
    vars_num_tw = np.array(df)[100:(100 + TW + T - 1)]


    pu_tw_single = [0 for s in range(TW)]
    o_tw_single = [0 for s in range(TW)]
    r_tw_single = [0 for s in range(TW)]

    r_num_tw_single = [0 for s in range(TW)]
    tw = 0
    ST1 = datetime.datetime.now()

    while (tw <= TW - 1):

        ST0 = datetime.datetime.now()

        vars_num = np.copy(vars_num_tw[tw:tw + T])
        vars_n = vars_n_all[tw]
        vars_e_s = vars_e_s0[tw]
        bound_r_num = np.floor(bound_r * np.sum(vars_n))
        vars_x,func =ga_structure(vars_n, vars_e_s, vars_num, bound_r_num,q)


        solution.append(vars_x)
        func_all.append(func)
        vars_num_tw = dynamic_num(vars_x, vars_num_tw, tw)


        pu_all.append(func_pu(solution, tw, tw + 1))
        o_tw_single_list = np.array([func_o(vars_num_tw[tw:], vars_e_all[s][tw:], 0, 1) for s in range(senario_e2)])

        a0 = np.sum(np.array(vars_x), axis=(0, 1))
        a = np.std(a0)
        a_all.append(a)

        o_mean.append(np.mean(o_tw_single_list))
        o_max.append(np.max(o_tw_single_list))
        o_min.append(np.min(o_tw_single_list))
        o_std.append(np.std(o_tw_single_list))
        r_all.append(func_r(vars_n_all, solution, tw, tw + 1))
        r_num_all.append(r_all[tw]*np.sum(vars_n))

        tw = tw+1

        ET0 = datetime.datetime.now()
        runtime = ET0 - ST0
        print("single time all", runtime)


    ET1 = datetime.datetime.now()
    runtime = ET1 - ST1
    senario_all.append(senario_e)
    q+=1

ET = datetime.datetime.now()

print("Total Run Time = ", ET - ST)
print("run time all", runtime_all)
df = pd.DataFrame([senario_all,pu_all, r_all,r_num_all,o_max,o_min,o_std,o_mean,a_all])
df.to_excel(writer, header=False, index=False)

writer.save()

