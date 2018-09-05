# -*- coding: utf-8 -*-

# import scipy as sp
import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt
# from scipy.special import comb
from itertools import permutations

N = 10000
step = 1
Lvec = 5
randscaling = 1

rn = np.random.randn(N, 1) * randscaling

z = np.zeros([Lvec, 1])

z = np.sin(np.linspace(-np.pi + 0.01, np.pi + 0.01, Lvec))
for i in range(0, N, 50):
    rn[i:i + Lvec, 0] = z

l = list(permutations(range(0, Lvec)))
print(l)

pkeys = list()
for t in l:
    pkeys.append("%s%s%s%s%s" % t)


# define ranking function
def rankdata(data, pkeys, step=1):
    Ns = len(data)
    perms = {pk: [] for pk in pkeys}
    for i in range(0, Ns - 5 * step):
        if i == 16:
            print("debug")
        ranks = data[i:i + 5 * step:step].transpose().argsort()
        a = ''.join(map(str, ranks[0]))
        perms[a].append(i)

    permn = []
    for key in pkeys:
        permn.append(len(perms[key]))
    permn = np.array(permn)

    return (perms, permn)


def find(a):
    return [i for (i, val) in enumerate(a) if val == True]


# find average of component
# for each permutation of rank
def generatesigmean(permind):
    # check if instance has more than N repetitions
    sigmean = np.zeros([1, Lvec])
    ls = len(perms[pkeys[permind]])
    for ii in perms[pkeys[permind]]:
        sigmean = sigmean + rn[ii:ii + Lvec].transpose() / ls
    return sigmean


def reconstructsignal(perms, pkeys, permind, L, step):
    rs = np.zeros(rn.shape)  # reconstructed signal
    sigmean = generatesigmean(permind)
    for pi in perms[pkeys[permind]]:
        rs[pi:pi + Lvec:step, 0] = sigmean
    return rs, sigmean


newsig = np.zeros(rn.shape)
newsigc = np.zeros(rn.shape)
rnt = rn
spaces = [1]
for i in spaces:
    print("Calculating spacing %d" % i)
    perms, permn = rankdata(rnt, pkeys, step=i)
    permind = find(permn == max(permn))[0]
    newsig, sigmean = reconstructsignal(perms, pkeys, permind, len, i)
    # rnt = rnt - newsig
    newsigc = newsigc + newsig / len(spaces)


plt.interactive(False)
plt.close("all")
f, ax = plt.subplots(2, figsize=(20, 20))
ax[0].plot(newsigc[0:1000])

# ax = plt.subplots()
ax[1].plot(sigmean.transpose())
plt.show()

# ax = plt.subplots()
# ax = plt.hist(permn)

# ax = plt.subplots()
# plt.plot(sigmean)
##plt.plot(z)











