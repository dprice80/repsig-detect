# -*- coding: utf-8 -*-

# import scipy as sp
import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt
# from scipy.special import comb
from itertools import permutations
from scipy.stats import zscore

N = 100000
step = 1
Lvec = 5
Lsig = 10
randscaling = 1

rn = np.random.randn(N, 1) * randscaling
rno = np.zeros(rn.shape)
z = np.zeros([Lvec, 1])

z = np.sin(np.linspace(-np.pi + 0.01, np.pi + 0.01, Lsig))
for i in range(0, N, 50):
    rno[i:i + Lsig, 0] = z

rn = rn + rno

l = list(permutations(range(0, Lvec)))
# print(l)

pkeys = list()
for t in l:
    pkeys.append("%s%s%s%s%s" % t)


# define ranking function
def rankdata(data, pkeys, rankstep=1):
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


def reconstructsignal(rn, perms, pkeys, permind, step):
    rs = np.zeros(rn.shape)  # reconstructed signal
    sigmean = generatesigmean(permind)
    for pi in perms[pkeys[permind]]:
        rs[pi:pi + Lvec:step, 0] = sigmean
    return rs, sigmean


def concatnplist(nplist):
    nout = np.zeros([len(nplist), N])
    for ii in range(0,len(nplist)):
        nout[ii, :] = nplist[ii].flatten()
    return nout


newsig = np.zeros(rn.shape)
newsigc = np.zeros(rn.shape)
spaces = np.array(range(1,2))
for rstep in spaces:
    print("Calculating spacing %d" % rstep)
    perms, permn = rankdata(rn, pkeys, rankstep=rstep)
    psi = permn.argsort()[::-1]
    print(perms[pkeys[psi[0]]][0:3])
    permncumsum = permn[psi].cumsum()
    ind = np.argmax(permncumsum > sum(permn)*0.05)  # Find top 10%
    newsigall = np.zeros([N,1])
    newsigall[:] = np.nan
    outind = 0
    clim = 0.9
    sigtemp = list()
    for pi in psi[0:10]:
        newsig, sig = reconstructsignal(rn, perms, pkeys, pi, rstep)
        # find correlation coef and decide whether to keep this measure.
        c = np.cov(newsig[np.isnan(newsig) == False],
                    rn[np.isnan(newsig) == False].T)
        print(c[0,1])
        # if c[0,1] > clim:
        newsig[np.isnan(newsig)] = 0
        sigtemp.append(newsig*c[0,1])
    # while len(newsiglist) > 0:

    newsigc = np.nanmean(concatnplist(sigtemp), 0)
    newsigc[np.isnan(newsigc)] = 0


plt.interactive(False)
plt.close("all")
f, ax = plt.subplots(2, figsize=(20, 20))
ax[0].plot(zscore(newsigc))
ax[0].plot(zscore(rno),'r--')
ax[0].set_xlim(0, 1000)

# ax = plt.subplots()
ax[1].plot(sig.transpose())
plt.show()

# ax = plt.subplots()
# ax = plt.hist(permn)

# ax = plt.subplots()
# plt.plot(sigmean)
##plt.plot(z)











