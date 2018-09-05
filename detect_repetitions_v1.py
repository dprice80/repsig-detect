# -*- coding: utf-8 -*-

# import scipy as sp
import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt
# from scipy.special import comb
from itertools import permutations

from numpy.core.multiarray import ndarray

N = 10000
step = 1
Lvec = 5
Lpac = 5
randscaling = 1

rn = np.random.randn(N, 1) * randscaling
rno = np.zeros([N, 1])

z = np.sin(np.linspace(-np.pi + 0.01, np.pi + 0.01, Lpac))
for rstep in range(0, N, 50):
    rno[rstep:rstep + Lpac, 0] = z

rn = rn + rno

#while i < N:


l = list(permutations(range(0, Lvec)))
print(l)

pkeys = list()
for t in l:
    pkeys.append("%s%s%s%s%s" % t)


# define ranking function
def rankdata(data, pkeys, rankstep=1):
    Ns = len(data)
    perms = {pk: [] for pk in pkeys}
    for i in range(0, Ns - 5 * rankstep):
        ranks = data[i:i + 5 * rankstep:rankstep].transpose().argsort()
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


def reconstructsignal(rn, perms, pkeys, pind, rstep):
    rs = np.zeros(rn.shape)  # reconstructed signal
    rs[:] = np.nan
    for pi in perms[pkeys[pind]]:
        sigmean = generatesigmean(pind)
        rs[pi:pi + Lvec * rstep:rstep, 0] = sigmean
    return rs.flatten(), sigmean.flatten()


def concatnplist(nplist):
    nout = np.zeros([len(nplist), N])
    for ii in range(0,len(nplist)):
        nout[ii, :] = nplist[ii]
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
    for pi in psi[0:1]:
        newsig, sig = reconstructsignal(rn, perms, pkeys, pi, rstep)
        # find correlation coef and decide whether to keep this measure.
        c = np.corrcoef(newsig[np.isnan(newsig) == False],
                    rn[np.isnan(newsig) == False].transpose())
        # if c[0,1] > clim:
        sigtemp.append(newsig)
    # while len(newsiglist) > 0:

    newsigc = np.nansum(concatnplist(sigtemp), 0)
    newsigc[np.isnan(newsigc)] = 0



plt.interactive(False)
plt.close("all")
f, ax = plt.subplots(2, figsize=(20, 20))
ax[0].plot(newsigc)
ax[0].plot(rno,'r--')
ax[0].set_xlim(0, 1000)
# ax = plt.subplots()
ax[1].plot(sig)
plt.show()

# ax = plt.subplots()
# ax = plt.hist(permn)

# ax = plt.subplots()
# plt.plot(sigmean)
##plt.plot(z)











