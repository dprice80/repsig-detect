# -*- coding: utf-8 -*-

# import scipy as sp
import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt
# from scipy.special import comb
from itertools import permutations
from scipy.stats import zscore

N = 1000000
step = 1
Lvec = 5
Lsig = 10
randscaling = 0.5

rn = np.random.randn(N) * randscaling
rno = np.zeros(N)

z = np.sin(np.linspace(-np.pi + 0.4, np.pi + 0.4, Lsig)) # shift signal by 45 deg
for i in range(0, N, 50):
    rno[i:i + Lsig] = z

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
        a = ''.join(map(str, ranks))
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


def concatnplist(nplist):
    nout = np.zeros([len(nplist), N])
    for ii in range(0,len(nplist)):
        nout[ii, :] = nplist[ii].flatten()
    return nout


plt.interactive(False)
plt.close("all")

rnn = np.copy(rn)

newsigc = np.zeros(rn.shape)
spaces = np.array(range(1,2))
rstep = [1, 2]
for ri in range(0,1):
    print("Calculating spacing %d" % rstep[0])
    perms, permn = rankdata(rn, pkeys, rankstep=rstep[0])
    psi = permn.argsort()[::-1]
    print(perms[pkeys[psi[0]]][0:3])
    permncumsum = permn[psi].cumsum()
    ind = np.argmax(permncumsum > sum(permn)*0.05)  # Find top 10%
    newsigall = np.zeros([N,1])
    newsigall[:] = np.nan
    outind = 0
    clim = 0.9
    sigtemp = list()
    permsfit = {k:[] for k in pkeys}
    newsig = np.zeros(rn.shape)
    for pi in psi[0:1]: # choose the vector with most occurences
        sig = generatesigmean(pi)
        # for each instance of vector (ppi is the index in rn)
        for ppi in perms[pkeys[pi]]:
            permsfit[pkeys[pi]].append(
                np.polyfit(sig.flatten(), rn[ppi:ppi + Lvec:step], 1))

        for pfi in range(0, len(perms[pkeys[pi]])):
            ppi = perms[pkeys[pi]][pfi]
            ysig = sig * permsfit[pkeys[pi]][pfi][0] + permsfit[pkeys[pi]][pfi][1]
            newsig[ppi:ppi + Lvec:step] += ysig.flatten()  # changed this from += to =
    rn -= newsig

    plt.figure(ri)
    f, ax = plt.subplots(3, figsize=(20, 20))
    ax[0].plot(newsig)
    ax[0].plot(rno,'r--')
    ax[0].set_xlim(0, 1000)
    ax[0].set_ylim(-1.5, 1.5)

    # ax = plt.subplots()
    ax[1].plot(sig.transpose())

    ax[2].plot(rnn, 'b')
    ax[2].plot(rn,'r--')
    ax[2].plot(newsig,'g')
    ax[2].plot(rno,'m--')
    ax[2].set_xlim(0, 1000)
# end for

plt.show()

# ax = plt.subplots()
# ax = plt.hist(permn)

# ax = plt.subplots()
# plt.plot(sigmean)
##plt.plot(z)











