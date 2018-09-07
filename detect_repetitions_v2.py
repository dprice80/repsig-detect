# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations


N = 100000
step = 1
Lvec = 5
Lsig = 15
randscaling = 0.1

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
def rankdata(data, pkeys, rstep=1):
    Ns = len(data)
    perms = {pk: [] for pk in pkeys}
    for i in range(0, Ns - Lvec * rstep):
        ranks = data[i:i + Lvec * rstep:rstep].transpose().argsort()
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
def generatesigmean(permind, rstep):
    # check if instance has more than N repetitions
    sigmean = np.zeros(Lvec)
    ls = len(perms[pkeys[permind]])
    L = len(rn)
    for ii in perms[pkeys[permind]]:
        ind = range(ii, ii+(Lvec*rstep),rstep)
        if ind[-1] <= len(rn):
            sigmean = sigmean + rn[ind] / ls
    return sigmean


def concatnplist(nplist):
    nout = np.zeros([len(nplist), N])
    for ii in range(0,len(nplist)):
        nout[ii, :] = nplist[ii].flatten()
    return nout


def sampentropy(U, m, r):
    def _maxdist(x_i, x_j):
        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        return result

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)

    return -np.log(_phi(m + 1) / _phi(m))


plt.interactive(False)
plt.close("all")

rnn = np.copy(rn)

newsigc = np.zeros(rn.shape)
rstep = list(range(1,6,1))[::-1]
sampen = [1e6]
sampendiff = 1
ri = 0

while sampendiff > 0.005 or sampendiff < 0:
    rsi = np.mod(ri,5)
    newsigall = []
    newsigcall = []
    for rsi in range(len(rstep)):
        rncand = np.copy(rn)  # candidate rn
        newsigccand = np.copy(newsigc)
        print("Calculating spacing %d" % rstep[rsi])
        perms, permn = rankdata(rn, pkeys, rstep=rstep[rsi])
        psi = permn.argsort()[::-1]
        permncumsum = permn[psi].cumsum()
        sigtemp = list()
        permsfit = {k: [] for k in pkeys}
        newsig = np.zeros(rn.shape)
        pi = psi[0] # choose the vector with most occurences
        sig = generatesigmean(pi, rstep[rsi])
        # For each instance of vector (ppi is the index in rn)

        # calculate polyfit for each location
        for ppi in perms[pkeys[pi]]:
            rind = range(ppi, ppi + Lvec * rstep[rsi], rstep[rsi])
            permsfit[pkeys[pi]].append(
                np.polyfit(sig.flatten(), rn[rind], 1))

        # calculate regression at each location of chosen motif
        for pfi in range(0, len(perms[pkeys[pi]])):
            ppi = perms[pkeys[pi]][pfi]
            rind = range(ppi, ppi + Lvec * rstep[rsi], rstep[rsi])
            ysig = sig * permsfit[pkeys[pi]][pfi][0] + permsfit[pkeys[pi]][pfi][1]
            newsig[rind] += ysig.flatten()

        # Delete extracted components from original source timeseries
        rncand -= newsig # candidate rn
        newsigccand += newsig # clean signal
        newsigall.append(rncand) # save all condidate signals
        newsigcall.append(newsigccand) # save clean signal candidates

    v = []
    for rsi in range(len(rstep)):
        v.append(np.var(newsigall[rsi]))

    rn = newsigall[find(v == min(v))[0]]
    newsigc = newsigcall[find(v == min(v))[0]]

    plt.figure(ri)
    f, ax = plt.subplots(3, 1, figsize=(20, 20))
    ax[0].plot(newsigc)
    ax[0].plot(rno, 'r--')
    ax[0].set_xlim(0, 1000)
    ax[0].set_ylim(-1.5, 1.5)

    ax[1].plot(sig.T)

    ax[2].plot(rnn, 'b')
    ax[2].plot(rn, 'r--')
    ax[2].set_xlim(0, 1000)

    # sampen.append(sampentropy(rn[1:100], 2, 0.2 * np.std(rn)))
    sampen.append(np.var(rn))
    sampendiff = sampen[-2] - sampen[-1]
    print("Variance: Run %d = %f, permn=%d" % (ri, sampen[ri], permn[pi]))
    ri += 1

# end for


plt.show()

# ax = plt.subplots()
# ax = plt.hist(permn)

# ax = plt.subplots()
# plt.plot(sigmean)
# plt.plot(z)











