# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import scipy.io as io
from scipy import signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


# define ranking function
def rankdata(data, pkeys, lvec=5, rstep=1):
    Ns = len(data)
    perms = {pk: [] for pk in pkeys}
    for i in range(0, Ns - lvec * rstep):
        ranks = data[i:i + lvec * rstep:rstep].transpose().argsort()
        a = ''.join(map(str, ranks))
        perms[a].append(i)

    permn = []
    for key in pkeys:
        permn.append(len(perms[key]))
    permn = np.array(permn)

    return perms, permn


def find(a):
    return [i for (i, val) in enumerate(a) if val == True]


# find average of component
# for each permutation of rank
def generatesigmean(rn, permind, rstep, pkeys, perms, lvec=5):
    # check if instance has more than N repetitions
    sigmean = np.zeros(lvec)
    ls = len(perms[pkeys[permind]])
    L = len(rn)
    for ii in perms[pkeys[permind]]:
        ind = range(ii, ii+(lvec*rstep), rstep)
        if ind[-1] <= len(rn):
            sigmean = sigmean + rn[ind] / ls
    return sigmean


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


def rankdata_multi(rn, pkeys, rstep, lvec=5):
    newsigall = []
    newsigcall = []
    allperms = []
    allpermn = []
    enlog = []
    newsigc = np.zeros(rn.shape)

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
        sig = generatesigmean(rn, pi, rstep[rsi], pkeys, perms, lvec)
        # For each instance of vector (ppi is the index in rn)

        # calculate polyfit for each location
        for ppi in perms[pkeys[pi]]:
            rind = range(ppi, ppi + lvec * rstep[rsi], rstep[rsi])
            permsfit[pkeys[pi]].append(
                np.polyfit(sig.flatten(), rn[rind], 1))

        # calculate regression at each location of chosen motif
        for pfi in range(0, len(perms[pkeys[pi]])):
            ppi = perms[pkeys[pi]][pfi]
            rind = range(ppi, ppi + lvec * rstep[rsi], rstep[rsi])
            ysig = sig * permsfit[pkeys[pi]][pfi][0] + permsfit[pkeys[pi]][pfi][1]
            newsig[rind] += ysig.flatten()

        # Delete extracted components from original source timeseries
        rncand -= newsig # candidate rn
        newsigccand += newsig # clean signal
        newsigall.append(rncand) # save all condidate signals
        newsigcall.append(newsigccand) # save clean signal candidates

        permn = permn/sum(permn)
        allperms.append(perms)
        allpermn.append(permn)
        enlog.append(np.sum(-permn * np.log2(permn)))

    # find best by checking variance of new signals
    v = []
    for rsi in range(len(rstep)):
        v.append(np.var(newsigall[rsi]))

    ind = find(v == min(v))[0]
    ind = find(enlog == max(enlog))[0]

    return newsigall[ind], newsigcall[ind], rstep[ind], allperms[ind], allpermn[ind], enlog[ind]
