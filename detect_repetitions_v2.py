# -*- coding: utf-8 -*-

# TO DO
# issue of overlapping components not addressed.
# need a better way to select correct number of components. Choosing 0.1 as
# percentage change limit is probably not good enough. Could use baysian method
# perhaps AIC?

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import scipy.io as io
from scipy import signal
import entropydecomp as ed

# io.loadmat("")

N = 10000
step = 1
Lvec = 5
Lsig = 15
randscaling = 0.1

rn = np.random.randn(N) * randscaling
rn_noise = np.copy(rn)
rno = np.zeros(N)

z = np.sin(np.linspace(-np.pi + 0.4, np.pi + 0.4, Lsig)) # shift signal by 45 deg

for i in range(0, N, 50):
    rno[i:i + Lsig] = z

# f1 = 1/np.linspace(0, 100, len(rn)+1)[1:] \
#      * np.exp(-1j * 2 * np.pi * np.random.rand(len(rn)))
# f1[int(len(rn)/2)+1:] = 0
# rno = np.fft.ifft(f1).real
# rn = ed.butter_bandpass_filter(rn, 10, 100, 1000, 5).real

# plt.plot(f1)
# plt.show()

rn = rn + rno

l = list(permutations(range(0, Lvec)))
# print(l)

pkeys = list()
for t in l:
    pkeys.append("%s%s%s%s%s" % t)


plt.interactive(False)
plt.close("all")

rnn = np.copy(rn)


rstep = 2**np.array(range(0, 7, 1))
# rstep = np.array(range(1, 8, 1))

sampen = [1e6]
sampendiff = 1
ri = 0

newsig = np.zeros(rn.shape)

for ii in range(10):
    ri += 1
    # run through all rstep and find best spacing for next component
    rn, newsigc, sp, perms, permn = ed.rankdata_multi(rn, pkeys, rstep, Lvec)
    newsig += newsigc

    plt.figure(ri)
    f, ax = plt.subplots(3, 1, figsize=(20, 20))
    ax[0].plot(newsig)
    ax[0].plot(rno, 'r--')
    ax[0].plot(rnn, 'g--')
    ax[0].set_xlim(0, 300)
    ax[0].set_ylim(-1.5, 1.5)

    ax[1].plot(rnn, 'g--')
    ax[1].plot(rn, 'r--')
    ax[1].plot(rn_noise, 'b--')
    ax[1].set_xlim(0, 300)

    ax[2].plot(permn)

    # sampen.append(sampentropy(rn[1:100], 2, 0.2 * np.std(rn)))
    sampen.append(np.var(rn))
    sampendiff = (sampen[-2] - sampen[-1]) / sampen[-2]
    print("Variance perc change: Run %d = %f" % (ri, sampendiff))

    # end for


plt.show()

# ax = plt.subplots()
# ax = plt.hist(permn)

# ax = plt.subplots()
# plt.plot(sigmean)
# plt.plot(z)











