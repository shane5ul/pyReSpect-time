#
# Running some tests;
# (i)   original setting used in ReSpect
# (ii)  "jit"-ing made it costlier
# (iii) explicitly writing out dot_products "kernel_loop" helped jit, but still costlier than orig
# (iv)  so far, the best is kernel_prestore: shaves a factor of 3
#

import numpy as np
import matplotlib.pyplot as plt

from numba import jit
from time import time


def kernel_orig(H, t, s):
	"""
	original kernel function; no savings (in fact more costly) with jit.		
	"""
	ns = len(s);
	hs = np.zeros(ns);

	#
	# integration uses trapezoidal rule
	# end steps have half the value as middle steps
	#
	hs[0]      = 0.5 * np.log(s[1]/s[0])
	hs[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
	hs[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))


	S, T       = np.meshgrid(s, t);
	kern       = np.exp(-T/S);
	K          = np.dot(kern, hs * np.exp(H))
	
	return K

#@jit
def kernel_loop(H, t, s):
	"""
	original kernel function; loop becomes expensive
	jit works better on this, but original is still better!	
	"""
	ns = len(s);
	hs     = np.log(s[1]/s[0]) * np.ones(ns)
	hs[0]  = 0.5*hs[0]
	hs[-1] = 0.5*hs[-1]

	K = np.zeros(len(t))

	for i in range(len(t)):
		K[i] = np.sum(np.exp(H - t[i]/s) * hs)


#	S, T       = np.meshgrid(s, t);
#	kern       = np.exp(-T/S);
#	K          = np.dot(kern, hs * np.exp(H))
	
	return K



def kernel_prestore(H, hs, kern):
	"""
		store hs and kern: 25-30% the run time
	"""
	K          = np.dot(kern, hs * np.exp(H))
	
	return K


def kernel_prestore1(H, Kp):
	"""
		store hs and kern: 25-30% the run time
	"""
	K          = np.dot(Kp, np.exp(H))
	
	return K


#####################
# main code
#####################

s    = np.logspace(-3, 3, 100)
t    = np.logspace(-3, 3, 50)	
H    = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)	
	

nrun = 1000

# original
if True:
	tp = time()
	for _ in range(nrun):
		K = kernel_orig(H, t, s)
	print("original time (ms) ", (time()-tp)/nrun*1000)
	plt.plot(t, K)

# original
if True:
	tp = time()
	for _ in range(nrun):
		K = kernel_loop(H, t, s)
	print("loop time (ms) ", (time()-tp)/nrun*1000)
	plt.plot(t, K, 'o-')	
	
if True:
	# version prestore: save hs and kern
	tp = time()
	ns = len(s);
	hs = np.zeros(ns);
	hs[0]      = 0.5 * np.log(s[1]/s[0])
	hs[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
	hs[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))
	S, T       = np.meshgrid(s, t);
	kern       = np.exp(-T/S);

	for _ in range(nrun):
		K = kernel_prestore(H, hs, kern)
	print("prestore time (ms) ", (time()-tp)/nrun*1000)
	plt.plot(t, K, 'o-')	

if True:
	# version prestore1: save hs and kern into single unit
	tp = time()
	ns = len(s);
	hs = np.zeros(ns);
	hs[0]      = 0.5 * np.log(s[1]/s[0])
	hs[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
	hs[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))
	S, T       = np.meshgrid(s, t);
	Kp         = np.exp(-T/S) * hs;
	
	for _ in range(nrun):
		K = kernel_prestore1(H, Kp)
	print("prestore1 time (ms) ", (time()-tp)/nrun*1000)
	plt.plot(t, K, 'o-')

plt.show()
