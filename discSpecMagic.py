# optimize
# change interface
# git it up
#
#####
#####

#
# 8/21/2018: 
# (*) AutoMagic Mode: need only par verbose and plotting flags; auto Nopt
# (*) switching to nnls as default fitting engine
# (*) changed older MaxwellModes and LLS -> nnls
# (*) some printing modifications
# (*) hardcoding prune = True everywhere; doesn't seem to be use case otherwise
# (*) making jupyter interact compliant
 

# Function: getdiscSpecMagic(par)
#
# Uses the continuous relaxation spectrum extracted using contSpec()
# to determine an approximate discrete approximation.
#
# Scan wb = wtBaseDist from 0 to 1; At each wb use AIC to determine best Nbst(wb) so long as the condition
#           number is below log10(cond) < 10. Then, pick the smallest Nbst(wb) as Nopt; Seems to do well
#           as a criterion;
#
# Input: Communicated by the datastructure "par"
#
#        fGstFile = name of file that contains G(t) in 2 columns [t Gt]
#                   default: 'Gt.dat' is assumed.
#        verbose  = True, then prints onscreen messages, and prints datafiles
#        plotting = True, then plots to stdio.#
#
# Output: Nopt    = optimum number of discrete modes
#         [g tau] = spectrum
#         error   = error norm of the discrete fit
#        
#         dmodes.dat : Prints the [g tau] for the particular Nopt
#         AIC.dat   : If Nopt not supplied, then optimum [wb Nbest AICbest]
#         Gfitd.dat  : The discrete G(t) for Nopt [t Gt] 
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

import os
import time
import common

plt.style.use('ggplot')		

#~ from matplotlib import rcParams
#~ rcParams['axes.labelsize'] = 28 
#~ rcParams['xtick.labelsize'] = 20
#~ rcParams['ytick.labelsize'] = 20 
#~ rcParams['legend.fontsize'] = 20
#~ rcParams['lines.linewidth'] = 2

def MaxwellModes(z, t, Gt):
	"""
	%
	% Function: MaxwellModes(input)
	%
	% Solves the linear least squares problem to obtain the DRS
	%
	% Input: z = points distributed according to the density,
	%        t  = n*1 vector contains times,
	%        Gt = n*1 vector contains G(t),
	%
	%        Prune = Avoid modes with -ve weights (=1), or don't care (0) 
	%
	% Output: g, tau = spectrum  (array)
	%         error = relative error between the input data and the G(t) inferred from the DRS
	%         condKp = condition number
	%
	"""

	N      = len(z)
	tau    = np.exp(z)
	n      = len(t)
	Gexp   = Gt

	#
	# Prune small -ve weights g(i)
	#
	g, error, condKp = nnLLS(t, tau, Gexp)

	izero = np.where(g < 1e-8)
	tau   = np.delete(tau, izero)
	g     = np.delete(g, izero)


	return g, tau, error, condKp

def nnLLS(t, tau, Gexp):
	
	"""
	#
	# Helper subfunction which does the actual LLS problem
	# helps MaxwellModes
	#
	"""
	from scipy.optimize import nnls
	
	n       = len(Gexp)
	S, T    = np.meshgrid(tau, t)
	K		= np.exp(-T/S)		# n * nmodes
		
	#
	# gets (Gt/GtE - 1)^2, instead of  (Gt -  GtE)^2
	#

	Kp      = np.dot(np.diag((1./Gexp)), K)
	condKp  = np.linalg.cond(Kp)
	g       = nnls(Kp, np.ones(len(Gexp)))[0]
#	g       = np.linalg.lstsq(Kp, np.ones(len(Gexp)), rcond=-1)[0]
		
	GtM   	= np.dot(K, g)
	error 	= np.sum((GtM/Gexp - 1.)**2)

	return g, error, condKp

def GetWeights(H, t, s):

	"""
	%
	% Function: GetWeights(input)
	%
	% Finds the weight of "each" mode by taking a weighted average of its contribution
	% to G(t)
	%
	% Input: H = CRS (ns * 1)
	%        t = n*1 vector contains times
	%        s = relaxation modes (ns * 1)
	%
	% Output: wt = weight of each mode
	%
	"""
  
	ns         = len(s)
	n          = len(t)

	hs         = np.zeros(ns)
	wt         = hs
	
	hs[0]      = 0.5 * np.log(s[1]/s[0])
	hs[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
	hs[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))

	S, T    = np.meshgrid(s, t)
	kern    = np.exp(-T/S)		# n * ns
	wij     =  np.dot(kern, np.diag(hs * np.exp(H)))  # n * ns
	K       = np.dot(kern, hs * np.exp(H))         # n * 1, comparable with Gexp

	for i in np.arange(n):
		wij[i,:] = wij[i,:]/K[i]

	for j in np.arange(ns):
		wt[j] = np.sum(wij[:,j])

	return wt

def GridDensity(x, px, N):

	"""#
	#  PROGRAM: GridDensity(input)
	#
	#	Takes in a PDF or density function, and spits out a bunch of points in
	#       accordance with the PDF
	#
	#  Input:
	#       x  = vector of points. It need *not* be equispaced,
	#       px = vector of same size as x: probability distribution or
	#            density function. It need not be normalized but has to be positive.
	#  	    N  = Number of points >= 3. The end points of "x" are included
	#  	     necessarily,
	# 
	#  Output:
	#       z  = Points distributed according to the density
	#       hz = width of the "intervals" - useful to apportion domain to points
	#            if you are doing quadrature with the results, for example.
	#
	#  (c) Sachin Shanbhag, November 11, 2015
	#"""

	npts = 100;                              # can potentially change
	xi   = np.linspace(min(x),max(x),npts)   # reinterpolate on equi-spaced axis
	fint = interp1d(x,px,'cubic')	         # smoothen using cubic splines
	pint = fint(xi)        					 # interpolation
	ci   = cumtrapz(pint, xi, initial=0)                
	pint = pint/ci[npts-1]
	ci   = ci/ci[npts-1]                     # normalize ci

	alfa = 1./(N-1)                          # alfa/2 + (N-1)*alfa + alfa/2
	zij  = np.zeros(N+1)                     # quadrature interval end marker
	z    = np.zeros(N)                       # quadrature point

	z[0]    = min(x);  
	z[N-1]  = max(x); 

	#
	# ci(Z_j,j+1) = (j - 0.5) * alfa
	#
	beta       = np.arange(0.5, N-0.5) * alfa
	zij[0]     = z[0]
	zij[N]     = z[N-1]
	fint       = interp1d(ci, xi, 'cubic')
	zij[1:N]   = fint(beta)
	h          = np.diff(zij)

	#
	# Quadrature points are not the centroids, but rather the center of masses
	# of the quadrature intervals
	#

	beta     = np.arange(1, N-1) * alfa
	z[1:N-1] = fint(beta)

	return z, h

def getDiscSpecMagic(par):

	# read input; initialize parameters
	if par['verbose']:
		print('\n(*) Start\n(*) Loading Data Files: ... {}...'.format(par['GexpFile']))

	t, Gexp, s, H = ReadData(par)

	n   = len(t);
	ns  = len(s);

	# range of N scanned
	Nmax  = min(np.floor(3.0 * np.log10(max(t)/min(t))),n/4); # maximum Nopt
	Nmin  = max(np.floor(0.5 * np.log10(max(t)/min(t))),3);   # minimum Nopt
	Nv    = np.arange(Nmin, Nmax + 1).astype(int)
	npts  = len(Nv)

	# range of wtBaseDist scanned
	wtBase = 0.05 * np.arange(1, 20)
	AICbst = np.zeros(len(wtBase))
	Nbst   = np.zeros(len(wtBase))
	nzNbst = np.zeros(len(wtBase))  # number of nonzeros

	# Estimate Error Weight from Continuous Curve Fit
	kernMat = common.getKernMat(s,t)
	Gc      = common.kernel_prestore(H, kernMat);
	kernMat = None # clear from memory
	Cerror  = 1./(np.std(Gc/Gexp - 1.))  #	Cerror = 1.?
	
	# main loop over wtBaseDist
	for ib, wb in enumerate(wtBase):
		
		#
		# Find the distribution of nodes you need
		#

		wt  = GetWeights(H, t, s)
		wt  = wt/np.trapz(wt, np.log(s))
		wt  = (1. - wb) * wt + (wb * np.mean(wt)) * np.ones(len(wt))

		# 
		# Scan the range of number of Maxwell modes N = (Nmin, Nmax) 
		#
		ev    = np.zeros(npts)
		condN = np.zeros(npts)
		nzNv  = np.zeros(npts)  # number of nonzero modes 

		for i in np.arange(npts):
			N      = Nv[i]
			z, hz  = GridDensity(np.log(s), wt, N)     # Select "tau" Points
			
			# Get g_i
			g, tau, ev[i], condN[i] = MaxwellModes(z, t, Gexp)
			nzNv[i]                 = len(g)

		# store the best solution for this particular wb
		condn      = np.log10(condN) < 10.
		AIC        = 2. * Nv + 2. * Cerror * ev
		
#		Nbst[ib]   = Nv[np.argmin(AIC[condn])]
		AICbst[ib] = min(AIC)
		Nbst[ib]   = Nv[np.argmin(AIC)]
		nzNbst[ib] = nzNv[np.argmin(AIC)]
		
	# global best settings of wb and Nopt; note this is nominal Nopt (!= len(g) due to NNLS)
	
	Nopt  = int(Nbst[np.argmin(AICbst)])
	wbopt = wtBase[np.argmin(AICbst)]

	#
	# Recompute the best data-set stats
	#
	wt  = GetWeights(H, t, s)
	wt  = wt/np.trapz(wt, np.log(s))
	wt  = (1. - wbopt) * wt + (wbopt * np.mean(wt)) * np.ones(len(wt))	
	
	z, hz              = GridDensity(np.log(s), wt, Nopt)           # Select "tau" Points
	g, tau, error, cKp = MaxwellModes(z, t, Gexp)   # Get g_i

	if par['verbose']:
		print('\n(*) Number of optimum nodes = {0:d}\n'.format(len(g)))

	#
	# Some Plotting
	#

	if par['plotting']:

		plt.clf()
		plt.plot(wtBase, AICbst, label='AIC')
		plt.plot(wtBase, nzNbst, label='Nbst')
		plt.scatter(wbopt, len(g), color='k')
		plt.scatter(wbopt, np.min(AICbst), color='k')
		plt.yscale('log')
		plt.xlabel('baseDistWt')
		plt.legend()
		plt.tight_layout()
		plt.savefig('output/AIC.pdf')		


		plt.clf()
		plt.loglog(tau,g,'o-', label='disc')
		plt.loglog(s, np.exp(H), label='cont')
		plt.xlabel('tau')
		plt.ylabel('g')
		plt.legend(loc='lower right')
		plt.tight_layout()
		plt.savefig('output/dmodes.pdf')		


		plt.clf()
		S, T    = np.meshgrid(tau, t)
		K		= np.exp(-T/S)		# n * nmodes			
		GtM   	= np.dot(K, g)
		plt.loglog(t,Gexp,'o')
		plt.loglog(t,GtM, label='disc')	
		

		plt.loglog(t, Gc, '--', label='cont')
		plt.xlabel('t')
		plt.ylabel('G(t)')
		plt.legend()
		plt.tight_layout()
		plt.savefig('output/Gfitd.pdf')
  
	#
	# Some Printing
	#

	if par['verbose']:

		print('(*) log10(Condition number) of matrix equation: {0:.2f}\n'.format(np.log10(cKp)))

		print('\n\t\tModes\n\t\t-----\n\n')
		print('i \t    g(i) \t    tau(i)\n')
		print('---------------------------------------\n')
		
		for i in range(len(g)):
			print('{0:3d} \t {1:.5e} \t {2:.5e}'.format(i+1,g[i],tau[i]))
		print("\n")

		np.savetxt('output/dmodes.dat', np.c_[g, tau], fmt='%e')
		np.savetxt('output/aic.dat', np.c_[wtBase, nzNbst, AICbst], fmt='%f\t%i\t%e')

		S, T    = np.meshgrid(tau, t)
		K		= np.exp(-T/S)		# n * nmodes			
		GtM   	= np.dot(K, g)
		np.savetxt('output/Gfitd.dat', np.c_[t, GtM], fmt='%e')


	return Nopt, g, tau, error

def ReadData(par):
	"""
		Read experimental data, and the continuous spectrum
	"""
	fNameH = 'output/H.dat'

	# Read experimental data
	t, Gexp = common.GetExpData(par['GexpFile'])

	# Read continuous spectrum
	s, H    = np.loadtxt(fNameH, unpack=True)

	return t, Gexp, s, H

def guiFurnishGlobals(par):
	"""Globals for Smoother operation of Jupyter Notebook"""
	# toggle flags to prevent printing    
	par['verbose'] = False
	par['plotting'] = False

	# Read experimental data
	t, Gexp = common.GetExpData(par['GexpFile'])

	# Read continuous spectrum
	s, H    = np.loadtxt('output/H.dat', unpack=True)    

	# Read continuous fit
	_, Gfitc = np.loadtxt('output/Gfit.dat', unpack=True)    

	wt  = GetWeights(H, t, s)
	wt  = wt/np.trapz(wt, np.log(s))

	return s, H, t, Gexp, Gfitc, wt

#############################
#
# M A I N  P R O G R A M
#
#############################

if __name__ == '__main__':
	#
	# Read input parameters from file "inp.dat"
	#
	par = common.readInput('inp.dat')
	_ = getDiscSpecMagic(par)
