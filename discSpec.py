#
# 12/15/2018: 
# (*) Introducing NNLS optimization of previous optimal solution
#     - can make deltaBaseWeightDist : 0.2 or 0.25 [coarser from 0.05]
#     - wrote new routines: FineTuneSolution, res_tG (for vector of residuals)
# 

from common import *
np.set_printoptions(precision=2)

def initializeDiscSpec(par):
	"""Returns:
		(*)	the experimental data: t, Gexp
		(*) the continuous spectrum: s, H (from output/H.dat)
		(*) Number of modes range: Nv
		(*) Error Weight estimate from Continuous Curve (AIC criterion)
	"""
	
	# read input; initialize parameters
	if par['verbose']:
		print('\n(*) Start\n(*) Loading Data Files: ... {}...'.format(par['GexpFile']))

	# Read experimental data
	t, Gexp = GetExpData(par['GexpFile'])

	# Read the continuous spectrum
	fNameH  = 'output/H.dat'
	s, H    = np.loadtxt(fNameH, unpack=True)

	n    = len(t);
	ns   = len(s);
	
	# range of N scanned
	if(par['MaxNumModes'] == 0):
		Nmax  = min(np.floor(3.0 * np.log10(max(t)/min(t))),n/4); # maximum Nopt
		Nmin  = max(np.floor(0.5 * np.log10(max(t)/min(t))),2);   # minimum Nopt
		Nv    = np.arange(Nmin, Nmax + 1).astype(int)
	else:
		Nv    = np.arange(par['MaxNumModes'], par['MaxNumModes'] + 1).astype(int)

	# Estimate Error Weight from Continuous Curve Fit
	kernMat = getKernMat(s,t)
	Gc      = kernel_prestore(H, kernMat);
	Cerror  = 1./(np.std(Gc/Gexp - 1.))  #	Cerror = 1.?
	
	return t, Gexp, s, H, Nv, Gc, Cerror

def MaxwellModes(z, t, Gt):
	"""
	%
	% Function: MaxwellModes(input)
	%
	% Solves the linear least squares problem to obtain the DRS
	%
	% Input: z  = points distributed according to the density, [z = log(tau)]
	%        t  = n*1 vector contains times,
	%        Gt = n*1 vector contains G(t),
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
	# Prune small and -ve weights g(i)
	#
	g, error, condKp = nnLLS(t, tau, Gexp)

	izero = np.where(g < 1e-12)
	tau   = np.delete(tau, izero)
	g     = np.delete(g, izero)

	return g, tau, error, condKp

def nnLLS(t, tau, Gexp):
	"""
	#
	# Helper subfunction which does the actual LLS problem
	# helps MaxwellModes; relies on nnls
	#
	"""
	n       = len(Gexp)
	S, T    = np.meshgrid(tau, t)
	K		= np.exp(-T/S)		# n * nmodes
		
	#
	# gets (Gt/GtE - 1)^2, instead of  (Gt -  GtE)^2
	#
	Kp      = np.dot(np.diag((1./Gexp)), K)
	condKp  = np.linalg.cond(Kp)
	g       = nnls(Kp, np.ones(len(Gexp)))[0]
		
	GtM   	= np.dot(K, g)
	error 	= np.sum((GtM/Gexp - 1.)**2)

	return g, error, condKp

def GetWeights(H, t, s, wb):
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
	%       wb = weightBaseDist
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

	wt  = wt/np.trapz(wt, np.log(s))
	wt  = (1. - wb) * wt + (wb * np.mean(wt)) * np.ones(len(wt))

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

def mergeModes_magic(g, tau, imode, t, Gexp):
	"""merge modes imode and imode+1 into a single mode
	   return gp and taup corresponding to this new mode;
	   12/2018 - also tries finetuning before returning
	   
	   uses helper functions:
	   - normKern_magic()
	   - costFcn_magic()
	   
	   
	"""
	iniGuess = [g[imode] + g[imode+1], 0.5*(tau[imode] + tau[imode+1])]
	res      = minimize(costFcn_magic, iniGuess, args=(g, tau, imode))

	newtau   = np.delete(tau, imode+1)
	newtau[imode] = res.x[1]
		
	return newtau

def normKern_magic(t, gn, taun, g1, tau1, g2, tau2):
	"""helper function: for costFcn and mergeModes"""
	Gn = gn * np.exp(-t/taun)
	Go = g1 * np.exp(-t/tau1) + g2 * np.exp(-t/tau2)
	return (Gn/Go - 1.)**2

def costFcn_magic(par, g, tau, imode):
	""""helper function for mergeModes; establishes cost function to minimize"""
	gn   = par[0]
	taun = par[1]

	g1   = g[imode]
	g2   = g[imode+1]
	tau1 = tau[imode]
	tau2 = tau[imode+1]

	tmin = min(tau1, tau2)/10.
	tmax = max(tau1, tau2)*10.

	return quad(normKern_magic, tmin, tmax, args=(gn, taun, g1, tau1, g2, tau2))[0]

def FineTuneSolution(tau, t, Gexp, estimateError=False):
	"""Given a spacing of modes tau, tries to do NLLS to fine tune it further
	   If it fails, then it returns the old tau back
	   
	   Uses helper function: res_tG which computes residuals
	   """
	   
	try:
		res = least_squares(res_tG, tau, bounds=(0., np.inf),	args=(t, Gexp))
		tau = res.x
	
		# Error Estimate	
		if estimateError:
			tau0 = tau.copy()
			J = res.jac
			cov = np.linalg.pinv(J.T.dot(J)) * (res.fun**2).mean()
			dtau = np.sqrt(np.diag(cov))
		
	except:	
		pass

	g, tau, _, _ = MaxwellModes(np.log(tau), t, Gexp)   # Get g_i, taui

	if estimateError:
		
		for i in range(len(tau0)):
			if np.min(np.abs(tau0[i] - tau)) > 1e-12:
				dtau = np.delete(dtau, i)

		return g, tau, dtau
	else:
		return g, tau

def res_tG(tau, texp, Gexp):
	"""
		Helper function for final optimization problem
	"""
	g, _, _ = nnLLS(texp, tau, Gexp)
	Gmodel  = np.zeros(len(texp))

	for j in range(len(g)):
		Gmodel += g[j] * np.exp(-texp/tau[j])
		
	residual = Gmodel/Gexp - 1.
        
	return residual

def getDiscSpecMagic(par):
	"""
	# Function: getDiscSpecMagic(par)
	#
	# Uses the continuous relaxation spectrum extracted using getContSpec()
	# to determine an approximate discrete approximation.
	#
	# Input: Communicated by the datastructure "par"
	#
	# Output: Nopt    = optimum number of discrete modes
	#         [g tau] = spectrum
	#         error   = error norm of the discrete fit
	#        
	#         dmodes.dat : Prints the [g tau] for the particular Nopt
	#         aic.dat    : [N error aic]
	#         Gfitd.dat  : The discrete G(t) for Nopt [t Gt]"""

	t, Gexp, s, H, Nv, Gc, Cerror = initializeDiscSpec(par)
	
	n    = len(t);
	ns   = len(s);
	npts = len(Nv)

	# range of wtBaseDist scanned
	wtBase = par['deltaBaseWeightDist'] * np.arange(1, 1./par['deltaBaseWeightDist'])
	AICbst = np.zeros(len(wtBase))
	Nbst   = np.zeros(len(wtBase))
	nzNbst = np.zeros(len(wtBase))  # number of nonzeros
	
	# main loop over wtBaseDist
	for ib, wb in enumerate(wtBase):
		
		#
		# Find the distribution of nodes you need
		#
		wt  = GetWeights(H, t, s, wb)

		# 
		# Scan the range of number of Maxwell modes N = (Nmin, Nmax) 
		#
		ev    = np.zeros(npts)
		nzNv  = np.zeros(npts)  # number of nonzero modes 

		for i, N in enumerate(Nv):
			z, hz  = GridDensity(np.log(s), wt, N)     # Select "tau" Points
			g, tau, ev[i], _ = MaxwellModes(z, t, Gexp)
			nzNv[i]          = len(g)

		# store the best solution for this particular wb
		AIC        = 2. * Nv + 2. * Cerror * ev

		#
		# Fine-Tune the best in class-fit further by trying an NLLS optimization on it.
		#		
		N      = Nv[np.argmin(AIC)]
		z, hz  = GridDensity(np.log(s), wt, N)     		# Select "tau" Points
		g, tau, error, cKp = MaxwellModes(z, t, Gexp)   # Get g_i, taui
		g, tau = FineTuneSolution(tau, t, Gexp)


		AICbst[ib] = min(AIC)
		Nbst[ib]   = Nv[np.argmin(AIC)]
		nzNbst[ib] = nzNv[np.argmin(AIC)]
		
	# global best settings of wb and Nopt; note this is nominal Nopt (!= len(g) due to NNLS)
	
	Nopt  = int(Nbst[np.argmin(AICbst)])
	wbopt = wtBase[np.argmin(AICbst)]

	#
	# Recompute the best data-set stats, and fine tune it
	#
	wt           = GetWeights(H, t, s, wbopt)	
	z, hz        = GridDensity(np.log(s), wt, Nopt)           # Select "tau" Points
	g, tau, _, _ = MaxwellModes(z, t, Gexp)   # Get g_i, taui	
	g, tau, dtau = FineTuneSolution(tau, t, Gexp, estimateError=True)

	#
	# Check if modes are close enough to merge
	#
	indx       = np.argsort(tau)
	tau        = tau[indx]
	g          = g[indx]
	tauSpacing = tau[1:]/tau[:-1]
	itry       = 0

	while min(tauSpacing) < par['minTauSpacing'] and itry < 3:
		print("\tTau Spacing < minTauSpacing")

		imode   = np.argmin(tauSpacing)      # merge modes imode and imode + 1	
		tau     = mergeModes_magic(g, tau, imode, t, Gexp)
	
		g, tau, dtau  = FineTuneSolution(tau, t, Gexp, estimateError=True)
				
		tauSpacing = tau[1:]/tau[:-1]
		itry      += 1

	if par['verbose']:
		print('\n(*) Number of optimum nodes = {0:d}\n'.format(len(g)))

	#
	# Some Plotting
	#

	if par['plotting']:

		plt.clf()
		plt.plot(wtBase, AICbst, label='AIC')
		plt.plot(wtBase, nzNbst, label='Nbst')
		#~ plt.scatter(wbopt, len(g), color='k')
		plt.axvline(x=wbopt, color='gray')
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
		print('  i \t    g(i) \t    tau(i)\t    dtau(i)\n')
		print('-----------------------------------------------------\n')
		
		for i in range(len(g)):
			print('{0:3d} \t {1:.5e} \t {2:.5e} \t {3:.5e}'.format(i+1,g[i],tau[i], dtau[i]))
		print("\n")

		np.savetxt('output/dmodes.dat', np.c_[g, tau, dtau], fmt='%e')
		np.savetxt('output/aic.dat', np.c_[wtBase, nzNbst, AICbst], fmt='%f\t%i\t%e')

		S, T    = np.meshgrid(tau, t)
		K		= np.exp(-T/S)		# n * nmodes			
		GtM   	= np.dot(K, g)
		np.savetxt('output/Gfitd.dat', np.c_[t, GtM], fmt='%e')


	return Nopt, g, tau, error

#############################
#
# M A I N  P R O G R A M
#
#############################

if __name__ == '__main__':
	#
	# Read input parameters from file "inp.dat"
	#
	par = readInput('inp.dat')
	_ = getDiscSpecMagic(par)	
