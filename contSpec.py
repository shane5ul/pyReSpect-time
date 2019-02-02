#
# Help to find continuous spectrum
#

from common import *

# HELPER FUNCTIONS

def InitializeH(Gexp, s, kernMat):
	"""
	Function: InitializeH(input)
	
	Input:  Gexp       = n*1 vector [Gt],
	           s       = relaxation modes,
			   kernMat = matrix for faster kernel evaluation
	 Output:   H = guessed H
	"""
	# To guess spectrum, pick a negative Hgs and a large value of lambda to get a
	# solution that is most determined by the regularization, then use that as
	# the next guess. 

	H    = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)

	lam  = 1e0
	Hlam = getH(lam, Gexp, H, kernMat)

	# Successively improve the initial guess until you have are reasonably good
	# guess for low lambda

	lam  = 1e-3;
	H    = getH(lam, Gexp, Hlam, kernMat)
	
	return H

def lcurve(Gexp, Hgs, kernMat, par):
	""" 
	 Function: lcurve(input)
	
	 Input: Gexp    = n*1 vector [Gt],
	        Hgs     = guessed H,
		    kernMat = matrix for faster kernel evaluation

	 Output: lamC and 3 vectors of size npoints*1 contains a range of lambda, rho
	         and eta. "Elbow"  = lamC is estimated using a *NEW* heuristic.
	         
	"""
	
	# take a coarse mesh: 2 lambda's per decade (auto)
	lam_max   = par['lam_max']
	lam_min   = par['lam_min']
	SmoothFac = par['SmFacLam']	

	npoints  = int(2 * (np.log10(lam_max) - np.log10(lam_min)))

	hlam    = (lam_max/lam_min)**(1./(npoints-1.))	
	lam     = lam_min * hlam**np.arange(npoints)

	eta     = np.zeros(npoints)
	rho     = np.zeros(npoints)
	H       = Hgs.copy()
	
	#
	# This is the costliest step
	#
	for i in range(len(lam)):
		lamb    = lam[i]
		H       = getH(lamb, Gexp, H, kernMat)
		rho[i]  = np.linalg.norm((1 - kernel_prestore(H,kernMat)/Gexp))/np.sqrt(len(Gexp))
		eta[i]  = np.linalg.norm(np.diff(H, n=2))/np.sqrt(len(H))
	#
	# 8/1/2018: Making newer strategy more accurate and robust: dividing by minimum rho/eta
	# which is not as sensitive to lam_min, lam_max. This makes lamC robust to range of lam explored
	#
	
	#er = rho/np.amin(rho) + eta/np.amin(eta);
	er    = rho/np.amin(rho) + eta/(np.sqrt(np.amax(eta)*np.amin(eta)));

	#
	# Since rho v/s lambda is smooth, we can interpolate the coarse mesh to find minimum
	#
	
	lami = np.logspace(np.log10(min(lam)), np.log10(max(lam)), 1000)
	erri = np.exp(interp1d(np.log(lam), np.log(er), kind='cubic')(np.log(lami)))

	ermin = np.amin(erri)
	eridx = np.argmin(erri)	
	lamC  = lami[eridx]
	

	#
	# 12/18; for extremely smooth data have cutoff at rho = 1e-2?
	#
	rhoF  = interp1d(lam, rho)

	if  rhoF(lamC) <= par['rho_cutoff']:
		try:
			eridx = (np.abs(rhoF(lami) - par['rho_cutoff'])).argmin()
			if lami[eridx] > lamC:
				lamC = lami[eridx]				
		except:
			pass





	# Dialling in the Smoothness Factor
	if SmoothFac > 0:
		lamC = np.exp(np.log(lamC) + SmoothFac*(np.log(lam_max) - np.log(lamC)));
	elif SmoothFac < 0:
		lamC = np.exp(np.log(lamC) + SmoothFac*(np.log(lamC) - np.log(lam_min)));

	return lamC, lam, rho, eta

def getH(lam, Gexp, H, kernMat):

	"""Purpose: Given a lambda, this function finds the H_lambda(s) that minimizes V(lambda)
	
	          V(lambda) := 1/n * ||Gexp - kernel(H)||^2 +  lambda/nl * ||L H||^2
	
	 Input  : lambda  = regularization parameter ,
	          Gexp    = experimental data,
	          H       = guessed H,
  		      kernMat = matrix for faster kernel evaluation
	
	 Output : H_lam
	          Default uses Trust-Region Method with Jacobian supplied by jacobianLM
	"""

	#~ res_lsq = least_squares(residualLM, H, args=(lam, Gexp, kernMat))
	res_lsq = least_squares(residualLM, H, jac=jacobianLM, args=(lam, Gexp, kernMat))

	return res_lsq.x

def residualLM(H, lam, Gexp, kernMat):
	"""
	%
	% HELPER FUNCTION: Gets Residuals r
	 Input  : H       = guessed H,
			  lambda  = regularization parameter ,
	          Gexp    = experimental data,
  		      kernMat = matrix for faster kernel evaluation
	
	 Output : H_lam
	          Default uses Trust-Region Method	
	
	%"""

	n   = kernMat.shape[0];
	ns  = kernMat.shape[1];
	nl  = ns - 2;
	r   = np.zeros(n + nl);

	# 
	# Get the residual vector first
	# r = vector of size (n+nl,1);
	#

	r[0:n]    = (1. - kernel_prestore(H,kernMat)/Gexp)/np.sqrt(n)  # the Gt and
	r[n:n+nl] = np.sqrt(lam) * np.diff(H, n=2)/np.sqrt(nl)  # second derivative
	
	return r
		
def jacobianLM(H, lam, Gexp, kernMat):
	"""
	 HELPER FUNCTION for optimization: Get Jacobian J
	"""

	n   = kernMat.shape[0];
	ns  = kernMat.shape[1];
	nl  = ns - 2;
	
	Jr  = np.zeros((n + nl,ns))

	#
	# L is a nl*ns tridiagonal matrix with 1
	# -2 and 1 on its diagonal. need to get L figured out, and perhaps not pass?
	#
	
	L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))	     
	L  = L[1:nl+1,:]
	 	
	#
	# Furnish the Jacobian Jr
	# (n+nl)*ns matrix
	#
	Kmatrix         = np.dot((1./Gexp).reshape(n,1), np.ones((1,ns)))/np.sqrt(n);
	Jr[0:n, 0:ns]   = -kernelD(H, kernMat) * Kmatrix;
	Jr[n:n+nl,0:ns] = np.sqrt(lam) * L/np.sqrt(nl);

	return	Jr

def kernelD(H, kernMat):
	"""
	 Function: kernelD(input)
	
	 outputs the n*ns dimensional vector DK(H)(t)
	 approximates dK/dHj
	
	 Input: H       = substituted CRS,
		    kernMat = matrix for faster kernel evaluation
	
	 Output: DK = Jacobian of H
	"""
	n   = kernMat.shape[0];
	ns  = kernMat.shape[1];


	Hsuper  = np.dot(np.ones((n,1)), np.exp(H).reshape(1, ns))  # A n*ns matrix with all the rows = H'
	DK      = kernMat  * Hsuper
		
	return DK
	
def getContSpec(par):
	"""
	This is the main driver routine for computing the continuous spectrum
	
	(*)   input  : "par" dictionary from "inp.dat" which specifies GexpFile (often 'Gt.dat')
	(*)   return : H and lambdaC; the latter can be used to microscpecify lambdaC as desired
	                without having to do the entire lcurve calculation again
	"""
	# read input
	if par['verbose']:
		print('\n(*) Start\n(*) Loading Data File: {}...'.format(par['GexpFile']))

	t, Gexp = GetExpData(par['GexpFile'])

	if par['verbose']:
		print('(*) Initial Set up...', end="")
  
	# Set up some internal variables
	n    = len(t)
	ns   = par['ns']    # discretization of 'tau'

	tmin = t[0];
	tmax = t[n-1];
	
	# determine frequency window
	if par['FreqEnd'] == 1:
		smin = np.exp(-np.pi/2) * tmin; smax = np.exp(np.pi/2) * tmax		
	elif par['FreqEnd'] == 2:
		smin = tmin; smax = tmax				
	elif par['FreqEnd'] == 3:
		smin = np.exp(+np.pi/2) * tmin; smax = np.exp(-np.pi/2) * tmax		

	hs   = (smax/smin)**(1./(ns-1))
	s    = smin * hs**np.arange(ns)

	kernMat = getKernMat(s,t)
	
	tic  = time.time()
	Hgs  = InitializeH(Gexp, s, kernMat)
		
	#~ plt.plot(s, Hgs)
	#~ plt.xscale('log')
	#~ plt.show()
	
	if par['verbose']:
		te   = time.time() - tic
		print('\t({0:.1f} seconds)\n(*) Building the L-curve ...'.format(te), end="")	
		tic  = time.time()

	#
	# Find Optimum Lambda with 'lcurve'
	#
	if par['lamC'] == 0:
		lamC, lam, rho, eta = lcurve(Gexp, Hgs, kernMat, par)
	else:
		lamC = par['lamC']


	if par['verbose']:
		te = time.time() - tic
		print('{0:0.3e} ({1:.1f} seconds)\n(*) Extracting the continuous spectrum, ...'.format(lamC, te), end="")
		tic  = time.time()

	#
	# Get the spectrum	
	#
	
	H  = getH(lamC, Gexp, Hgs, kernMat);

	#
	# Print some datafiles
	#

	if par['verbose']:
		te = time.time() - tic
		print('done ({0:.1f} seconds)\n(*) Writing and Printing, ...'.format(te), end="")

		# create output directory if none exists
		if not os.path.exists("output"):
			os.makedirs("output")

		np.savetxt('output/H.dat', np.c_[s, H], fmt='%e')
		
		K   = kernel_prestore(H, kernMat);	
		np.savetxt('output/Gfit.dat', np.c_[t, K], fmt='%e')

	#
	# Graphing
	#
	
	if par['plotting']:

		plt.clf()
		plt.semilogx(s,H,'o-')
		plt.xlabel(r'$s$')
		plt.ylabel(r'$H(s)$')
		plt.tight_layout()
		plt.savefig('output/H.pdf')

		plt.clf()
		K = kernel_prestore(H, kernMat)
		plt.loglog(t, Gexp,'o',t, K, 'k-')
		plt.xlabel(r'$t$')
		plt.ylabel(r'$G(t)$')
		plt.tight_layout()
		plt.savefig('output/Gfit.pdf')


		# if lam not explicitly specified then print rho-eta.pdf
		try:
			lam
		except NameError:
		  print("lamC prespecified, so not printing rho-eta.pdf/dat")
		else:
			
			np.savetxt('output/rho-eta.dat', np.c_[lam, rho, eta], fmt='%e')

			plt.clf()
			plt.plot(rho, eta, 'x-')

			rhost = np.exp(np.interp(np.log(lamC), np.log(lam), np.log(rho)))
			etast = np.exp(np.interp(np.log(lamC), np.log(lam), np.log(eta)))

			plt.plot(rhost, etast, 'o', color='k')
			plt.xscale('log')
			plt.yscale('log')
			
			
			plt.xlabel(r'$\rho$')
			plt.ylabel(r'$\eta$')
			plt.tight_layout()
			plt.savefig('output/rho-eta.pdf')

	if par['verbose']:
		print('done\n(*) End\n')
		
	return H, lamC

def guiFurnishGlobals(par):
	"""Furnish Globals to accelerate interactive plot in jupyter notebooks"""

	# plot settings
	from matplotlib import rcParams
	rcParams['axes.labelsize'] = 14 
	rcParams['xtick.labelsize'] = 12
	rcParams['ytick.labelsize'] = 12 
	rcParams['legend.fontsize'] = 12
	rcParams['lines.linewidth'] = 2

	# experimental data
	t, Gexp = GetExpData(par['GexpFile'])
	n    = len(t)
	ns   = par['ns']    # discretization of 'tau'

	tmin = t[0];
	tmax = t[n-1];

	# determine frequency window
	if par['FreqEnd'] == 1:
		smin = np.exp(-np.pi/2) * tmin; smax = np.exp(np.pi/2) * tmax		
	elif par['FreqEnd'] == 2:
		smin = tmin; smax = tmax
	elif par['FreqEnd'] == 3:
		smin = np.exp(+np.pi/2) * tmin; smax = np.exp(-np.pi/2) * tmax		

	hs   = (smax/smin)**(1./(ns-1))
	s    = smin * hs**np.arange(ns)
	kernMat = getKernMat(s,t)

	# toggle flags to prevent printing

	par['verbose'] = False
	par['plotting'] = False

	# load lamda, rho, eta
	lam, rho, eta = np.loadtxt('output/rho-eta.dat', unpack=True)


	plt.clf()

	return s, t, kernMat, Gexp, par, lam, rho, eta
    
#	 
# Main Driver: This part is not run when contSpec.py is imported as a module
#              For example as part of GUI
#
if __name__ == '__main__':
	#
	# Read input parameters from file "inp.dat"
	#
	par = readInput('inp.dat')
	H, lamC = getContSpec(par)
