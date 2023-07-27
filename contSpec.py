#
# 7/2023: allowing an optional weight column in the input data file
#         improving encapsulation of functions

# Help to find continuous spectrum
# March 2019 major update:
# (i)   added plateau modulus G0 (also in pyReSpect-time) calculation
# (ii)  following Hansen Bayesian interpretation of Tikhonov to extract p(lambda)
# (iii) simplifying lcurve (starting from high lambda to low)
# (iv)  changing definition of rho2 and eta2 (no longer dividing by 1/n and 1/nl)

from common import *

# HELPER FUNCTIONS

# def InitializeH(Gexp, s, kernMat, *argv):
def InitializeH(Gexp, wexp, s, kernMat, *argv):
    """
    Function: InitializeH(input)
    
    Input:  Gexp       = n*1 vector [Gt],
            wexp       = n*1 weight vector,
               s       = relaxation modes,
               kernMat = matrix for faster kernel evaluation
               G0      = optional; if plateau is nonzero
               
     Output:   H = guessed H
              G0 = optional guess if *argv is nonempty
    """
    #
    # To guess spectrum, pick a negative Hgs and a large value of lambda to get a
    # solution that is most determined by the regularization
    # March 2019; a single guess is good enough now, because going from large lambda to small
    #             lambda in lcurve.

    H    = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)
    lam  = 1e0
    
    if len(argv) > 0:
        G0       = argv[0]
        Hlam, G0 = getH(lam, Gexp, wexp, H, kernMat, G0)		
        return Hlam, G0
    else:
        Hlam     = getH(lam, Gexp, wexp, H, kernMat)
        return Hlam

def getAmatrix(ns):
    """Generate symmetric matrix A = L' * L required for error analysis:
       helper function for lcurve in error determination"""
    # L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
    nl = ns - 2
    L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
    L  = L[1:nl+1,:]
            
    return np.dot(L.T, L)
    
def getBmatrix(H, kernMat, Gexp, wexp, *argv):
    """get the Bmatrix required for error analysis; helper for lcurve()
       not explicitly accounting for G0 in Jr because otherwise I get underflow problems"""
    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;
    r   = np.zeros(n);   	  # vector of size (n);

    # furnish relevant portion of Jacobian and residual

    # Kmatrix = np.dot((1./Gexp).reshape(n,1), np.ones((1,ns)));
    Kmatrix = np.dot((wexp/Gexp).reshape(n,1), np.ones((1,ns)))
    Jr      = -kernelD(H, kernMat) * Kmatrix;    

    # if plateau then unfurl G0
    if len(argv) > 0:
        G0 = argv[0]
        # r  = (1. - kernel_prestore(H, kernMat, G0)/Gexp)
        r  = wexp * (1. - kernel_prestore(H, kernMat, G0)/Gexp)

    else:
        # r = (1. - kernel_prestore(H, kernMat)/Gexp)
        r = wexp * (1. - kernel_prestore(H, kernMat)/Gexp)
    
    B = np.dot(Jr.T, Jr) + np.diag(np.dot(r.T, Jr))

    return B


def lcurve(Gexp, wexp, Hgs, kernMat, par, *argv):

    """ 
     Function: lcurve(input)
    
     Input: Gexp    = n*1 vector [Gt],
             wexp    = weights associated with datapoints
            Hgs     = guessed H,
            kernMat = matrix for faster kernel evaluation
            par     = parameter dictionary
            G0      = optionally

     Output: lamC and 3 vectors of size npoints*1 contains a range of lambda, rho
             and eta. "Elbow"  = lamC is estimated using a *NEW* heuristic AND by Hansen method
             
             
    March 2019: starting from large lambda to small cuts calculation time by a lot
                also gives an error estimate 
             
    """
    if par['plateau']:
        G0 = argv[0]
        

    npoints = int(par['lamDensity'] * (np.log10(par['lam_max']) - np.log10(par['lam_min'])))
    hlam    = (par['lam_max']/par['lam_min'])**(1./(npoints-1.))	
    lam     = par['lam_min'] * hlam**np.arange(npoints)
    eta     = np.zeros(npoints)
    rho     = np.zeros(npoints)
    logP    = np.zeros(npoints)
    H       = Hgs.copy()
    n       = len(Gexp)
    ns      = len(H)
    nl      = ns - 2
    logPmax = -np.inf					# so nothing surprises me!
    Hlambda = np.zeros((ns, npoints))

    # Error Analysis: Furnish A_matrix
    Amat       = getAmatrix(len(H))
    _, LogDetN = np.linalg.slogdet(Amat)
        
    #
    # This is the costliest step
    #
    for i in reversed(range(len(lam))):
        
        lamb    = lam[i]
        
        if par['plateau']:
            H, G0   = getH(lamb, Gexp, wexp, H, kernMat, G0)			
            # rho[i]  = np.linalg.norm((1. - kernel_prestore(H, kernMat, G0)/Gexp))
            rho[i]  = np.linalg.norm(wexp*(1. - kernel_prestore(H, kernMat, G0)/Gexp))
            Bmat    = getBmatrix(H, kernMat, Gexp, wexp, G0)			
        else:
            H       = getH(lamb, Gexp, wexp, H, kernMat)
            # rho[i]  = np.linalg.norm((1. - kernel_prestore(H,kernMat)/Gexp))
            rho[i]  = np.linalg.norm(wexp*(1. - kernel_prestore(H, kernMat)/Gexp))
            Bmat    = getBmatrix(H, kernMat, Gexp, wexp)

        eta[i]       = np.linalg.norm(np.diff(H, n=2))
        Hlambda[:,i] = H


        _, LogDetC = np.linalg.slogdet(lamb*Amat + Bmat)
        V          =  rho[i]**2 + lamb * eta[i]**2		
                    
        # this assumes a prior exp(-lam)
        logP[i]    = -V + 0.5 * (LogDetN + ns*np.log(lamb) - LogDetC) - lamb
        
        if(logP[i] > logPmax):
            logPmax = logP[i]
        elif(logP[i] < logPmax - 18):
            break

    # truncate all to significant lambda
    lam  = lam[i:]
    logP = logP[i:]
    eta  = eta[i:]
    rho  = rho[i:]
    logP = logP - max(logP)

    Hlambda = Hlambda[:,i:]
    
    #
    # currently using both schemes to get optimal lamC
    # new lamM works better with actual experimental data  
    #
    # lamC = oldLamC(par, lam, rho, eta)
    plam = np.exp(logP); plam = plam/np.sum(plam)
    lamM = np.exp(np.sum(plam*np.log(lam)))

    #
    # Dialling in the Smoothness Factor
    #
    if par['SmFacLam'] > 0:
        lamM = np.exp(np.log(lamM) + par['SmFacLam']*(max(np.log(lam)) - np.log(lamM)));
    elif par['SmFacLam'] < 0:
        lamM = np.exp(np.log(lamM) + par['SmFacLam']*(np.log(lamM) - min(np.log(lam))));

    #
    # printing this here for now because storing lamC for sometime only
    #
    if par['plotting']:
        plt.clf()
        # plt.axvline(x=lamC, c='k', label=r'$\lambda_c$')
        plt.axvline(x=lamM, c='gray', label=r'$\lambda_m$')
        plt.ylim(-20,1)
        plt.plot(lam, logP, 'o-')
        plt.xscale('log')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\log\,p(\lambda)$')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('output/logP.pdf')

    return lamM, lam, rho, eta, logP, Hlambda


def getH(lam, Gexp, wexp, H, kernMat, *argv):

    """Purpose: Given a lambda, this function finds the H_lambda(s) that minimizes V(lambda)
    
              V(lambda) := ||(Gexp - kernel(H)) * (wexp/Gexp)||^2 +  lambda * ||L H||^2
    
     Input  : lambda  = regularization parameter,
              Gexp    = experimental data,
              wexp    = weighting factors,
              H       = guessed H,
                kernMat = matrix for faster kernel evaluation
                G0      = optional
    
     Output : H_lam, [G0]
              Default uses Trust-Region Method with Jacobian supplied by jacobianLM
    """

    # send Hplus = [H, G0], on return unpack H and G0
    if len(argv) > 0:
        Hplus= np.append(H, argv[0])
        res_lsq = least_squares(residualLM, Hplus, jac=jacobianLM, args=(lam, Gexp, wexp, kernMat))
        return res_lsq.x[:-1], res_lsq.x[-1]
        
    # send normal H, and collect optimized H back
    else:
        res_lsq = least_squares(residualLM, H, jac=jacobianLM, args=(lam, Gexp, wexp, kernMat))			
        return res_lsq.x


def residualLM(H, lam, Gexp, wexp, kernMat):
    """
    %
    % HELPER FUNCTION: Gets Residuals r
     Input  : H       = guessed H,
              lambda  = regularization parameter ,
              Gexp    = experimental data,
              wexp    = weighting factors,
              kernMat = matrix for faster kernel evaluation
                G0      = plateau
    
     Output : a set of n+nl residuals,
              the first n correspond to the kernel
              the last  nl correspond to the smoothness criterion
    %"""


    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;

    r   = np.zeros(n + nl);
    
    # if plateau then unfurl G0
    if len(H) > ns:
        G0     = H[-1]
        H      = H[:-1]
        # r[0:n] = (1. - kernel_prestore(H, kernMat, G0)/Gexp)  # the Gt and
        r[0:n] = wexp * (1. - kernel_prestore(H, kernMat, G0)/Gexp)  # the Gt and
    else:
        # r[0:n] = (1. - kernel_prestore(H, kernMat)/Gexp)
        r[0:n] = wexp * (1. - kernel_prestore(H, kernMat)/Gexp)
    
    # the curvature constraint is not affected by G0
    r[n:n+nl] = np.sqrt(lam) * np.diff(H, n=2)  # second derivative

        
    return r
        
def jacobianLM(H, lam, Gexp, wexp, kernMat):
    """
    HELPER FUNCTION for optimization: Get Jacobian J
    
    returns a (n+nl * ns) matrix Jr; (ns + 1) if G0 is also supplied.
    
    Jr_(i, j) = dr_i/dH_j
    
    It uses kernelD, which approximates dK_i/dH_j, where K is the kernel
    
    """
    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;

    # L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
    L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
    L  = L[1:nl+1,:]	
    
    # Furnish the Jacobian Jr (n+ns)*ns matrix
    # Kmatrix         = np.dot((1./Gexp).reshape(n,1), np.ones((1,ns)));
    Kmatrix         = np.dot((wexp/Gexp).reshape(n,1), np.ones((1,ns)));

    if len(H) > ns:

        G0     = H[-1]
        H      = H[:-1]
        
        Jr  = np.zeros((n + nl, ns+1))

        Jr[0:n, 0:ns]   = -kernelD(H, kernMat) * Kmatrix;
        # Jr[0:n, ns]     = -1./Gexp							# column for dr_i/dG0
        Jr[0:n, ns]     = -wexp/Gexp							# column for dr_i/dG0

        Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;
        Jr[n:n+nl, ns]  = np.zeros(nl)						# column for dr_i/dG0 = 0
        
    else:

        Jr  = np.zeros((n + nl, ns))

        Jr[0:n, 0:ns]   = -kernelD(H, kernMat) * Kmatrix;
        Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;        
         
    return	Jr

def kernelD(H, kernMat):
    """
     Function: kernelD(input)
    
     outputs the (n*ns) dimensional matrix DK(H)(t)
     It approximates dK_i/dH_j = K * e(H_j):
    
     Input: H       = substituted CRS,
            kernMat = matrix for faster kernel evaluation
    
     Output: DK = Jacobian of H
    """
    
    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];


    # A n*ns matrix with all the rows = H'
    Hsuper  = np.dot(np.ones((n,1)), np.exp(H).reshape(1, ns))  
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

    # t, Gexp = GetExpData(par['GexpFile'])
    t, Gexp, wexp = GetExpData(par['GexpFile'])

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
    
    kernMat = getKernMat(s, t)
    tic     = time.time()
    
    # get an initial guess for Hgs, G0
    if par['plateau']:
        Hgs, G0  = InitializeH(Gexp, wexp, s, kernMat, np.min(Gexp))		
    else:
        Hgs      = InitializeH(Gexp, wexp, s, kernMat)
    
    if par['verbose']:
        te   = time.time() - tic
        print('\t({0:.1f} seconds)\n(*) Building the L-curve ...'.format(te), end="")	
        tic  = time.time()

    # Find Optimum Lambda with 'lcurve'
    if par['lamC'] == 0:
        if par['plateau']:
            lamC, lam, rho, eta, logP, Hlam = lcurve(Gexp, wexp, Hgs, kernMat, par, G0)
        else:
            lamC, lam, rho, eta, logP, Hlam = lcurve(Gexp, wexp, Hgs, kernMat, par)
    else:
        lamC = par['lamC']

    if par['verbose']:
        te = time.time() - tic
        print('({1:.1f} seconds)\n(*) Extracting CRS, ...\n\t... lamC = {0:0.3e}; '.
              format(lamC, te), end="")
        
        tic  = time.time()

    # Get the best spectrum	
    if par['plateau']:
        H, G0  = getH(lamC, Gexp, wexp, Hgs, kernMat, G0);
        print('G0 = {0:0.3e} ...'.format(G0), end="")
    else:
        H  = getH(lamC, Gexp, wexp, Hgs, kernMat);

    #----------------------
    # Print some datafiles
    #----------------------

    if par['verbose']:
        te = time.time() - tic
        print('done ({0:.1f} seconds)\n(*) Writing and Printing, ...'.format(te), end="")

        # Save inferred G(t)
        if par['plateau']:
            K   = kernel_prestore(H, kernMat, G0);	
            np.savetxt('output/H.dat', np.c_[s, H], fmt='%e', header='G0 = {0:0.3e}'.format(G0))
        else:
            K   = kernel_prestore(H, kernMat);
            np.savetxt('output/H.dat', np.c_[s, H], fmt='%e')
            
        np.savetxt('output/Gfit.dat', np.c_[t, K], fmt='%e')

        # print Hlam, rho-eta, and logP if lcurve has been visited
        if par['lamC'] == 0:
            if os.path.exists("output/Hlam.dat"):
                os.remove("output/Hlam.dat")
                
        fHlam = open('output/Hlam.dat','ab')
        for i, lamb in enumerate(lam):
            np.savetxt(fHlam, Hlam[:,i])	
        fHlam.close()	

        # print logP
        np.savetxt('output/logPlam.dat', np.c_[lam, logP])
        
        # print rho-eta
        np.savetxt('output/rho-eta.dat', np.c_[lam, rho, eta], fmt='%e')

    #------------
    # Graphing
    #------------

    if par['plotting']:

        # plot spectrum "H.pdf" with errorbars
        plt.clf()

        plt.semilogx(s,H,'o-')
        plt.xlabel(r'$s$')
        plt.ylabel(r'$H(s)$')

        # error bounds are only available if lcurve has been implemented
        if par['lamC'] == 0:
            plam = np.exp(logP); plam = plam/np.sum(plam)			
            Hm   = np.zeros(len(s))
            Hm2  = np.zeros(len(s))
            cnt  = 0
            for i in range(len(lam)):	
                #~ Hm   += plam[i]*Hlam[:,i]
                #~ Hm2  += plam[i]*Hlam[:,i]**2
                # count all spectra within a threshold
                if plam[i] > 0.1:
                    Hm   += Hlam[:,i]
                    Hm2  += Hlam[:,i]**2
                    cnt  += 1

            Hm = Hm/cnt
            dH = np.sqrt(Hm2/cnt - Hm**2)

            plt.semilogx(s,Hm+2.5*dH, c='gray', alpha=0.5)
            plt.semilogx(s,Hm-2.5*dH, c='gray', alpha=0.5)

        plt.tight_layout()
        plt.savefig('output/H.pdf')


        #
        # plot comparison with input spectrum
        #

        plt.clf()

        if par['plateau']:
            K   = kernel_prestore(H, kernMat, G0);	
        else:
            K   = kernel_prestore(H, kernMat);

        plt.loglog(t, Gexp,'o',t, K, 'k-')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$G(t)$')
        plt.tight_layout()
        plt.savefig('output/Gfit.pdf')


        #
        # if lam not explicitly specified then print rho-eta.pdf
        #

        try:
            lam
        except NameError:
          print("lamC prespecified, so not printing rho-eta.pdf/dat")
        else:
            plt.clf()
            plt.scatter(rho, eta, marker='x')
            plt.plot(rho, eta)


            rhost = np.exp(np.interp(np.log(lamC), np.log(lam), np.log(rho)))
            etast = np.exp(np.interp(np.log(lamC), np.log(lam), np.log(eta)))

            plt.plot(rhost, etast, 'o', color='k')
            plt.xscale('log')
            plt.yscale('log')
            
            #~ print(rhost, etast)
            
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
    t, Gexp, wG = GetExpData(par['GexpFile'])
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
