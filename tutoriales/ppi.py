# -*- coding: utf-8 -*-
"""Economic Complexity and Sustainable Development: 
    A Computational Framework for Policy Priorities

Code to run the computational model of the book.

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Pyhton 3.7


Example
-------



Rquired external libraries
--------------------------
- Numpy


"""

# import necessary libraries
from __future__ import division, print_function
import numpy as np
import warnings
warnings.simplefilter("ignore")


def run_ppi(I0, alphas, betas, A=None, R=None, bs=None, qm=None, rl=None,
            Imax=None, Bs=None, B_dict=None, G=None, T=50, scalar=1., frontier=None):
    """Function to run one simulation of the Policy Priority Inference model.

    Parameters
    ----------
        I0: numpy array 
            Initial values of the development indicators.
        alphas: numpy array
            Vector with parameters representing the structural 
            factors of each indicator. If not provided, the structurral factors are
            assumed to have value 1 for all indicators.
        betas: numpy array
            Vector with parameters that normalize the public
            in an indicator into a success probabiity for its growth trial.
        A:  2D numpy array (default: a matrix full of zeros)
            The adjacency matrix of the spillover network of development 
            indicators. If not given, the model assumes a zero-matrix, so there 
            are no spillovers.
        R:  numpy array (default: a vector with ones)
            Vector that specifies whether an indicator is instrumental (1 or True)
            or collateral (0 or False).
        bs: numpy array (default: a vector full of ones)
            Vector with modulating factors for the budgetary allocation of
            each instrumental indicator.
        qm: numpy array (default: a vector full of 0.5)
            Vector with parametrs capturing the quality of the monitoring 
            mechanisms for each instrumental indicator. If the quality of 
            monitoring is an indicator for the model, then qm should be an integer
            corresponding to the index of the indicator.
        rl: numpy array (default: a vector full of 0.5)
            Vector with parametrs capturing the quality of the rule of law 
            mechanisms for each instrumental indicator. If the rule of 
            law is an indicator for the model, then rl should be an integer
            corresponding to the index of the indicator.
        Imax: numpy array
            Vector with the technical upper bounds of the indicators. It an entry
            contains a missing value (NaN), then there is no upper bound defined
            for that indicator and it will grow indefinitely.
        Bs: numpy ndarray (default: a vector with 10e6 in every entry)
            Disbursement schedule across expenditure programs for the entire
            simulation. In its 2D form, rows represent the expenditure programs 
            and columnts the simulation periods. If provided as a single number 
            or as a single-dimensional vector, it is assumed that there
            is only one expenditure program: the entire budget.
        B_dict: dictionary (keys must be integers and values must be lists)
            A dictionary that maps the indices of every indicator into a list
            with the associated expenditure programs. Every instrumental indicator
            must be associated to at least one expenditure program. Collateral
            indicators shuold map into empty lists. The expenditure programs
            should be integers matching the indices of the rows in Bs when 
            provided as a 2D array.
        G: numpy array
            The development goals to be achieved for each indicator.
        T: int
            The maximum number of simulation periods. If Bs is provided, then T
            is overwritten by the number of periods implied by the dirsbursement 
            schedule.
        scalar: float
            The factor by which the indicators may be scaled. It is necessary 
            if the indicators (I0 and Imax) are not in the [0,1] range.
        frontier: numpy array
            
        
    Returns
    -------
        tsI: 2D numpy array
            Matrix with the time series of the simulated indicators. Each column 
            corresponds to a simulation step.
        tsC: 2D numpy array
            Matrix with the time series of the simulated contributions. Each column 
            corresponds to a simulation step.
    """
    
    
    
    
    ## SET DEFAULT PARAMETERS & CHECK INPUT INTEGRITY 
    
    # Number of indicators
    assert np.sum(np.isnan(I0)) == 0, 'The indicator levels "I0" values contain missing values'
    N = len(I0) 
    
    # Structural factors
    assert np.sum(np.isnan(alphas)) == 0, 'The structural factors "alphas" contain missing values'
    assert len(alphas) == N, 'The structural factors "alphas" should contain as many elements as indicators'
    
    # Normalizing factors
    assert np.sum(np.isnan(betas)) == 0, 'The normalizing factors "betas" contain missing values'
    assert len(alphas) == N, 'The normalizing factors "betas" should contain as many elements as indicators'
    
    # Spillover network
    if A is None:
        A = np.zeros((N,N))
    else:
        assert np.sum(np.isnan(A)) == 0, 'The spillover network "A" contains missing values'
        assert len(A.shape) == 2 and A.shape[0] == N and A.shape[1] == N, 'The spillover network "A" has to be represented by an adjacency matrix of size NxN'
        A = A.copy()
        np.fill_diagonal(A, 0) # make sure there are no self-loops
    
    # Instrumental indicators
    if R is None:
        R = np.ones(N).astype(bool)
    else:
        R[R!=1] = 0
        R = R.astype(bool)
        assert np.sum(R) > 0, 'At least one instrumental indicator is needed'
        assert len(R) == N, 'Vector "R" should contain as many elements as indicators'
    
    # Number of instrumental indicators
    n = int(R.sum())
        
    # Modulating factors
    if bs is None:
        bs = np.ones(n)
    else:
        assert np.sum(np.isnan(bs)) == 0, 'The modulating factors "bs" contain missing values'
        assert len(bs) == n, 'The number of modulating factors "bs" should contain as many elements as instrumental indicators'
        
    # Quality of monitoring
    if qm is None:
        qm = np.ones(n)*.5
    else:
        assert np.sum(np.isnan(qm)) == 0, 'The monitoring-quality parameters "qm" contain missing values'
        assert len(qm) == n, 'The number of monitoring-quality parameters "qm" should contain as many elements as instrumental indicators'
        
    # Quality of the rule of law
    if rl is None:
        rl = np.ones(n)*.5
    else:
        assert np.sum(np.isnan(rl)) == 0, 'The rule-of-law parameters "rl" contain missing values'
        assert len(rl) == n, 'The number of rule-of-law parameters "rl" should contain as many elements as instrumental indicators'
        
    # Technical upper bounds
    if Imax is not None:
        assert len(Imax) == N, 'The number of maximum theoretical values needs to be the same as indicators'
        if np.sum(~np.isnan(Imax)) > 0:
            assert np.sum(Imax[~np.isnan(Imax)] < I0[~np.isnan(Imax)]) == 0, 'All technical upper bounds must be larger than the initial values of their indicators'

    # Payment schedule
    if Bs is None:
        Bs = np.array([np.ones(T)*10e6])
        B_dict = dict([(i,[0]) for i in range(N) if R[i]==1])
    elif type(Bs) is np.ndarray and len(Bs.shape) == 1:
        Bs = np.array([Bs])
        B_dict = dict([(i,[0]) for i in range(N) if R[i]==1])
        T = Bs.shape[0]
    
    assert np.sum(np.isnan(Bs)) == 0, 'The payment schedule contains missing values'
    T = Bs.shape[1]
    # Dictionary linking indicators to expenditure programs
    assert B_dict is not None, 'You must provide a budget dictionary linking the instrumental indicators to the expenditure programs in the disbursement schedule'
    assert len(B_dict) == n, 'The budget dictionary must have entries for every instrumental indicator'
    assert np.sum(np.in1d(np.array(list(B_dict.keys())), np.arange(N))) == n, 'The budget dictionary keys do not match the indicators indices (they must be between 0 and N-1)'
    assert sum([type(val) is list for val in B_dict.values()]) == n, 'Every value in the budget dictionary must be a list'
    assert sum([True for i in range(N) if R[i] and i not in B_dict]) == 0, 'There are instrumental indicators without expenditure programs in the budget dictionary'
    assert sum([True for i in B_dict.keys() if not R[i]]) == 0, 'Collateral indicators cannot be linked to expenditure programs in the budget dictionary'
    # Create reverse disctionary linking expenditure programs to indicators
    programs = sorted(np.unique([item for sublist in B_dict.values() for item in sublist]).tolist())
    program2indis = dict([(program, []) for program in programs])
    sorted_programs = sorted(program2indis.keys())
    for indi, programs in B_dict.items():
        for program in programs:
            if R[indi]:
                program2indis[program].append( indi )
    inst2idx = np.ones(N)*np.nan
    inst2idx[R] = np.arange(n)
    # Create initial allocation profile
    if G is not None:
        gaps = G-I0
        gaps[G<I0] = 0
        p0 = gaps/gaps.sum()
        P0 = np.zeros(n)
    else:
        P0 = np.zeros(n)
        p0 = np.random.rand(n)
    i=0
    for program in sorted_programs:
        indis = program2indis[program]
        relevant_indis = inst2idx[indis].astype(int)
        P0[relevant_indis] += Bs[i,0]*p0[relevant_indis]/p0[relevant_indis].sum()
        i+=1

    
    
    # Prevent null allocations
    Bs[Bs==0] = 10e-12
    P0[P0==0] = 10e-12
    
    
    
    ## INSTANTIATE ALL VARIABLES AND CREATE CONTAINERS TO STORE DATA
        
    P = P0.copy()
    F = np.random.rand(n) # policymakers' benefits
    Ft = np.random.rand(n) # lagged benefits
    X = np.random.rand(n)-.5 # policymakers' actions
    Xt = np.random.rand(n)-.5 # lagged actions
    H = np.ones(n) # cumulative spotted inefficiencies
    HC = np.ones(n) # number of times spotted so far
    signt = np.sign(np.random.rand(n)-.5) # directions of previous actions
    changeFt = np.random.rand(n)-.5 # changes in benefits
    C = np.random.rand(n)*P # contributions
    I = I0.copy() # initial levels of the indicators
    It = np.random.rand(N)*I # lagged indicators

    tsI = [] # stores time series of indicators
    tsC = [] # stores time series of contributions
    tsF = [] # stores time series of benefits
    tsP = [] # stores time series of allocations
    tsS = [] # stores time series of spillovers
    tsG = [] # stores time series of gammas
    
    step = 0
    for t in range(T):
        
        step = min([step+1, T-1]) # increase the payment period
        tsI.append(I.copy()) # store this period's indicators
        tsP.append(P.copy()) # store this period's allocations


        ### REGISTER CHANGES ###
        deltaBin = (I>It).astype(int) # binary for the indicators' improvements
        deltaIIns = (I[R]-It[R]).copy() # instrumental indicators' changes
        if np.sum(np.abs(deltaIIns)) > 0: # relative change of instrumental indicators
            deltaIIns = deltaIIns/np.sum(np.abs(deltaIIns))
        

        ### DETERMINE CONTRIBUTIONS ###
        changeF = F - Ft # change in benefits
        changeX = X - Xt # change in actions
        sign = np.sign(changeF*changeX) # direction of the next action
        changeF[changeF==0] = changeFt[changeF==0] # if the benefit did not change, keep the last change
        sign[sign==0] = signt[sign==0] # if the sign is undefined, keep the last one
        Xt = X.copy() # update lagged actions
        X = X + sign*np.abs(changeF) # determine current action
        assert np.sum(np.isnan(X)) == 0, 'X has invalid values!'
        C = P/(1 + np.exp(-X)) # map action into contribution
        assert np.sum(np.isnan(C)) == 0, 'C has invalid values!'
        assert np.sum(P < C)==0, 'C larger than P!'
        signt = sign.copy() # update previous signs
        changeFt = changeF.copy() # update previous changes in benefits
        
        tsC.append(C.copy()) # store this period's contributions
        tsF.append(F.copy()) # store this period's benefits
                
        
        ### DETERMINE BENEFITS ###
        if type(qm) is int or type(qm) is np.int64:
            trial = (np.random.rand(n) < (I[qm]/scalar) * P/P.max() * (P-C)/P) # monitoring outcomes
        else:
            trial = (np.random.rand(n) < qm * P/P.max() * (P-C)/P)
        theta = trial.astype(float) # indicator function of uncovering inefficiencies
        H[theta==1] += (P[theta==1] - C[theta==1])/P[theta==1]
        HC[theta==1] += 1
        if type(rl) is int or type(rl) is np.int64:
            newF = deltaIIns*C/P + (1-theta*(I[rl]/scalar))*(P-C)/P # compute benefits
        else:
            newF = deltaIIns*C/P + (1-theta*rl)*(P-C)/P
        Ft = F.copy() # update lagged benefits
        F = newF # update benefits
        assert np.sum(np.isnan(F)) == 0, 'F has invalid values!'
        
        
        ### DETERMINE INDICATORS ###
        deltaM = np.array([deltaBin,]*len(deltaBin)).T # reshape deltaIAbs into a matrix
        S = np.sum(deltaM*A, axis=0) # compute spillovers
        assert np.sum(np.isnan(S)) == 0, 'S has invalid values!'
        tsS.append(S) # store spillovers
        cnorm = np.zeros(N) # initialize a zero-vector to store the normalized contributions
        cnorm[R] = C # compute contributions only for instrumental nodes
        gammas = ( betas*(cnorm + C.sum()/(P.sum()+1)) )/( 1 + np.exp(-S) ) # compute probability of succesful growth
        assert np.sum(np.isnan(gammas)) == 0, 'gammas has invalid values!'
        assert np.sum(gammas==0) == 0, 'some gammas have zero value!'
        
        if frontier is not None:
            gammas = frontier
        tsG.append(gammas) # store gammas
        success = (np.random.rand(N) < gammas).astype(int) # determine if there is succesful growrth
        newI = I + alphas * success # compute new indicators
        assert np.sum(newI < 0) == 0, 'indicators cannot be negative!'
        
        # if theoretical maximums are provided, make sure the indicators do not surpass them
        if Imax is not None:
            with_bound = ~np.isnan(Imax)
            newI[with_bound & (newI[with_bound] > Imax[with_bound])] = Imax[with_bound & (newI[with_bound] > Imax[with_bound])]
            assert np.sum(newI[with_bound] > Imax[with_bound])==0, 'some indicators have surpassed their theoretical upper bound!'
              
        # if governance parameters are endogenous, make sure they are not larger than 1
        if (type(qm) is int or type(qm) is np.int64) and newI[qm] > scalar:
            newI[qm] = scalar
        
        if (type(rl) is int or type(rl) is np.int64) and newI[rl] > scalar:
            newI[rl] = scalar
            
        It = I.copy() # update lagged indicators
        I =  newI.copy() # update indicators
        
        
        ### DETERMINE ALLOCATION PROFILE ###
        P0 += np.random.rand(n)*H/HC # interaction between random term and inefficiancy history
        assert np.sum(np.isnan(P0)) == 0, 'P0 has invalid values!'
        assert np.sum(P0==0) == 0, 'P0 has a zero value!'
        
        P = np.zeros(n)
        for i, program in enumerate(sorted_programs):
            indis = program2indis[program]
            relevant_indis = inst2idx[indis].astype(int)
            q = P0[relevant_indis]/P0[relevant_indis].sum()
            assert np.sum(np.isnan(q)) == 0, 'q has invalid values!'
            assert np.sum(q == 0 ) == 0, 'q has zero values!'
            qs_hat = q**bs[relevant_indis]
            qs_hat[qs_hat==1] = 10e-12
            P[relevant_indis] += Bs[i, step]*qs_hat/qs_hat.sum()
            i+=1
        assert abs( (P.sum() - Bs[:, step].sum())/Bs[:, step].sum() ) < .05, 'unequal budgets '+str(int(P.sum())) + ' '+str(int(Bs[:, step].sum()))
        assert np.sum(np.isnan(P)) == 0, 'P has invalid values!'
        assert np.sum(P==0) == 0, 'P has zero values!'



            
    return np.array(tsI).T, np.array(tsC).T, np.array(tsF).T, np.array(tsP).T, np.array(tsS).T, np.array(tsG).T



    







































