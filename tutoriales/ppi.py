# -*- coding: utf-8 -*-
"""Policy Priority Inference for Sustainable Development

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Pyhton 3.7
Acknowledgments: This product was developed through the sponsorship of the
    United Nations Development Programme (bureau for Latin America) 
    and with the support of the National Laboratory for Public Policies (Mexico City), 
    the Centro de Investigación y Docencia Económica (CIDE, Mexico City), 
    and The Alan Turing Institute (London).


Example
-------
To run PPI in a Python script, just add the following line:

    tsI, tsC, tsF, tsP, tsD, tsS, ticks, H = run_ppi(I0, T)
    
This will simulate the policymaking process for initial values I0 and targets T.
This example assumes no network of spillovers. All other arguments can be
passed as explained in the function run_ppi.


Rquired external libraries
--------------------------
- Numpy


"""

# import necessary libraries
from __future__ import division, print_function
import numpy as np
import warnings
warnings.simplefilter("ignore")


def run_ppi(I0, A=None, R=None, alpha=.1, cc=1, rl=1, betas=None, get_gammas=False, force_gammas=False, 
            max_theo=None, scalar=1., budget_hash=None, B_sequence=None, conv_vals=None, multi_map=None):
    """Function to run one simulation of the Policy Priority Inference model.

    Parameters
    ----------
        I0: numpy array 
            Initial values of the development indicators.
        A:  2D numpy array
            The adjacency matrix of the spillover network of development 
            indicators. If not given, the model assumes a zero-matrix, so there 
            are no spillovers.
        
    Returns
    -------
        tsI: 2D numpy array
            Matrix with the time series of the simulated indicators. Each column 
            corresponds to a simulation step.
        tsC: 2D numpy array
            Matrix with the time series of the simulated contributions. Each column 
            corresponds to a simulation step.
    """
    
    N = len(I0) # number of indicators
    
    ## Check data integrity
    assert np.sum(np.isnan(I0)) == 0, 'Initial values must be valid numbers'
    if max_theo is not None:
        assert len(max_theo) == N, 'The number of maximum theoretical values needs to be the same as indicators.'
    

    # if no network is provided, create a zero-matrix
    if A is None:
        A = np.zeros((N,N))
    else:
        assert np.sum(np.isnan(A)) == 0, 'The spillover network contains invalid values'
        A = A.copy()
        np.fill_diagonal(A, 0)
    
    if R is None:
        R = np.ones(N).astype(bool)
    else:
        R[R!=1] = 0
        R = R.astype(bool)
        assert np.sum(R) > 0, 'At least one instrumental indicator is needed'

    n = int(R.sum())

    tsI = [] # stores time series of indicators
    tsC = [] # stores time series of contributions
    tsF = [] # stores time series of benefits
    tsP = [] # stores time series of allocations
    tsD = [] # stores time series of corruption
    tsX = [] # stores time series of actions
    tsS = [] # stores time series of spillovers
    
    B_sequence[B_sequence==0] = 10e-12

    programs = sorted(np.unique([item for sublist in budget_hash.values() for item in sublist]).tolist())
    program2indis = dict([(program, []) for program in programs])
    P0 = np.zeros(n)
    
    for indi, programs in budget_hash.items():
        for program in programs:
            if R[indi] == 1:
                program2indis[program].append( indi )
    
    i=0
    p0 = np.random.rand(n)
    inst2idx = np.ones(N)*np.nan
    inst2idx[R] = np.arange(n)
    for program, indis in program2indis.items():
        P0[inst2idx[indis].astype(int)] += B_sequence[i,0]*p0[inst2idx[indis].astype(int)]/p0[inst2idx[indis].astype(int)].sum()
        max_steps = 50
        if len(B_sequence.shape) == 2:
            max_steps = B_sequence.shape[1]
        i+=1
    
    
    P0[P0==0] = 10e-12
    P = P0.copy()
    F = np.random.rand(n) # vector of benefits
    Ft = np.random.rand(n) # vectors of lagged benefits
    X = np.random.rand(n)-.5 # vector of actions
    Xt = np.random.rand(n)-.5 # vector of lagged actions
    H = np.ones(n) # vector of historical inefficiencies
    HC = np.ones(n)
    signt = np.sign(np.random.rand(n)-.5) # vector of previous signs for directed learning
    changeFt = np.random.rand(n)-.5 # vector of changes in benefits
    C = np.random.rand(n)*P # vector of contributions
    
    I = I0.copy() # vector of indicators
    It = np.random.rand(N)*I # vector of lagged indicators
    
    ticks = np.ones(N)*np.nan # simulation period in which each indicator reaches its target
    
    if betas is None:
        betas = np.ones(N)
    betas = betas.copy()
    # betas[~R] = 0


    all_gammas = []
    gammas = np.ones(N)
    
        
    go_on = True
    step = 0
    while go_on:
        
        step += 1 # increase counter (used to indicate period of convergence, so starting value is 2)
        tsI.append(I.copy()) # store this period's indicators
        tsP.append(P.copy()) # store this period's allocations

        deltaIAbs = I-It # change of all indicators
        deltaIIns = deltaIAbs[R].copy() # change of instrumental indicators
        deltaBin = (I>It).astype(int)
        
        # relative change of instrumental indicators
        if np.sum(deltaIIns) == 0:
            deltaIIns = np.zeros(len(deltaIIns))
        else:
            deltaIIns = deltaIIns/np.sum(np.abs(deltaIIns))
        

        ### DETERMINE CONTRIBUTIONS ###
        
        changeF = F - Ft # change in benefits
        changeX = X - Xt # change in actions
        sign = np.sign(changeF*changeX) # sign for the direction of the next action
        changeF[changeF==0] = changeFt[changeF==0] # if the benefit did not change, keep the last change
        sign[sign==0] = signt[sign==0] # if the sign is undefined, keep the last one
        Xt = X.copy() # update lagged actions
        X = X + sign*np.abs(changeF) # determine current action
        assert np.sum(np.isnan(X)) == 0, 'X has invalid values!'
        C = P/(1 + np.exp(-X)) # map action into contribution
        assert np.sum(np.isnan(C)) == 0, 'C has invalid values!'
        signt = sign.copy() # update previous signs
        changeFt = changeF.copy() # update previous changes in benefits
        
        tsC.append(C.copy()) # store this period's contributions
        tsD.append((P-C).copy()) # store this period's inefficiencies
        assert np.sum(P < C)==0, 'C larger than P!'
        tsF.append(F.copy()) # store this period's benefits
        tsX.append(X.copy()) # store this period's actions
        
                
        
        ### DETERMINE BENEFITS ###
        if type(cc) is int or type(cc) is np.int64:
            trial = (np.random.rand(n) < (I[cc]/scalar) * P/P.max() * (P-C)/P) # monitoring outcomes
        else:
            trial = (np.random.rand(n) < cc * P/P.max() * (P-C)/P)
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
        tsS.append(S) # save spillovers
        cnorm = np.zeros(N) # initialize a zero-vector to store the normalized contributions
        cnorm[R] = C # compute contributions only for instrumental nodes
        # print(np.mean(C), np.mean(C/P))
        gammas = ( betas*(cnorm + np.mean(C/P)) )/( 1 + np.exp(-S) ) # compute probability of succesful growth
        assert np.sum(np.isnan(gammas)) == 0, 'gammas has invalid values!'
        assert np.sum(gammas==0) == 0, 'some gammas have zero value!'
        # print(gammas.min())
        
        if force_gammas:
            success = np.ones(N).astype(int)
        else:      
            success = (np.random.rand(N) < gammas).astype(int) # determine if there is succesful growrth
        newI = I + alpha * success # compute new indicators
        assert np.sum(newI < 0) == 0, 'indicators cannot be negative!'
        
        # if theoretical maximums are provided, make sure the indicators do not surpass them
        if max_theo is not None:
            with_bound = ~np.isnan(max_theo)
            newI[with_bound & (newI[with_bound] > max_theo[with_bound])] = max_theo[with_bound & (newI[with_bound] > max_theo[with_bound])]
            assert np.sum(newI[with_bound] > max_theo[with_bound])==0, 'some indicators have surpassed their theoretical upper bound!'
            
            
        # if governance parameters are endogenous, make sure they are not larger than 1
        if (type(cc) is int or type(cc) is np.int64) and newI[cc] > scalar:
            newI[cc] = scalar
        
        if (type(rl) is int or type(rl) is np.int64) and newI[rl] > scalar:
            newI[rl] = scalar
            
        It = I.copy() # update lagged indicators
        I =  newI.copy() # update indicators
        
        if get_gammas:
            all_gammas.append( gammas )
        
        
        P0 += np.random.rand(n)*H/HC
        assert np.sum(np.isnan(P0)) == 0, 'P0 has invalid values!'
        assert np.sum(P0==0) == 0, 'P0 has a zero value!'
        
        i=0
        P = np.zeros(n)
        for program in sorted(program2indis.keys()):
            indis = program2indis[program]
            relevant = inst2idx[indis].astype(int)
            q = P0[relevant]/P0[relevant].sum()
            assert np.sum(np.isnan(q)) == 0, 'q has invalid values!'
            assert np.sum(q == 0 ) == 0, 'q has zero values!'
            if len(B_sequence.shape) == 2:
                P[relevant] += B_sequence[i, step-1]*q/q.sum()
            # else:
            #     P[relevant] += B_sequence[i]*q/q.sum()
            i+=1
        assert int(P.sum()) == int(B_sequence[:, step-1].sum()), 'unequal budgets '+str(int(P.sum())) + ' '+str(int(B_sequence[:, step-1].sum()))
        P[P==0] = 10e-12
                
            
        # assert np.sum(qs_hat>1)==0, 'Propensities larger than 1!'
        assert np.sum(np.isnan(P)) == 0, 'P has invalid values!'
        assert np.sum(P==0) == 0, 'P has zero values!'

        if conv_vals is not None:
            converged = It >= conv_vals
            ticks[(converged) & (np.isnan(ticks))] = step

        if (conv_vals is None and step==max_steps) or (conv_vals is not None and np.sum(np.isnan(ticks))==0):
            go_on = False
            

            
    return np.array(tsI).T, np.array(tsC).T, np.array(tsF).T, np.array(tsP).T, np.array(tsD).T, np.array(tsS).T, ticks, H, all_gammas



    







































