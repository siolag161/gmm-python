import numpy as np
import random as rd
import math

import hungarian as hg

def random_parameters(data, K):
    """ init the means, covariances and mixing coefs"""
    cols = (data.shape)[1]
 
    mu = np.zeros((K, cols))
    for k in range(K):
        idx = np.floor(rd.random()*len(data))
        for col in range(cols):
            mu[k][col] += data[idx][col]
 
    sigma = []
    for k in range(K):
        sigma.append(np.cov(data.T))
 
    pi = np.ones(K)*1.0/K
 
    return mu, sigma, pi

def _e_step(data, K, mu, sigma, pi):
    """ evaluate the responsabilities using the current parameters """
    idvs = len(data)
    
    resp = np.zeros((idvs, K))
    for k in range(K):
        for i in range(idvs):
            resp[i][k] = pi[k]*gaussian(data[i], mu[k], sigma[k])

    return resp

def e_step(data, K, mu, sigma, pi):
    idvs = (data.shape)[0]
    cols = (data.shape)[1]
 
    resp = np.zeros((idvs, K))
 
    for i in range(idvs):
        for k in range(K):
            resp[i][k] = pi[k]*gaussian(data[i], mu[k], sigma[k])/likelihood(data[i], K, mu, sigma, pi)

    return resp

def log_likelihood(data, K, mu, sigma, pi):
    """ marginal over X """
    log_likelihood = 0.0
    for n in range (len(data)):
        log_likelihood += np.log(likelihood(data[n], K, mu, sigma, pi))
    return log_likelihood 

 
def likelihood(x, K, mu, sigma, pi):
    rs = 0.0
    for k in range(K):
        rs += pi[k]*gaussian(x, mu[k], sigma[k])
    return rs


def m_step(data, K, resp):
    """ find the parameters that maximize the log-likelihood given the current resp."""
    idvs = (data.shape)[0]
    cols = (data.shape)[1]
    
    mu = np.zeros((K, cols))
    sigma = np.zeros((K, cols, cols))
    pi = np.zeros(K)

    marg_resp = np.zeros(K)
    for k in range(K):
        for i in range(idvs):
            marg_resp[k] += resp[i][k]
            mu[k] += (resp[i][k])*data[i]
        mu[k] /= marg_resp[k]

        for i in range(idvs):
            #x_i = (np.zeros((1,cols))+data[k])
            x_mu = np.zeros((1,cols))+data[i]-mu[k]
            sigma[k] += (resp[i][k]/marg_resp[k])*x_mu*x_mu.T

        pi[k] = marg_resp[k]/idvs        
        
    return mu, sigma, pi


def gaussian(x, mu, sigma):
    """ compute the pdf of the multi-var gaussian """
    idvs = len(x)
    norm_factor = (2*np.pi)**idvs

    norm_factor *= np.linalg.det(sigma)
    norm_factor = 1.0/np.sqrt(norm_factor)

    x_mu = np.matrix(x-mu)

    rs = norm_factor*np.exp(-0.5*x_mu*np.linalg.inv(sigma)*x_mu.T)
    return rs

def _log_likelihood(data, K, mu, sigma, pi):
    """ evalutate the (marginal) log-likelihood """
    score = 0.0
    for n in range (len(data)):
        for k in range(K):
            score += np.log(pi[k]*gaussian(data[n], mu[k], sigma[k]))
    return score


def EM(data, rst, K, threshold):
    converged = False
    mu, sigma, pi = random_parameters(data, K)
    
    current_log_likelihood = log_likelihood(data, K, mu, sigma, pi)
    max_iter = 100
    for it in range(max_iter):
        print rst, "       |       ", it, "     |     ", current_log_likelihood[0][0]
        resp = e_step(data, K, mu, sigma, pi)
        mu, sigma, pi = m_step(data, K, resp)

        new_log_likelihood = log_likelihood(data, K, mu, sigma, pi)
        if (abs(new_log_likelihood-current_log_likelihood) < threshold):
            converged = True
            break

        current_log_likelihood = new_log_likelihood
            
    return current_log_likelihood, mu, sigma, resp

#######################################################################
def assign_clusters(K, resp):
    idvs = len(resp)
    clusters = np.zeros(idvs, dtype=int)

    for i in range(idvs):
        #clusters[i][k] = 0
        clss = 0
        for k in range(K):
            if resp[i][k] > resp[i][clss]:
                clss = k
        clusters[i] = clss

    return clusters

def compute_statistics(clusters, ref_clusters, K):
    mat = make_ce_matrix(clusters, ref_clusters, K)
    hung_solver = hg.Hungarian()
    rs = hung_solver.compute(mat, False)

    tmp_clusters = np.array(clusters)
    for old, new in rs:
        clusters[np.where(tmp_clusters == old)] = new
        #print old, new

    #print clusters, ref_clusters
    nbrIts = 0
    for k in range(K):
        ref = np.where(ref_clusters == k)[0]
        clust = np.where(clusters == k)[0]
        nbrIts += len(np.intersect1d(ref, clust))
        print len(np.intersect1d(ref, clust))
    return nbrIts

def make_ce_matrix(clusters, ref_clusters, K):    
    mat = np.zeros((K, K), dtype=int)    
    for i in xrange(K):
        for j in xrange(K):
            ref_i = np.where(ref_clusters == i)[0]
            clust_j = np.where(clusters == j)[0]
            its = np.intersect1d(ref_i, clust_j)
            mat[i,j] = len(ref_i) + len(clust_j) -2*len(its)

    return mat

            
########################################################################
def read_data(file_name):
    """ read the data from filename as numpy array """    
    with open(file_name) as f:
        data =  np.loadtxt(f, delimiter=",", dtype = "float", 
                          skiprows=0, usecols=(0,1,2,3))
        
    with open(file_name) as f:
        ref_classes =  np.loadtxt(f, delimiter=",", dtype = "string", 
                                   skiprows=0, usecols=[4])
        unique_ref_classes = np.unique(ref_classes)
        ref_clusters = np.argmax(ref_classes[np.newaxis,:]==unique_ref_classes[:,np.newaxis],axis=0)
       
            
    return data, ref_clusters

def main():
    print "begining..."
    file_name = "iris.data"
    nbr_restarts = 5
    threshold = 0.001
    K = 3
    
    data, ref_clusters = read_data(file_name)
    mu_lst = []
    sigma_lst = []

    print "#restart | EM iteration | log likelihood"
    print "----------------------------------------"

    max_likelihood_score = float("-inf")
    for rst in range(nbr_restarts):
        log_likelihood, mu, sigma, resp = EM(data, rst, K, threshold)
        if log_likelihood > max_likelihood_score:
            max_likelihood_score = log_likelihood
            max_mu, max_sigma, max_resp = mu, sigma, resp

    clusters = assign_clusters(K, max_resp)
    cost = compute_statistics(clusters, ref_clusters, K)
    #print clusters
    #print ref_clusters
    print cost*1.0/len(data)

if __name__ == '__main__':
    main()
