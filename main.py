import sys
import argparse
import jax
from jax import (random, numpy as jnp)
import numpyro as npr
from numpyro.infer import MCMC, NUTS, Predictive
from model import HDF_model
from data_simulation import dgp





def main(args):
    # initialization
    npr.enable_validation()
    N = args.N
    K = args.K
    M = args.M
    T = args.T
    L = args.L
    # simulate data
    X, Y = dgp(N,K,M,T,L)

    nuts_kernel = NUTS(HDF_model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500)
    mcmc.run(random.PRNGKey(args.seed),  N=N, K=K, M=M, T=T,L=L,X=X, Y=Y.flatten())

    # print
    sys.stdout = open('output.txt', 'w')
    mcmc.print_summary()
    sys.stdout.close()
    sys.stdout = sys.__stdout__


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse args')
    parser.add_argument('-seed', default=2024, type=int)
    parser.add_argument('--N', default=100, type=int)
    parser.add_argument('--K', default=5, type=int)
    parser.add_argument('--M', default=3, type=int)
    parser.add_argument('--T', default=20, type=int)
    parser.add_argument('--L', default=4, type=int)
    args = parser.parse_args()

    main(args)
