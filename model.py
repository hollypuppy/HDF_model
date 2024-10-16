import numpyro as npr
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random

from jax import (vmap, nn, numpy as jnp)

def matern_32_kernel(x, y, eta, rho):
    """Matérn 3/2 kernel function."""
    distance = jnp.abs(x - y)
    factor = jnp.sqrt(3) * distance / rho
    return eta**2 * (1 + factor) * jnp.exp(-factor)

def generate_matern_kernel(T, eta, rho):
    points = jnp.arange(T)
    kernel_matrix = jnp.array(
        [[matern_32_kernel(points[i], points[j], eta, rho) for j in range(T)] for i in range(T)]
    )
    return kernel_matrix


def HDF_model(N, K, M, T, L, X, Y):
    with npr.plate('L', L):
        rho = npr.sample("rho", npr.distributions.Weibull(0.1, 1))
        kernel_matrices = jnp.stack([generate_matern_kernel(T, 1, rho_i) for rho_i in rho], axis=0)  # Shape (L, T, T)
        theta = npr.sample('theta', dist.MultivariateNormal(loc=jnp.zeros(T), covariance_matrix=kernel_matrices))

    Lambda = npr.sample('Lambda', npr.distributions.LKJ(K, 2))
    tau = npr.sample('tau', dist.HalfNormal(5).expand([K]))
    D = jnp.diag(tau)
    Sigma_w = npr.deterministic('Sigma_w', jnp.matmul(D, jnp.matmul(Lambda, D)))

    with npr.plate('N*L', N * L):  # w_il
        # w_tilde ~ N(0,Sigma_w)
        W_tilde = npr.sample('W_tilde', dist.MultivariateNormal(loc=jnp.zeros(K), covariance_matrix=Sigma_w))

    W_tilde = W_tilde.reshape(N, L, K).transpose(0, 2, 1)
    W = jnp.log(1 + jnp.exp(W_tilde))
    mu_alpha = npr.sample('mu_alpha', dist.Normal(0, 5).expand((K,)))
    Lambda_alpha = npr.sample('Lambda_alpha', npr.distributions.LKJ(K, 2))
    tau_alpha = npr.sample('tau_alpha', dist.HalfNormal(5).expand([K]))
    D = jnp.diag(tau_alpha)
    Sigma_alpha = npr.deterministic('Sigma_alpha', jnp.matmul(D, jnp.matmul(Lambda_alpha, D)))

    with npr.plate('N', N):  # alpha_i
        # alpha_i ~ N(mu_alpha,Sigma_alpha)
        alpha = npr.sample('alpha', dist.MultivariateNormal(loc=mu_alpha, covariance_matrix=Sigma_alpha))

    beta = alpha[..., jnp.newaxis] + jnp.tensordot(W, theta, axes=([2], [0]))
    logits = jnp.einsum('nkt,kmt->nkmt', beta, X)  # Resulting shape will be (N, K, M, T)

    # Likelihood for Y
    with npr.plate('customers', N * K * M * T):
        npr.sample('Y', dist.Bernoulli(logits=logits.flatten()), obs=Y)

