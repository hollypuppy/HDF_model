import numpy as np
from scipy.stats import wishart


def matern_32_kernel(x, y, eta, rho):
    """Matérn 3/2 kernel function."""
    distance = np.abs(x - y)
    factor = np.sqrt(3) * distance / rho
    return eta ** 2 * (1 + factor) * np.exp(-factor)
def generate_matern_kernel(T, eta=1, rho=1):
    """Generate a T x T kernel matrix using the Matérn 3/2 kernel."""
    # Create a grid of points
    points = np.arange(T)
    # Initialize the kernel matrix
    kernel_matrix = np.zeros((T, T))
    # Compute the kernel matrix
    for i in range(T):
        for j in range(T):
            kernel_matrix[i, j] = matern_32_kernel(points[i], points[j], eta, rho)
    return kernel_matrix

def sample_multivariate_normal(mean, covariance, size=1):
    """Sample from a multivariate normal distribution."""
    return np.random.multivariate_normal(mean, covariance, size)



# generate W
def sample_lkj_correlation_matrix(dim, eta):
    """Sample a correlation matrix from the LKJ distribution."""
    # Use the Wishart distribution as an approximation to LKJ for demo purposes
    # The LKJ distribution isn't directly available in NumPy or SciPy,
    # so we use a workaround by creating a random correlation matrix.
    # A proper implementation may require using PyMC3 or another library.
    wishart_sample = wishart.rvs(df=dim + eta - 1, scale=np.eye(dim))
    correlation_matrix = np.corrcoef(wishart_sample)
    return correlation_matrix
def generate_sigma_matrix(tau, Lambda):
    """Compute the Sigma matrix as diag(tau) * Lambda * diag(tau)."""
    diag_tau = np.diag(tau)
    return diag_tau @ Lambda @ diag_tau

def dgp(N,K,M,T,L):
    # generate theta
    theta = np.zeros((L, T))
    for l in range(L):
        rho_l = (np.random.weibull(a=1, size=1) * 0.1)[0]
        m_l = np.zeros(T)
        # Generate the kernel matrix
        kernel_matrix = generate_matern_kernel(T, eta=1, rho=rho_l)
        theta_l = sample_multivariate_normal(m_l, kernel_matrix)
        theta[l] = theta_l
    # Sample a correlation matrix from the LKJ distribution
    Lambda = sample_lkj_correlation_matrix(K, eta=2)
    # Generate tau vector from a half-normal distribution
    tau = np.abs(np.random.normal(loc=0, scale=5, size=K))
    # Compute the Sigma matrix
    Sigma_w = generate_sigma_matrix(tau, Lambda)
    W = np.zeros((N, K, L))
    for i in range(N):
        for l in range(L):
            w_tilde_il = sample_multivariate_normal(np.zeros(K), Sigma_w)
            w_il = np.log1p(np.exp(w_tilde_il))
            W[i, :, l] = w_il

    # generate alpha
    mu_alpha = np.random.normal(loc=0, scale=5, size=K)
    tau_alpha = np.abs(np.random.normal(loc=0, scale=5, size=K))
    Lambda_alpha = sample_lkj_correlation_matrix(K, eta=2)
    Sigma_alpha = generate_sigma_matrix(tau_alpha, Lambda_alpha)
    alpha = np.zeros((N, K))
    for i in range(N):
        alpha_i = sample_multivariate_normal(mu_alpha, Sigma_alpha)
        alpha[i] = alpha_i

    # obtain beta
    beta = alpha[:, :, np.newaxis] + np.einsum('ikl,lt->ikt', W, theta)

    # obtain observations Y
    X = np.random.normal(0, 1, (K, M, T))
    Y = np.zeros((N, K, M, T))
    # Compute Y matrix using logistic regression
    for i in range(N):
        for c in range(K):
            for j in range(M):
                for t in range(T):
                    # Compute the probability using logistic function
                    log_odds = np.dot(beta[i, c, t], X[c, j, t])
                    probability = 1 / (1 + np.exp(-log_odds))
                    # Draw a Bernoulli sample using the computed probability
                    Y[i, c, j, t] = np.random.binomial(1, probability)

    return(X,Y)
