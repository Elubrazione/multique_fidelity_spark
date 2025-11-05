import math
import torch
import gpytorch
import numpy as np
from openbox.utils.config_space.util import convert_configurations_to_array

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class MultiTaskGP:
    """
    Gaussian Processes with identifying the important configuration parameters (IICP) by CPS and CPE.
    CPS: configuration parameter selection using Spearman Correlation Coefficient.
    CPE: configuration parameter extraction using Kernel PCA.
    """
    def __init__(self, history):
        self.history = history
        self.history_X = torch.from_numpy(np.array(convert_configurations_to_array(history.configs)))
        self.history_Y = torch.from_numpy(np.array([x[0] for x in history.objectives]))

    def train(self, X: np.ndarray, Y: np.ndarray):
        
        train_x1 = torch.from_numpy(X)
        train_x2 = self.history_X

        train_y1 = torch.from_numpy(Y)
        train_y2 = self.history_Y
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        train_i_task1 = torch.full((train_x1.shape[0],1), dtype=torch.long, fill_value=0)
        train_i_task2 = torch.full((train_x2.shape[0],1), dtype=torch.long, fill_value=1)

        full_train_x = torch.cat([train_x1, train_x2])
        full_train_i = torch.cat([train_i_task1, train_i_task2])
        full_train_y = torch.cat([train_y1, train_y2])

        # Here we have two iterms that we're passing in as train_inputs
        self.model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, self.likelihood)
        training_iterations = 50

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iterations):
            self.optimizer.zero_grad()
            output = self.model(full_train_x, full_train_i)
            loss = -self.mll(output, full_train_y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X: np.ndarray):

        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()

        # Test points every 0.02 in [0,1]
        test_x = torch.from_numpy(X)
        test_i = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=0)

        # Make predictions - one task at a time
        # We control the task we cae about using the indices

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_y = self.likelihood(self.model(test_x, test_i))

        low, up = observed_pred_y.confidence_region()
        mean = (up + low) / 2
        std = up - mean
        var = std * std

        mean = mean.numpy()
        var = var.numpy()

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var

    def predict_marginalized_over_instances(self, X: np.ndarray):
        return self.predict(X)