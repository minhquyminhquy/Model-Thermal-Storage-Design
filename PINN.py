import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Model import FCN
from Dataset import DatasetHandler
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(123)

class PINN():
    "Defines a PINNs. Input (t, x) => Output (T_f, T_s)"
    def __init__(self, N_INTERIOR, N_SPATIAL, N_TEMPORAL):
      super().__init__()
      self.n_int_ = N_INTERIOR # number of interior point
      self.n_sb_ = N_SPATIAL # number of spatial boundary point
      self.n_tb_ = N_TEMPORAL # number of temporal point

      self.alpha_fluid = 0.05
      self.alpha_solid = 0.08
      self.heat_transfer_fluid = 5
      self.heat_transfer_solid = 6
      self.temperature_hot = 4
      self.temperature_initial = 1
      self.thermal_conductivity = 1

      # Set extrema for solution domain of x and t
      self.domain_boundaries = torch.tensor([[0,1], [0,1]])

      # Set number of space dimensions
      self.space_dim = 1

      self.model = FCN(IN_DIM=self.domain_boundaries.shape[0], OUT_DIM=2,
                                              N_HIDDEN_LAYERS=4, N_NEURONS=20, REGULARIZE_PARAM=0.,
                                              REGULARIZE_EXP=2.) # step 1
        
      # Sobol sequence generator for input sampling
      self.sobol_gen = torch.quasirandom.SobolEngine(dimension=self.domain_boundaries.shape[0])
      self.dataset_handler = DatasetHandler(self.sobol_gen, self.domain_boundaries)
      self.training_sets = self.dataset_handler.create_datasets(self.n_int_, self.n_sb_, self.n_tb_)

    def apply_initial_conditions(self, input_data):
        return self.model(input_data)

    def apply_boundary_conditions(self, boundary_inputs):
        boundary_inputs.requires_grad = True
        lower_boundary, upper_boundary = boundary_inputs[:boundary_inputs.shape[0] // 2], boundary_inputs[boundary_inputs.shape[0] // 2:]

        fluid_temp_lower = self.model(lower_boundary)[:, 0]
        solid_temp_lower = self.model(lower_boundary)[:, 1]
        fluid_temp_upper = self.model(upper_boundary)[:, 0]
        solid_temp_upper = self.model(upper_boundary)[:, 1]

        grad_solid_lower = torch.autograd.grad(solid_temp_lower.sum(), lower_boundary, create_graph=True)[0][:, 1]
        grad_fluid_upper = torch.autograd.grad(fluid_temp_upper.sum(), upper_boundary, create_graph=True)[0][:, 1]

        results = torch.cat([fluid_temp_lower, grad_solid_lower, fluid_temp_upper, grad_fluid_upper], dim=0)
        return results

    def compute_pde_residual(self, input_data):
        input_data.requires_grad = True
        predictions = self.model(input_data)

        fluid_temp = predictions[:, 0]
        solid_temp = predictions[:, 1]

        grad_fluid = torch.autograd.grad(fluid_temp.sum(), input_data, create_graph=True)[0]
        grad_solid = torch.autograd.grad(solid_temp.sum(), input_data, create_graph=True)[0]

        grad_fluid_time = grad_fluid[:, 0]
        grad_fluid_space = grad_fluid[:, 1]
        grad_solid_time = grad_solid[:, 0]
        grad_solid_space = grad_solid[:, 1]

        grad_fluid_space_2 = torch.autograd.grad(grad_fluid_space.sum(), input_data, create_graph=True)[0][:, 1]
        grad_solid_space_2 = torch.autograd.grad(grad_solid_space.sum(), input_data, create_graph=True)[0][:, 1]

        residual_fluid = (grad_fluid_time) + (self.thermal_conductivity * grad_fluid_space) - (self.alpha_fluid * grad_fluid_space_2) + (self.heat_transfer_fluid * (fluid_temp - solid_temp))
        residual_solid = (grad_solid_time) - (self.alpha_solid * grad_solid_space_2) - (self.heat_transfer_solid * (fluid_temp - solid_temp))

        return residual_fluid.reshape(-1,), residual_solid.reshape(-1,)

    def compute_loss(self, boundary_data, temporal_data, interior_data):
        boundary_preds = self.apply_boundary_conditions(boundary_data[0])
        temporal_preds = self.apply_initial_conditions(temporal_data[0])

        pde_residuals = self.compute_pde_residual(interior_data[0])

        boundary_loss = torch.mean(torch.abs(boundary_preds) ** 2)
        temporal_loss = torch.mean(torch.abs(temporal_data[1] - temporal_preds) ** 2)
        pde_loss = torch.mean(torch.abs(pde_residuals[0]) ** 2) + torch.mean(torch.abs(pde_residuals[1]) ** 2)

        total_loss = boundary_loss + temporal_loss + pde_loss
        return total_loss

    def fit(self, epochs, optimizer):
        history = []

        for _ in range(epochs):
            for batch in zip(*self.training_sets):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(*batch)
                    loss.backward()
                    history.append(loss.item())
                    return loss

                optimizer.step(closure)

        return history

    def visualize_results(self):
        test_points = self.sobol_gen.draw(10000)
        output = self.model(test_points)

        fluid_output = output[:, 0].reshape(-1,)
        solid_output = output[:, 1].reshape(-1,)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].scatter(test_points[:, 0].detach(), test_points[:, 1].detach(), c=fluid_output.detach(), cmap="coolwarm")
        axes[0].set_title("Fluid Temperature")
        axes[1].scatter(test_points[:, 0].detach(), test_points[:, 1].detach(), c=solid_output.detach(), cmap="coolwarm")
        axes[1].set_title("Solid Temperature")
        plt.show()