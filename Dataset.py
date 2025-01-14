import torch
from torch.utils.data import DataLoader

class DatasetHandler:
    def __init__(self, sobol_gen, domain_boundaries):
        self.sobol_gen = sobol_gen
        self.domain_boundaries = domain_boundaries

    def convert_to_domain(self, tensor):
        """
        Convert into domain boundaries by scaling and transform

        Params:
        tensor (tensor): input tensor, values in range [0,1]

        Returns:
        in_domain_tensor (tensor): tensor after convert to domain
        """
        domain_range = self.domain_boundaries[:, 1] - self.domain_boundaries[:, 0]
        in_domain_tensor = tensor * domain_range + self.domain_boundaries[:, 0]
        return in_domain_tensor

    def create_datasets(self, n_interior, n_boundary, n_initial):
        """
        Create datasets for training using sobol sequence generator for better fitting the space more evenly.

        Params:
        n_points (int): input tensor - maybe out of domain
        n_boundary (int): number of boundary point
        n_temporal (int): number of temporal point

        Returns:
        in_domain_tensor (tensor): tensor after convert to domain
        """
        boundary_points = self.generate_boundary_condition_points(n_boundary)
        initial_points = self.generate_initial_condition_points(n_initial)
        interior_points = self.generate_interior_points(n_interior)

        boundary_loader = DataLoader(torch.utils.data.TensorDataset(boundary_points), batch_size=2*n_boundary, shuffle=True)
        intial_loader = DataLoader(torch.utils.data.TensorDataset(initial_points), batch_size=n_initial, shuffle=True)
        interior_loader = DataLoader(torch.utils.data.TensorDataset(interior_points), batch_size=n_interior, shuffle=True)

        #print(f"Boundary Loader: {boundary_loader}")
        #print(f"Temporal Loader: {temporal_loader}")
        #print(f"Interior Loader: {interior_loader}")

        return interior_loader, boundary_loader, intial_loader
    
    def generate_initial_condition_points(self, n_initial_points):
        """
        Generate temporal points

        Params:
        n_boundary (int): number of temporal boundary points

        Returns:
        intial_points (tensor): 2D array of [x, 0]
        """
        # generate set of sobol points [0,1]
        sobol_intial_points = self.sobol_gen.draw(n_initial_points)
        intial_points = self.convert_to_domain(sobol_intial_points)
        t0 = self.domain_boundaries[0, 0]

        # change into temporal boundary point (t = t0)
        intial_points[:, 1] = t0

        assert ((intial_points[3][1]).item()) == t0

        return intial_points #, t0_boundary_outputs

    def generate_boundary_condition_points(self, n_boundary_points): # x = x0 (3)
        # generate set of sobol points [0,1]
        sobol_boundary_points = self.sobol_gen.draw(n_boundary_points)
        boundary_points = self.convert_to_domain(sobol_boundary_points)
        x0 = self.domain_boundaries[1, 0] # boundaries x0
        xL = self.domain_boundaries[1, 1]
        # change into spatial boundary point 
        lower_boundary_points = boundary_points.clone()
        lower_boundary_points[:, 0] = x0
        upper_boundary_points = boundary_points.clone()
        upper_boundary_points[:, 0] = xL 

        boundary_points = torch.cat([lower_boundary_points, upper_boundary_points], 0)

        return boundary_points

    def generate_interior_points(self, n_points):
        sobol_interior_points = self.sobol_gen.draw(n_points)
        interior_points = self.convert_to_domain(sobol_interior_points)

        return interior_points
