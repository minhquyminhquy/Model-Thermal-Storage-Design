import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, IN_DIM, OUT_DIM, N_HIDDEN_LAYERS, 
        N_NEURONS, REGULARIZE_PARAM, REGULARIZE_EXP):
        super().__init__()
        torch.manual_seed(42)
        
        self.input_dimension_ = IN_DIM
        self.output_dimension_ = OUT_DIM
        self.num_hidden_layer_ = N_HIDDEN_LAYERS
        self.num_neurons_ = N_NEURONS
        activation = nn.Tanh # for PINN, smooth activation func works

        # input layer
        self.fcs = nn.Sequential(*[
                        nn.Linear(self.input_dimension_, self.num_neurons_),
                        activation()])
        # hidden layer
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(self.num_neurons_, self.num_neurons_),
                            activation()]) for _ in range(self.num_hidden_layer_-1)])
        # output layer
        self.fce = nn.Linear(self.num_neurons_, self.output_dimension_)

        # Initialize weights and biases
        self._initialize_weights()

        # regularization term - L2
        self.regularization_param = REGULARIZE_PARAM
        self.regularization_exp = REGULARIZE_EXP

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
    def _initialize_weights(self):
        """
        Initialize the weights of the network using Xavier initialization for weights.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def compute_loss(self, output, target):
        # MSE loss
        mse_loss = F.mse_loss(output, target)

        # regularization term - L2
        if self.regularization_param > 0:
            l2_loss = 0
            for param in self.parameters():
                l2_loss += torch.sum(param ** self.regularization_exp)
            mse_loss += self.regularization_param * l2_loss

        return mse_loss
        




