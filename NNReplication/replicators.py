# %%
import torch.nn as nn
import torch

class RegenerationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_params = None

    def predict_own_weights(self, *args, **kwargs):
        raise NotImplementedError

    def set_weights(self, *args, **kwargs):
        raise NotImplementedError

    def predict_param_by_idx(self, idx):
        raise NotImplementedError

class SimpleModel(RegenerationBase):
    def __init__(self, layers_size=100):
        super().__init__()
        self.activation = nn.SELU()
        self.layer1 = nn.Linear(layers_size, layers_size, bias=False)
        self.layer2 = nn.Linear(layers_size, 1, bias=False)

        self.target_params = []
        for pmatrix in self.parameters():
            for p in pmatrix.view(-1):
                self.target_params.append(p.data)

        self.projection = nn.Linear(
            layers_size, len(self.target_params)
        ).weight.data.requires_grad_(False)

    def forward(self, x):
        x = x @ self.projection
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x
    
    @torch.no_grad()
    def predict_own_weights(self, shift_by=0):
        num_params = len(self.target_params)
        one_hot_coordinates = torch.eye(num_params)
        weights = self.forward(one_hot_coordinates) + shift_by
        return weights.squeeze().tolist()

    @torch.no_grad()
    def set_weights(self, weights):
        if len(weights) != len(self.target_params):
            raise ValueError(
                "length of weights must match length of self.target_params"
            )

        for i in range(len(weights)):
            self.target_params[i].fill_(
                weights[i]
            )  # this also fills orignal params. (self.target_params holds references)

    def predict_param_by_idx(self, idx):
        num_params = len(self.target_params)
        one_hot_coordinate = torch.eye(num_params)[idx]
        return self.forward(one_hot_coordinate)
# %%

if __name__ == "__main__":
    model = SimpleModel(150)

    from time import time
    s = time()
    new_weights = model.predict_own_weigths(shift_by=0.1)
    print(time() - s)
    print(new_weights[:3])

    s = time()
    new_weights_2 = model.predict_own_weights2(shift_by=0.1)
    print(time() - s)
    print(new_weights_2[:3])
# %%