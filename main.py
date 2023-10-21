# %%
# %load_ext autoreload
# %autoreload 2
# %%
import random
from typing import List

import torch
from torch import Tensor
from NNReplication import replicators
import matplotlib.pyplot as plt
import numpy as np

# %%
# %%
from tqdm import tqdm
def regeneration_process(model: replicators.RegenerationBase, regen_steps, shift_by=0):
    norms = []
    for i in tqdm(range(regen_steps)):
        norm = torch.norm(torch.tensor(model.target_params), 1)
        norms.append(norm.item())
        new_weights = model.predict_own_weights(shift_by=shift_by)
        model.set_weights(new_weights)
    return norms


model_1 = replicators.SimpleModel(layers_size=20)
norms_1 = regeneration_process(model_1, regen_steps=20)

model_2 = replicators.SimpleModel(layers_size=40)
norms_2 = regeneration_process(model_2, regen_steps=20)
plt.figure(figsize=(10, 6))
plt.plot(norms_1, label='Model 1', marker='o')
plt.plot(norms_2, label='Model 2', marker='o')
plt.title('Norms Trend for Model 1 and Model 2')
plt.xlabel('Generation')
plt.ylabel('L1 Norm')
plt.legend()
plt.grid(True)
plt.show()


# %%
def plot_family_of_networks(num_networks, regen_steps,shift_by):
    plt.figure(figsize=(10, 6))

    model_norms = []
    for i in range(num_networks):
        layer_size = random.randint(10,40)
        model = replicators.SimpleModel(layers_size=layer_size)
        norms = regeneration_process(model, regen_steps,shift_by)

        if norms[-1] <= 100: # only models that don't blow up
            model_norms.append((model,norms))
        
        plt.plot(norms, alpha=0.5)
        
    plt.title(f'Norms Trend for {num_networks} Networks')
    plt.xlabel('Generation')
    plt.ylabel('L1 Norm')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0,100)
    plt.show()
    return model_norms

models_norms = plot_family_of_networks(num_networks=20, regen_steps=20,shift_by=0.03)

# %%
import scipy.stats
def plot_corr(model: replicators.RegenerationBase):
    params = model.target_params
    num_params = len(params)
    indices = torch.arange(num_params)
    predictions = model.predict_param_by_idx(indices)
    predictions = predictions.squeeze().tolist()
    weights = [param.item() for param in params]
    
    correlation_coefficient, p_value = scipy.stats.pearsonr(predictions, weights)
    print(f"Pearson correlation coefficient: {correlation_coefficient:.3f}")
    print(f"P-value: {p_value}")

    plt.title(f'Correlation between weights and predictions (coef:{correlation_coefficient:.3f})')
    plt.scatter(weights, predictions)
    plt.xlabel("Weight")
    plt.ylabel("Prediction")
    plt.grid(True)
    plt.show()



# %%
first_model = models_norms[0][0]
plot_corr(first_model)
# %%
# %%
model_2 = replicators.SimpleModel(layers_size=100)
norms = regeneration_process(model_2, regen_steps=10,shift_by=0.025)
print(f"last norm : {norms[-1]}")
plot_corr(model_2) # might give an error if model norms blew up. 
# %%
