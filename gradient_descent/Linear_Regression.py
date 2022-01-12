# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:03:32 2022

@author: yayaw
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output  # for the following animation


n_samples = 40
training_fraction = 0.8
n_models = 1000
confidence = 90 # percent
max_power = 1  # linear model

X = np.hstack((np.linspace(0, 3, num=n_samples),
               np.linspace(6, 10, num=n_samples))).reshape(2 * n_samples, 1)
T = -1 + 0 * X + 0.1 * X**2 - 0.02 * X**3 + 0.5 * np.random.normal(size=(2 * n_samples, 1))
plt.plot(X, T, '.-')


n_rows = X.shape[0]
row_indices = np.arange(n_rows)
n_train = round(n_rows * training_fraction)
n_test = n_rows - n_train

Xtrain = X[row_indices[:n_train], :]
Ttrain = T[row_indices[:n_train], :]
Xtest = X[row_indices[n_train:], :]
Ttest = T[row_indices[n_train:], :]

plt.plot(Xtrain, Ttrain, 'o', label='Train')
plt.plot(Xtest, Ttest, 'ro', label='Test')
plt.legend(loc='best')


def make_powers(X, max_power):
    return np.hstack([X ** p for p in range(1, max_power + 1)])

def train(X, T, n_epochs, rho):
    
    means = X.mean(0)
    stds = X.std(0)
    # Replace stds of 0 with 1 to avoid dividing by 0.
    stds[stds == 0] = 1
    Xst = (X - means) / stds
    
    Xst = np.insert(Xst, 0, 1, axis=1)  # Insert column of 1's as first column in Xst
    
    # n_samples, n_inputs = Xst.shape[0]
    n_samples, n_inputs = Xst.shape
    
    # Initialize weights to all zeros
    W = np.zeros((n_inputs, 1))  # matrix of one column
    
    # Repeat updates for all samples for multiple passes, or epochs,
    for epoch in range(n_epochs):
        
        # Update weights once for each sample.
        for n in range(n_samples):
        
            # Calculate prediction using current model, w.
            #    n:n+1 is used instead of n to preserve the 2-dimensional matrix structure
            Y = Xst[n:n + 1, :] @ W
            
            # Update w using negative gradient of error for nth sample
            W += rho * Xst[n:n + 1, :].T * (T[n:n + 1, :] - Y)
                
    # Return a dictionary containing the weight matrix and standardization parameters.
    return {'W': W, 'means' : means, 'stds' :stds, 'max_power': max_power}

def use(model, X):
    Xst = (X - model['means']) / model['stds']
    Xst = np.insert(Xst, 0, 1, axis=1)
    Y = Xst @ model['W']
    return Y

def rmse(A, B):
    return np.sqrt(np.mean( (A - B)**2 ))

max_power = 1
Xtrain = X[row_indices[:n_train], :]
Xtest = X[row_indices[n_train:], :]

Xtrain = make_powers(Xtrain, max_power)
Xtest = make_powers(Xtest, max_power)


n_epochs = 1000
rho = 0.01

n_models = 10

models = []
for model_i in range(n_models):
    train_rows = np.random.choice(list(range(n_train)), n_train)
    Xtrain_boot = Xtrain[train_rows, :]
    Ttrain_boot = Ttrain[train_rows, :]
    model = train(Xtrain_boot, Ttrain_boot, n_epochs, rho)
    models.append(model)

use(models[0], Xtest)
Y_all = []
for model in models:
    Y_all.append( use(model, Xtest) )
    
Y_all = np.array(Y_all).squeeze().T  # I like putting each model's output in a column, so `Y_all` now has each model's output for a sample in a row.
Ytest = np.mean(Y_all, axis=1)

RMSE_test = np.sqrt(np.mean((Ytest - Ttest)**2))
print(f'Test RMSE is {RMSE_test:.4f}')

n_plot = 200
Xplot = np.linspace(0, 12.5, n_plot).reshape(n_plot, 1)
Xplot_powers = make_powers(Xplot, max_power)
Ys = []
for model in models:
    Yplot = use(model, Xplot_powers)
    Ys.append(Yplot)

Ys = np.array(Ys).squeeze().T
Ys.shape
    
plt.figure(figsize=(10, 10))
plt.plot(Xtrain[:, 0], Ttrain, 'o')
plt.plot(Xtest[:, 0], Ttest, 'o')
plt.plot(Xplot, Ys, alpha=0.5);
plt.ylim(-14, 2);



max_power = 6
Xtrain = X[row_indices[:n_train], :]
Xtest = X[row_indices[n_train:], :]
Xtrain = make_powers(Xtrain, max_power)
Xtest = make_powers(Xtest, max_power)

n_epochs = 2000
rho = 0.05

n_models = 100

models = []
for model_i in range(n_models):
    train_rows = np.random.choice(list(range(n_train)), n_train)
    Xtrain_boot = Xtrain[train_rows, :]
    Ttrain_boot = Ttrain[train_rows, :]
    model = train(Xtrain_boot, Ttrain_boot, n_epochs, rho)
    models.append(model)
    print(f'Model {model_i}', end=' ')
    
n_plot = 200
Xplot = np.linspace(0, 12.5, n_plot).reshape(n_plot, 1)
Xplot_powers = make_powers(Xplot, max_power)
Ys = []
for model in models:
    Yplot = use(model, Xplot_powers)
    Ys.append(Yplot)

Ys = np.array(Ys).squeeze().T

plt.figure(figsize=(10, 10))
plt.plot(Xtrain[:, 0], Ttrain, 'o')
plt.plot(Xtest[:, 0], Ttest, 'o')
plt.plot(Xplot, Ys, alpha=0.5);
plt.ylim(-14, 2);  


all_Ws = [model['W'] for model in models]

print(all_Ws)
all_Ws = np.array(all_Ws).squeeze()
all_Ws = all_Ws[:, 1:]

all_Ws = np.sort(all_Ws, axis=0)
low_high = all_Ws[[9, 89], :].T
for i, row in enumerate(low_high):
    print(f'Power {i + 1:2} Low {row[0]:6.2f} High {row[1]:6.2f}')
