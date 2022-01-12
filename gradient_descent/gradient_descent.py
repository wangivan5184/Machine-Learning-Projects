import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output  # for the following animation

n_samples = 100
X = np.random.uniform(0, 10,(n_samples,1))
T = 2 - 0.1 * X + 0.05 * (X - 6)**2 + np.random.normal(0, 0.1, (n_samples,1))
plt.plot(X, T, 'o')
X1 = np.insert(X, 0, 1, axis=1)

learning_rate = 0.01
n_samples = X1.shape[0]  # number of rows in data equals the number of samples

W = np.zeros((2, 1))                # initialize the weights to zeros
for epoch in range(10):             # train for this many epochs, or passes through the data set
    for n in range(n_samples):
        Y = X1[n:n + 1, :] @ W      # predicted value, y, for sample n
        error = (T[n:n + 1, :] - Y)  # negative gradient of squared error
        
        # update weights by fraction of negative derivative of square error with respect to weights
        W -=  learning_rate * -2 * X1[n:n + 1, :].T * error  
        
print(W)
plt.plot(X, T, 'o', label='Data')
plt.plot(X, X1 @ W, 'ro', label='Predicted')

print(X1@W)
plt.legend();

def run(rho, n_epochs, stepsPerFrame=10):

    # Initialize weights to all zeros
    # For this demonstration, we will have one variable input. With the constant 1 input, we have 2 weights.
    W = np.zeros((2,1))

    # Collect the weights after each update in a list for later plotting. 
    # This is not part of the training algorithm
    ws = [W.copy()]

    # Create a bunch of x values for plotting
    xs = np.linspace(0, 10, 100).reshape((-1, 1))
    xs1 = np.insert(xs, 0, 1, axis=1)

    fig = plt.figure(figsize=(8, 8))

    # For each pass (one epoch) through all samples ...
    for iter in range(n_epochs):
        # For each sample ...
        for n in range(n_samples):
        
            # Calculate prediction using current model, w.
            #    n:n+1 is used instead of n to preserve the 2-dimensional matrix structure
            Y = X1[n:n + 1,:] @ W
            
            # Update w using negative gradient of error for nth sample
            W += rho * X1[n:n + 1, :].T * (T[n:n + 1, :] - Y)
            
            # Add new w to our list of past w values for plotting
            ws.append(W.copy())
        
            if n % stepsPerFrame == 0:
                fig.clf()

                # Plot the X and T data.
                plt.subplot(2, 1, 1)
                plt.plot(X, T, 'o', alpha=0.6, label='Data')
                plt.plot(X[n,0], T[n], 'ko', ms=10, label='Last Trained Sample')

                # Plot the output of our linear model for a range of x values
                plt.plot(xs, xs1 @ W, 'r-', linewidth=5, label='Model')
                plt.xlabel('$x$')
                plt.legend(loc='upper right')
                plt.xlim(0, 10)
                plt.ylim(0, 5)

                # In second panel plot the weights versus the epoch number
                plt.subplot(2, 1, 2)
                plt.plot(np.array(ws)[:, :, 0])
                plt.xlabel('Updates')
                plt.xlim(0, n_epochs * n_samples)
                plt.ylim(-1, 3)
                plt.legend(('$w_0$', '$w_1$'))
        
                clear_output(wait=True)
                display(fig)
    
    clear_output(wait=True)
    
    return W
run(0.01, n_epochs=1, stepsPerFrame=1)
run(0.01, n_epochs=20, stepsPerFrame=10)