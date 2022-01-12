import numpy as np
import matplotlib.pyplot as plt
import pandas  # for reading csv file
from IPython.display import display, clear_output  # for animations later in this notebook

data = pandas.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',', usecols=range(15), na_values=-200)
data = data.dropna(axis=0)

hour = [int(t[:2]) for t in data['Time']]
CO = data['CO(GT)']
T = CO
T = np.array(T).reshape((-1, 1))  # make T have one column and as many rows as needed to hold the values of T
Tnames = ['CO']
X = np.array(hour).reshape((-1, 1))
Xnames = ['Hour']
print('X.shape =', X.shape, 'Xnames =', Xnames, 'T.shape =', T.shape, 'Tnames =', Tnames)

plt.plot(X, T, '.')
plt.xlabel(Xnames[0])
plt.ylabel(Tnames[0]);  # semi-colon here prevents printing the cryptic result of call to plt.ylabel()

def linear_model(x, w0, w1):
    return w0 + x * w1

def rmse(X, T, w0, w1):
    Y = linear_model(X, w0, w1)
    return np.sqrt(np.mean((T - Y)**2))
# Need a function for this.  Let's optimize w_bias then w
def coordinate_descent(errorF, X, T, w0, w1, dw, nSteps):
    step = 0
    current_error = errorF(X, T, w0, w1)
    error_sequence = [current_error]
    W_sequence = [[w0, w1]]
    changed = False

    while step < nSteps:

        step += 1
        
        if not changed:
            dw = dw * 0.1
            
        changed = False
        
        # first vary w_bias
        up_error = errorF(X, T, w0 + dw, w1)
        down_error = errorF(X, T, w0 - dw, w1)
        
        if down_error < current_error:
            dw = -dw
            
        while True:
            new_w0 = w0 + dw
            new_error = errorF(X, T, new_w0, w1)
            if new_error >= current_error or step > nSteps:
                break
            changed = True
            w0 = new_w0
            W_sequence.append([w0, w1])
            error_sequence.append(new_error)
            current_error = new_error
            step += 1

        # now vary w
        up_error = errorF(X, T, w0, w1 + dw)
        down_error = errorF(X, T, w0, w1 - dw)
        
        if down_error < current_error:
            dw = -dw
            
        while True:
            new_w1 = w1 + dw
            new_error = errorF(X, T, w0, new_w1)
            if new_error >= current_error or step > nSteps:
                break
            changed = True
            w1 = new_w1
            W_sequence.append([w0, w1])
            error_sequence.append(new_error)
            current_error = new_error
            step += 1

    return w0, w1, error_sequence, W_sequence
#We will need some functions to help us create plots showing the error going down and the sequence of weight values that were tried.

def plot_sequence(error_sequence, W_sequence, label):
    plt.subplot(1, 2, 1)
    plt.plot(error_sequence, 'o-', label=label)
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.subplot(1, 2, 2)
    W_sequence = np.array(W_sequence)
    plt.plot(W_sequence[:, 0], W_sequence[:, 1], '.-', label=label)
    plot_error_surface()

def plot_error_surface():
    n = 20
    w0s = np.linspace(-5, 5, n) 
    w1s = np.linspace(-0.5, 1.0, n) 
    w0s, w1s = np.meshgrid(w0s, w1s)
    surface = []
    for w0i in range(n):
        for w1i in range(n):
            surface.append(rmse(X, T, w0s[w0i, w1i], w1s[w0i, w1i]))
    plt.contourf(w0s, w1s, np.array(surface).reshape((n, n)), cmap='bone')
    # plt.colorbar()
    plt.xlabel('w_bias')
    plt.ylabel('w')
    
def show_animation(model, error_sequence, W_sequence, X, T, label):
    W_sequence = np.array(W_sequence)
    fig = plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    error_line, = plt.plot([], [])
    plt.xlim(0, len(error_sequence))
    plt.ylim(0, max(error_sequence))

    plt.subplot(1, 3, 2)
    plot_error_surface()
 
    w_line, = plt.plot([], [], '.-', label=label)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(X, T, 'o')
    model_line, = plt.plot([], [], 'r-', lw=3, alpha=0.5, label=label)
    plt.xlim(0, 24)
    plt.ylim(np.min(T), np.max(T))

    for i in range(len(W_sequence)):
        
        error_line.set_data(range(i), error_sequence[:i])
        w_line.set_data(W_sequence[:i, 0], W_sequence[:i, 1])
        Y = model(X, W_sequence[i, 0], W_sequence[i, 1])
        model_line.set_data(X, Y)

        plt.pause(0.001)

        clear_output(wait=True)
        display(fig)
w0 = -2
w1 = 0.5
nSteps = 200
dw = 10
w0, w1, error_sequence, W_sequence = coordinate_descent(rmse, X, T, w0, w1, dw, nSteps)
print(f'Coordinate Descent: Error is {rmse(X, T, w0, w1):.2f}   W is {w0:.2f}, {w1:.2f}')
show_animation(linear_model, error_sequence, W_sequence, X, T, 'coord desc')