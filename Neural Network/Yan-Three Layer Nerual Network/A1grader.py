run_my_solution = False
import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A1mysolution import *
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')

    assignmentNumber = '1'

    import subprocess, glob, pathlib
    nb_name = '*A{}*.ipynb'
    # nb_name = '*.ipynb'
    filename = next(glob.iglob(nb_name.format(assignmentNumber)), None)
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         nb_name.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ClassDef)):
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *


    
exec_grade = 0

for func in ['forward', 'gradient', 'train', 'use']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

            
print('''\nTesting
    X = np.arange(4).reshape(-1, 1)
    T = np.log(X + 10)

    ni = 1
    nu = 2
    nv = 3
    U = np.arange((ni + 1) * nu).reshape(ni + 1, nu) * 0.1
    V = (np.arange((nu + 1) * nv).reshape(nu + 1, nv) - 6) * 0.1
    W = np.arange(nv + 1).reshape(nv + 1, 1) * -0.1

    Zu, Zv, Y = forward(addOnes(X), U, V, W)
''')


try:
    pts = 20

    X = np.arange(4).reshape(-1, 1)
    T = np.log(X + 10)

    ni = 1
    nu = 2
    nv = 3
    U = np.arange((ni + 1) * nu).reshape(ni + 1, nu) * 0.1
    V = (np.arange((nu + 1) * nv).reshape(nu + 1, nv) - 6) * 0.1
    W = np.arange(nv + 1).reshape(nv + 1, 1) * -0.1

    Zu, Zv, Y = forward(addOnes(X), U, V, W)
    
    Zu_answer = np.array([[0.        , 0.09966799],
                          [0.19737532, 0.37994896],
                          [0.37994896, 0.60436778],
                          [0.53704957, 0.76159416]])
    Zv_answer = np.array([[-0.53704957, -0.45424278, -0.36276513],
                          [-0.57783916, -0.46328044, -0.3308191 ],
                          [-0.61316945, -0.47426053, -0.30690171],
                          [-0.64173317, -0.4863364 , -0.29258059]])
    Y_answer = np.array([[0.25338305],
                         [0.24968573],
                         [0.24823956],
                         [0.24921478]])

    close = [np.allclose(Zu, Zu_answer, 0.1),
             np.allclose(Zv, Zv_answer, 0.1),
             np.allclose(Y, Y_answer, 0.1)]
    if all(close):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Zu, Zv and Y are correct.')
    else:
        if not close[0]:
            print(f'\n---  0/{pts} points. Returned incorrect value for Zu.')
        if not close[1]:
            print(f'\n---  0/{pts} points. Returned incorrect value for Zv.')
        if not close[2]:
            print(f'\n---  0/{pts} points. Returned incorrect value for Y.')
        print(f'Correct values are\n Zu should be {Zu_answer}\n Zv should be {Zv_answer}\n Y should be {Y_answer}')
        print(f'Your values are \n Zu = {Zu}\n Zv = {Zv}\n Y = {Y}')

except Exception as ex:
    print(f"\n--- 0/{pts} points. Function 'forward' raised the exception\n")
    print(ex)




print('''\nTesting
    X = np.arange(4).reshape(-1, 1)
    T = np.log(X + 10)

    ni = 1
    nu = 2
    nv = 3
    U = np.arange((ni + 1) * nu).reshape(ni + 1, nu) * 0.1
    V = (np.arange((nu + 1) * nv).reshape(nu + 1, nv) - 6) * 0.1
    W = np.arange(nv + 1).reshape(nv + 1, 1) * -0.1

    Zu = np.array([[0.        , 0.09966799],
                   [0.19737532, 0.37994896],
                   [0.37994896, 0.60436778],
                   [0.53704957, 0.76159416]])
    Zv = np.array([[-0.53704957, -0.45424278, -0.36276513],
                   [-0.57783916, -0.46328044, -0.3308191 ],
                   [-0.61316945, -0.47426053, -0.30690171],
                   [-0.64173317, -0.4863364 , -0.29258059]])
    Y = np.array([[0.25338305],
                  [0.24968573],
                  [0.24823956],
                  [0.24921478]])

    grad_wrt_U, grad_wrt_V, grad_wrt_W = gradient(addOnes(X), T, Zu, Zv, Y, U, V, W)
''')


try:
    pts = 20

    X = np.arange(4).reshape(-1, 1)
    T = np.log(X + 10)

    ni = 1
    nu = 2
    nv = 3
    U = np.arange((ni + 1) * nu).reshape(ni + 1, nu) * 0.1
    V = (np.arange((nu + 1) * nv).reshape(nu + 1, nv) - 6) * 0.1
    W = np.arange(nv + 1).reshape(nv + 1, 1) * -0.1

    Zu = np.array([[0.        , 0.09966799],
                   [0.19737532, 0.37994896],
                   [0.37994896, 0.60436778],
                   [0.53704957, 0.76159416]])
    Zv = np.array([[-0.53704957, -0.45424278, -0.36276513],
                   [-0.57783916, -0.46328044, -0.3308191 ],
                   [-0.61316945, -0.47426053, -0.30690171],
                   [-0.64173317, -0.4863364 , -0.29258059]])
    Y = np.array([[0.25338305],
                  [0.24968573],
                  [0.24823956],
                  [0.24921478]])

    grad_wrt_U, grad_wrt_V, grad_wrt_W = gradient(addOnes(X), T, Zu, Zv, Y, U, V, W)

    grad_wrt_U_answer = np.array([[-0.5952239 ,  0.43237751],
                                  [-0.82940893,  0.53004459]])

    grad_wrt_V_answer = np.array([[0.56468907, 1.36302356, 2.35084051],
                          [0.15442332, 0.3882369 , 0.68537352],
                          [0.25698882, 0.63947139, 1.12241063]])
    grad_wrt_W_answer = np.array([[-8.74981325],
                          [ 5.19938229],
                          [ 4.11304763],
                          [ 2.81802374]])

    close = [np.allclose(grad_wrt_U, grad_wrt_U_answer, 0.1),
             np.allclose(grad_wrt_V, grad_wrt_V_answer, 0.1),
             np.allclose(grad_wrt_W, grad_wrt_W_answer, 0.1)]
    if all(close):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. grad_wrt_U, grad_wrt_V, grad_wrt_W are correct.')
    else:
        if not close[0]:
            print(f'\n---  0/{pts} points. Returned incorrect value for grad_wrt_U.')
        if not close[1]:
            print(f'\n---  0/{pts} points. Returned incorrect value for grad_wrt_V.')
        if not close[2]:
            print(f'\n---  0/{pts} points. Returned incorrect value for grad_wrt_W.')
        print(f'Correct values are\n grad_wrt_U should be {grad_wrt_U_answer}\n grad_wrt_V should be {grad_wrt_V_answer}\n grad_wrt_W should be {grad_wrt_W_answer}')
        print(f'Your values are \n grad_wrt_U = {grad_wrt_U}\n grad_wrt_V = {grad_wrt_V}\n grad_wrt_W = {grad_wrt_W}')

except Exception as ex:
    print(f"\n--- 0/{pts} points. Function 'gradient' raised the exception\n")
    print(ex)





print('''\nTesting
    X = (np.arange(40).reshape(-1, 2) - 10) * 0.1
    T = X ** 3

    U, V, W, X_means, X_stds, T_means, T_stds = train(X, T, 100, 50, 10, 0.005)
''')


try:
    pts = 20

    X = (np.arange(40).reshape(-1, 2) - 10) * 0.1
    T = X ** 3

    U, V, W, X_means, X_stds, T_means, T_stds = train(X, T, 100, 50, 10000, 0.005)


    n_tests_correct = 0
    if U.shape == (3, 100):
        n_tests_correct += 1
        print('U.shape is correct')
    else:
        print(f'U.shape should be (3, 100).  Yours is {U.shape}')

    if V.shape == (101, 50):
        n_tests_correct += 1
        print('V.shape is correct')
    else:
        print(f'V.shape should be (101, 50).  Yours is {V.shape}')

    if W.shape == (51, 2):
        n_tests_correct += 1
        print('W.shape is correct')
    else:
        print(f'W.shape should be (51, 2).  Yours is {W.shape}')
        
    X_means_answer = np.array([0.9, 1.])
    X_stds_answer = np.array([1.15325626, 1.15325626])
    T_means_answer = np.array([4.32, 4.99])
    T_stds_answer = np.array([6.51909442, 7.25495617])

    if np.allclose(X_means, X_means_answer):
        n_tests_correct += 1
        print('X_means is correct.')
    else:
        print(f'X_means should be {X_means_answer}.\n Yours is {X_means}')
                       
    if np.allclose(X_stds, X_stds_answer):
        n_tests_correct += 1
        print('X_stds is correct.')
    else:
        print(f'X_stds should be {X_stds_answer}.\n Yours is {X_stds}')
                       
    if np.allclose(T_means, T_means_answer):
        n_tests_correct += 1
        print('T_means is correct.')
    else:
        print(f'T_means should be {T_means_answer}.\n Yours is {T_means}')
                       
    if np.allclose(T_stds, T_stds_answer):
        n_tests_correct += 1
        print('T_stds is correct.')
    else:
        print(f'T_stds should be {T_stds_answer}.\n Yours is {T_stds}')
                       
    if n_tests_correct == 7:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Shapes of weight matrices and standardization parameters are correct')
    else:
        print(f'\n--0  0/{pts} points.  {n_tests_correct} of 7 tests were correct.')

except Exception as ex:
    print(f"\n--- 0/{pts} points. Function 'train' raised the exception\n")
    print(ex)






print('''\nTesting
    X = (np.arange(40).reshape(-1, 2) - 10) * 0.1
    T = X ** 3

    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    vs = []
    for i in range(10):
        U, V, W, X_means, X_stds, T_means, T_stds = train(X, T, 100, 50, 10000, 0.005)
        Y = use(X, X_means, X_stds, T_means, T_stds, U, V, W)
        vs.append(rmse(Y, T))

    rmse_min, rmse_max = min(vs), max(vs)
''')


try:
    pts = 30

    X = (np.arange(40).reshape(-1, 2) - 10) * 0.1
    T = X ** 3

    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    vs = []
    for i in range(10):
        U, V, W, X_means, X_stds, T_means, T_stds = train(X, T, 100, 50, 10000, 0.005)
        Y = use(X, X_means, X_stds, T_means, T_stds, U, V, W)
        vs.append(rmse(Y, T))

    rmse_min, rmse_max = min(vs), max(vs)

    if rmse_min > 0.001 and rmse_max < 0.5:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Range of RMSE results are what is expected: 0.001 < RMSE < 0.5')
    else:
        print(f'\n--0  0/{pts} points.  Range of RMSE resuls should be between 0.001 and 0.5.  Yours are from {rmse_min} to {rmse_max}')

except Exception as ex:
    print(f"\n--- 0/{pts} points. Functions 'train' or 'use' raised the exception\n")
    print(ex)






name = os.getcwd().split('/')[-1]

print()
print('='*70)
print('{} Execution Grade is {} / 90'.format(name, exec_grade))
print('='*70)


print('''\n __ / 10 points Based on your discussion.''')

print()
print('='*70)
print('{} FINAL GRADE is  _  / 100'.format(name))
print('='*70)

print('''
Extra Credit:

Apply your functions to a data set from the UCI Machine Learning Repository.
Explain your steps and results in markdown cells.
''')

print('\n{} EXTRA CREDIT is 0 / 1'.format(name))

if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')
