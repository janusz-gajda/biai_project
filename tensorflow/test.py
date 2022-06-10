import subprocess

functions = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
for fun1 in functions:
    for fun2 in functions:
        subprocess.Popen("python tensorflow_main.py" + " fun1=" + fun1 +  " fun2=" + fun2+  " tensor1=8096" +  " tensor2=8096")