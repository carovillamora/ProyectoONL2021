import numpy as np
from scipy.optimize import minimize


def func(x, sign=1.0):
    #Función Objetivo
    f=sign*(0.063*x[3]*x[6]-5.04*x[0]-0.035*x[1]-10*x[2]-3.36*x[4])
    return f
# Epsilon
h=0.000000000000915935

def grad(f,x,h):         #funcion que calcula el grafiente
    idt=h*np.eye(len(x)) #creamos una matriz identidad. 
                          #Nos ayudara a calcular las deriv parciales numéricas
    g=np.zeros(len(x))   #g es el vector gradiente pero inicialmente lo declaramos vacio

    for i in range(len(x)):   #este ciclo calcula cada entrada del vactor gradiente
        g[i] = (f(x+idt[i])-f(x-idt[i]))/(2*h) # Formula de Diferencias Finitas
    return g

# Restricciones (igualdades y desigualdades)
cons = [{'type': 'eq',
         'fun': lambda x: np.array([1.22*x[3]-x[0]-x[4]])},
        {'type': 'eq',
         'fun': lambda x: np.array([((98000*x[2])/(x[3]*x[8]+1000*x[2]))-x[5]])},
        {'type': 'eq',
         'fun': lambda x: np.array([((x[1]+x[4])/x[0])-x[7]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([x[0]*(1.12+0.13167*x[7]-0.00667*x[7]**2)-(99/100)*x[3]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([-x[0]*(1.12+0.13167*x[7]-0.00667*x[7]**2)+(100/99)*x[3]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([86.35+1.098*x[7]-0.038*x[7]**2+0.325*(x[5]-89)-(99/100)*x[6]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([-(86.35+1.098*x[7]-0.038*x[7]**2+0.325*(x[5]-89))+(100/99)*x[6]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([35.82-0.222*x[9]-(9/10)*x[8]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([-(35.82-0.222*x[9])+(10/9)*x[8]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([-133+3*x[6]-(99/100)*x[9]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([-(-133+3*x[6])+(100/99)*x[9]])}]

print("\nLos siguientes valores son usando un error de:", h, "\n")

res = minimize(func, [1745,12000,110,3048,1974,89.2,92.8,8,3.6,145], args=(-1.0), jac=None, 
               constraints=cons, method='SLSQP', options={'disp': True,'finite_diff_rel_step': grad,'eps': h})
 
print("\n", res.x)