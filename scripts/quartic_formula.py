from math import pi
from cmath import exp, sqrt
import numpy as np
import csv

def roots2(a,b,c):
    bp=b/2    
    delta=bp*bp-a*c
    u1=(-bp-delta**.5)/a
    u2=-u1-b/a
    return u1,u2  

J=exp(2j*pi/3)
Jc=1/J

def cardan(a,b,c,d):
    u=np.empty(2,np.complex128)
    z0=b/3/a
    a2,b2 = a*a,b*b    
    p=-b2/3/a2 +c/a
    q=(b/27*(2*b2/a2-9*c/a)+d)/a
    D=-4*p*p*p-27*q*q
    r=sqrt(-D/27+0j)        
    u=((-q-r)/2)**0.33333333333333333333333
    v=((-q+r)/2)**0.33333333333333333333333
    w=u*v
    w0=abs(w+p/3)
    w1=abs(w*J+p/3)
    w2=abs(w*Jc+p/3)
    if w0<w1: 
        if w2<w0 : v*=Jc
    elif w2<w1 : v*=Jc
    else: v*=J        
    return [u+v-z0, u*J+v*Jc-z0,u*Jc+v*J-z0]

p = 0.6
q = 0.2
r = 3 # FORMERLY: r = 1
b = 0.9

solutions = [["solution 1", "solution 2", "solution 3", "beta"]]
beta_vals = np.arange(0, 1, step = 0.001)
for b in beta_vals:
    a = (1 - p) * ((q - 1) ** 2) - ((1 - p) ** 3)

    b_1 = (1 - p) * (q - 1) * (2 * r * b - q + 1) + (q - 1) * (2 * r * (1 - b) * (1 - p) + ((q - 1) ** 2) - 2 * r * b * (q - 1))
    b_2 = 2 * r * (1 - b) * ((q - 1) ** 2)
    b_3 = -2 * ((1 - p) ** 2) * (1 - q) - 4 * r  * b * ((1 - p) ** 2)
    b_val = b_1 + b_2 - b_3

    c_1 = (2 * r * b - q + 1) * (2 * r * (1 - b) * (1 - p) + ((q - 1) ** 2) - 2 * r * b * (q - 1)) + (q - 1) * (2 * r * (1 - b) * (q - 1) - 4 * (r ** 2) * b * (1 - b))
    c_2 = 8 * (r ** 2) * ((1 - b) ** 2) * (q - 1)
    c_3 = (1 - p) * ((1 - q) ** 2) + 4 * r * b * (1 - p) * (1 - q) + 4 * (r ** 2) * (b ** 2) * (1 - p)
    c = c_1 + c_2 - c_3

    d_1 = (2 * r * b - q + 1) * (2 * r * (1 - b) * (q - 1) - 4 * (r ** 2) * b * (1 - b))
    d_2 = 8 * (r ** 3) * ((1 - b) ** 3)
    d_3 = 0
    d = d_1 + d_2 - d_3

    solution = cardan(a, b_val, c, d)
    solution = [value.real for value in solution]
    solution.append(b)
    solutions.append(solution)

# solns from newest a,b,c,d vals
with open('throughput/analytic_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(solutions)