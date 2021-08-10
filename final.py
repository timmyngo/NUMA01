# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:35:27 2021

@author: Timmy
"""
from numpy import *
from scipy import *
from pylab import *
from matplotlib.pyplot import *

class Mesh(object):
    def __init__(self, C, N, *a):
        self.C = C
        self.N = N
        self.element = array(a)
        self.centroid = array([sum(self.element[0,:]) / 3, sum(self.element[1,:]) / 3])

    def __repr__(self):
        return '(%s, %s)' % (self.C, self.N)

    def __Jacobian__(self):
        C = self.C
        N = self.N
        s = size(self.N[:, 0]) #number of triangles
        J = []
        
        #creating empty arrays of length s 
        [x_1,x_2,x_3,y_1,y_2,y_3] = [zeros(s),zeros(s),zeros(s),zeros(s),zeros(s),zeros(s)]

        #for loop to calculate the coordinates
        for i in range(s): 
            x_1[i] = C[int(N[i,0])-1, 0]
            x_2[i] = C[int(N[i,1])-1, 0]
            x_3[i] = C[int(N[i,2])-1, 0]
            y_1[i] = C[int(N[i,0])-1, 1]
            y_2[i] = C[int(N[i,1])-1, 1]
            y_3[i] = C[int(N[i,2])-1, 1]
            
            #creating the vectors
            a = array([x_3[i]-x_2[i], y_3[i]-y_2[i]])
            b = array([x_3[i]-x_1[i], y_3[i]-y_1[i]])
            c = array([x_2[i]-x_1[i], y_2[i]-y_1[i]])
            
            #calculating angles in each triangle using dot product
            alpha = arccos((abs(b @ c))/(norm(b)*norm(c)))
            beta = arccos((abs(a @ c))/(norm(a)*norm(c)))
            gamma = arccos((abs(b @ a))/(norm(a)*norm(b))) 
            minangle = min(alpha, beta, gamma)
            
            #jacobimatris
            j = [[x_2[i]-x_1[i], x_3[i]-x_1[i]],
                [y_2[i]-y_1[i], y_3[i]-y_1[i]]]
            
            if minangle >= 1.e-6: #minimum angle 
                J.append(abs(det(j)))
            else:  
                raise ValueError("minangle is:",minangle)
        return J
        
    def __Integral__(self, f):
        J = Mesh(self.C, self.N).__Jacobian__()
        S = sum(J)
        I = S*1/2*(1/3*f(0,0) + 1/3*f(0,1) + 1/3*f(1,0))
        return I

    def __Area__(self):
        C = self.C
        N = self.N
        s = size(self.N[:, 0]) #number of mesh
        [x_1,x_2,x_3,y_1,y_2,y_3] = [zeros(s),zeros(s),zeros(s),zeros(s),zeros(s),zeros(s)]
        A = []
        for i in range(s):
            x_1[i] = C[int(N[i,0])-1, 0]
            x_2[i] = C[int(N[i,1])-1, 0]
            x_3[i] = C[int(N[i,2])-1, 0]
            y_1[i] = C[int(N[i,0])-1, 1]
            y_2[i] = C[int(N[i,1])-1, 1]
            y_3[i] = C[int(N[i,2])-1, 1]
            a = array([x_3[i]-x_2[i], y_3[i]-y_2[i]])
            b = array([x_3[i]-x_1[i], y_3[i]-y_1[i]])
            A.append(abs(cross(a,b)/2))
        Area = sum(A) #sum of the area of all the triangles
        return Area
        
    def __plotting__(self):
        C = self.C
        N = self.N
        s = size(self.N[:, 0])
        x = C[:, 0]
        y = C[:, 1]
        L = []
        for n in range(s):
            L.append(n)
        return plt.tripcolor(x, y, N-1, edgecolors = 'b', facecolors = array(L))
    
#mesh
c1 = loadtxt('./coord1.txt').T
c2 = loadtxt('./coordinates_dolfin_coarse.txt').T
n1 = loadtxt('./elementnode1.txt').T
n2 = loadtxt('./nodes_dolfin_coarse.txt').T
m = Mesh(c1, n1)
m2 = Mesh(c2, n2)

#functions
f = lambda x,y: 1
J = m.__Jacobian__()
I = m.__Integral__(f)
A = m.__Area__()

#prints
print('Approximate integral:', I) 
print('Sum of area of total elements:', A)
print('Absolute error between approximate integral method and sum of all elements area:', abs(I-A))

#plot
figure(0)
m.__plotting__()
title('Mesh')
xlabel('$x$')
ylabel('$y$')
Mesh(c1, n1)
show()

#mesh dolphin
J2 = m2.__Jacobian__()
I2 = m2.__Integral__(f)  
A2 = m2.__Area__()

#prints
print('Approximate integral:', I2) 
print('Sum of area of total elements:', A2)
print('The area of the dolphin is approximately', 1-A2)
print('Absolute error between approximate integral method and sum of all elements area:', abs(I2-A2))

#plot of dolphin
figure(1)
m2.__plotting__()
title('Dolphin')
xlabel('$x$')
ylabel('$y$')
Mesh(c2, n2)
show()