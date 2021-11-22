# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:36:02 2011

@author: Peter Kner
Zernikes
"""

from operator import itemgetter

import numpy as N
from scipy.special import factorial as fac


def mkl(mx):
    ''' UCSF Haase ordering '''
    a = []
    for n in range(mx):
        for m in range(-n,n+1,2):
            a.append((n+N.abs(m),n,m))
    a.sort(key=itemgetter(0,1,2)) #a.sort(lambda a,b: int(a[0]-b[0] or a[1]-b[1] or b[2]-a[2]))
    c = []
    for j,t in enumerate(a):
        c.append(t[1:])
    b = N.array(c)
    #return dict(zip(c,N.arange(mx)))
    return (b,dict(zip(c,N.arange(len(a)))))
    
def jnm(mx):
    ''' Noll Zernike ordering !!! '''
    a = []
    for n in range(mx):
        for m in range(-n,n+1,2):
            a.append((n,m,abs(m)))
    a.sort(key=itemgetter(0,2,1)) #a.sort(lambda a,b: int(a[0]-b[0] or a[2]-b[2] or a[1]-b[1]))
    c = []
    for j,t in enumerate(a):
        c.append(t[:2])
    b = N.array(c)
    #return dict(zip(c,N.arange(mx)))
    return (b,dict(zip(c,N.arange(len(a)))))

global nj, nb
noll = True
if noll:
    nb,nj = jnm(15)
    ordering = "Noll"
else:
    nb,nh = mkl(15)
    ordering = "Wyant"

def rhofunc(i,j,x0,y0,Nx):
#    i = (i<=Nx/2)*i + (i>Nx/2)*(Nx-i)
#    j = (j<=Nx/2)*j + (j>Nx/2)*(Nx-j)
    x = (i-x0)
    y = (j-y0)
    r = N.sqrt(x**2 + y**2)
    return r

def thetafunc(i,j,x0,y0,Nx):
#    i = (i<=Nx/2)*i + (i>Nx/2)*(Nx-i)
#    j = (j<=Nx/2)*j + (j>Nx/2)*(Nx-j)
    x = (i-x0)
    y = (j-y0)
    t = N.arctan2(y,x)
    return t

def getrho(rad,orig,Nx):
    x0 = orig[0]
    y0 = orig[1]
    rho = N.fromfunction(lambda i,j: rhofunc(i,j,x0,y0,Nx),(Nx,Nx),dtype=N.float32)
    rho = rho/rad #N.where(rho<=rad,rho/rad,0)
    return rho
    
def gettheta(rad,orig,Nx):
    x0 = orig[0]
    y0 = orig[1]
    theta = N.fromfunction(lambda i,j: thetafunc(i,j,x0,y0,Nx),(Nx,Nx),dtype=N.float32)
    return theta

def R(m,n,rad,orig,Nx):
    ''' n>=m, n-m even '''
    rho = getrho(rad,orig,Nx)
    bigr = N.zeros((Nx,Nx),dtype=N.float32)
    for s in range(1+int((n-m)/2)):
        coeff = (-1)**s*fac(n-s)/(fac(s)*fac((n+m)/2-s)*fac((n-m)/2-s))
        bigr = bigr + coeff*rho**(n-2*s)
    bigr = bigr*(rho<=1.0)
    return bigr
    
def Z(m,n,rad=None,orig=None,Nx=256):
    if rad==None:
        rad = int(Nx/2)
    t = Nx/2-0.5
    cntr = [t,t]
    if (abs(m)>n):
        raise ValueError('m must be less than n!')
    if not((n-abs(m))%2==0):
        raise ValueError('n-m must be even!')
    theta = gettheta(rad,cntr,Nx)
    if m==0:
        Z = N.sqrt(n+1)*R(0,n,rad,cntr,Nx)
    elif (m>0):
        Z = N.sqrt(2*(n+1))*R(abs(m),n,rad,cntr,Nx)*N.cos(m*theta)
    else:
        Z = N.sqrt(2*(n+1))*R(abs(m),n,rad,cntr,Nx)*N.sin(m*theta)
    if not orig==None:
        orig = N.array(orig, dtype=N.int)-int(Nx/2)
        Z = N.roll(N.roll(Z,orig[0],0),orig[1],1)
    return Z
    
def Zm(j,rad=None,orig=None,Nx=256):
    ''' check convention with ordering '''
    n,m = nb[j]
    return Z(m,n,rad,orig,Nx)