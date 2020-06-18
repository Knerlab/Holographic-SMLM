# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:20:52 2017

@author: abhij
"""

import os, sys, time
sys.path.append('..\\simsxy')
import glob
orig_path = "Z:/pythonfiles/storm/STORMholographic_postdec"

import scipy.ndimage as nd
from scipy.special import erf
#import Utility as U
#import zernike as Z

import numpy as N
#import numpy.random as rd
#import numpy.fft as ft
import tifffile as tf

def IntGauss1D(ii,x,sigma):
    '''Calculating PSF model using Error functions'''
    norm = 0.5/pow(sigma,2)
    return 0.5*(erf((ii-x+0.5)*N.sqrt(norm))-erf((ii-x-0.5)*N.sqrt(norm)))

def DerivativeIntGauss1D(ii,x,sigma,ph,PSFy):
	'''Calculating the Derivative of 1D Gaussian'''
	a = N.exp(-0.5*pow(((ii+0.5-x)/sigma), 2.0))
	b = N.exp(-0.5*pow(((ii-0.5-x)/sigma), 2.0))
	dudt =  -ph*(a-b)*PSFy/(N.sqrt(2.0*N.pi)*sigma)
	d2udt2 = -ph*((ii+0.5-x)*a-(ii-0.5-x)*b)*PSFy/(N.sqrt(2.0*N.pi)*pow(sigma, 3))
	return(dudt,d2udt2)

def DerivativeIntGauss1DSigma(ii,x,Sx,ph,PSFy):
	'''Calculating derivative of 1D Gaussian'''
	ax = N.exp(-0.5*pow(((ii+0.5-x)/Sx), 2.0))	
	bx = N.exp(-0.5*pow(((ii-0.5-x)/Sx), 2.0))
	dudt = -ph*(ax*(ii-x+0.5)-bx*(ii-x-0.5))*PSFy/(N.sqrt(2.0*N.pi)*pow(Sx,2))
	d2udt2 = -2.0*dudt/Sx-ph*(ax*pow((ii-x+0.5),3)-bx*pow((ii-x-0.5),3))*PSFy/(N.sqrt(2.0*N.pi)*pow(Sx,5))
	return(dudt,d2udt2)

def DerivativeIntGauss2DSigma(ii,jj,x,y,S,ph,PSFx,PSFy):
	'''Calculating derivative of 2D Gaussian'''
	doublederivative_x = DerivativeIntGauss1DSigma(ii,x,S,ph,PSFy)
	dSx = doublederivative_x[0]
	ddSx = doublederivative_x[1]
	doublederivative_y = DerivativeIntGauss1DSigma(jj,y,S,ph,PSFx)
	dSy = doublederivative_y[0]
	ddSy = doublederivative_y[1]
	dudt = dSx+dSy
	d2udt2 = ddSx+ddSy
	return(dudt,d2udt2)

def GaussFMaxMin2D(sz,sigma,data):
    MaxN = 0.0
    MinBG = 10.0e+10
    norm = 0.5/pow(sigma,2)
    '''loop over all pixels'''     
    for kk in range(sz):
        for ll in range(sz):
            filteredpixel = 0.0
            Sum = 0.0
            for ii in range(sz):
                for jj in range(sz):
                    filteredpixel+=N.exp(-pow((ii-kk-2),2)*norm)*N.exp(-pow((ll-jj-2),2)*norm)*data[ii,jj]
                    Sum+=N.exp(-pow((ii-kk-2),2)*norm)*N.exp(-pow((ll-jj-2),2)*norm)
            filteredpixel/=Sum
            MaxN = N.maximum(MaxN,filteredpixel)
            MinBG = N.minimum(MinBG,filteredpixel)
    return(MaxN,MinBG)
    
def com_2d(self,data,sz):
    '''Finds 2D center of mass'''    
    self.tmpx = 0.0
    self.tmpy = 0.0
    self.tmpsum = 0.0
    for a in range(sz):
        for b in range(sz):
            self.tmpx += data[a,b]*a
            self.tmpy += data[a,b]*b
            self.tmpsum += data[a,b]
    self.x = self.tmpx/self.tmpsum
    self.y = self.tmpy/self.tmpsum
    return(self.x,self.y)
   
            

    