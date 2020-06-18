# -*- coding: utf-8 -*-
"""
Created on Mon May 06 18:06:45 2013

This is a simulation of storm holographic imaging

v2: modify filenames for saving
add code to run sequence with different aberrations

@author: marar
"""
import os, sys, time
sys.path.append('..\\simsxy')

import scipy.ndimage as nd
import Utility as U
import zernike as Z
import glob

import numpy as N
import numpy.random as rd
import numpy.fft as ft
import tifffile as tf
import scipy.ndimage as nd
from scipy.special import erf
from scipy import interpolate
import gausslib as G
from collections import defaultdict

orig_path = "Z:/pythonfiles/storm/STORMholographic_postdec"
#orig_path = "C:\\Users\\Abhijit Marar\\Desktop\\Code\\STORMholographic_postdec"
#orig_path = "C:\\Users\\abhij\\Desktop\\Code\\Code\\STORMholographic_postdec"
import pylab

pi = N.pi
fft2 = N.fft.fft2
ifft2 = N.fft.ifft2
fftshift = N.fft.fftshift

class sim(object):

    def __init__(self):
        self.nnp = 5             # avg # of fluorophores per image
        self.iip =6000           # avg # of photons per fluorophore
        self.avg_no = 32
        self.nx = 1024
        self.dx = 13.
        self.na = 1.42
        self.wl = 0.670
        self.nzarr = 15
        self.zarr = 0.0*rd.randn(15)
        self.zarr[4] = 0.0
        self.zarr[11] = 0.0
        self.f_doe = 300.e+3       #focal length of SLM
        self.f_obj = 3.e+3          #focal length of objective    
        self.d1 = 3.e+3          #distance between objective and SLM
        self.z_h = 150.e+3         #distance between SLM and CCD aperture
        #self.mask = N.zeros((self.nx,self.nx),dtype=N.float32)
        self.img = N.zeros((self.nx,self.nx),dtype=N.float32)
        self.z_r = 0.            #reconstruction distance
        self.trans_mag = 0.      #transverse magnification
        self.Nangs = 3
        self.angles = N.array([0.0,2*pi/3,4*pi/3])
        self.mult = N.zeros((self.nx,self.nx),dtype=N.float32)        
        self.r = 163.0       
        
        
    def __del__(self):
        pass    
    
    
    def getoneptcr(self):
        ''' one flurophore at the center'''        
        dx = self.dx
        #nxh = int(self.nx/2)
        self.xcntr = 0.0
        self.ycntr = 0.0
        self.exth = int(1.0/dx) # half extent in pixels
    
    def getobj(self):
        ''' put fluorophores in circle '''
        dx = self.dx
        nxh = int(self.nx/2)
        Np = 1000
        rad = 0.5
        phi = N.linspace(0,2.0*pi,Np)
        self.xps = rad*N.cos(phi) 
        self.yps = rad*N.sin(phi) 
        self.exth = int(1.0/dx) # half extent in pixels
    
    def getobj3d(self):
        '''put fluorophores on a helix'''
        dx = self.dx         
        Np = 1000
        rad = 0.500
        phi = N.linspace(0,4*pi,Np)
        z = N.linspace(-5,5,Np)
        self.xps = rad*N.cos(phi)
        self.yps = rad*N.sin(phi)
        self.zps = z + self.f_obj
        self.exth = int(1.0/dx) # half extent in pixels
    
    def getobj3d_line(self):
        '''put fluorophores on a straight Line'''
        dx = self.dx        
        Np = 5000
        self.xcntr = 0.0
        self.ycntr = 0.0
        z = N.linspace(-10,10,Np)
        self.zps = z + self.f_obj
        self.exth = int(1.0/dx) # half extent in pixels
    
    def getonept(self):
        ''' one fluorophore '''
        Np = 2
        dx = self.dx
        nxh = int(self.nx/2)
        self.xps = N.array([nxh*dx,0.75*nxh*dx])
        self.yps = N.array([nxh*dx,nxh*dx])
        self.exth = int(1.0/dx) # half extent in pixels

    def getaberr(self):
        wl = self.wl
        na = self.na
        n2 = 1.512
        dp = 1/(self.nx*self.dx)
        radius = (na/wl)/dp
        ## prepare for focus mode
        x = N.arange(-self.nx/2,self.nx/2,1)
        X,Y = N.meshgrid(x,x)
        rho = N.sqrt(X**2 + Y**2)/radius
        msk = (rho<=1.0).astype(N.float64)
        self.defoc = msk*(2*pi)*(n2/wl)*N.sqrt(1-(na*msk*rho/n2)**2)
        #########################
        #msk = U.shift(U.discArray((self.nx,self.nx),radius))/(pi*radius**2)
        msk = U.shift(U.discArray((self.nx,self.nx),radius))/N.sqrt(pi*radius**2)/self.nx
        phi = N.zeros((self.nx,self.nx))
        for m in range(1,self.nzarr):
            phi = phi + self.zarr[m]*Z.Zm(m,radius,[0,0],self.nx)
        self.wf = msk*N.exp(1j*phi).astype(N.complex64)

    def addpsf(self,x,y,z,I):
        # create phase
        nx = self.nx
        alpha = 2*pi/nx/self.dx
        g = lambda m, n: N.exp(1j*alpha*(m*x+n*y)).astype(N.complex64)
        defoc = N.exp(1j*z*self.defoc)
        ph = N.fromfunction(g, (nx,nx), dtype=N.float32)
        ph = U.shift(ph*defoc)
        wfp = N.sqrt(I)*ph*self.wf
        #wfp = wfp
        self.img = self.img + abs(fft2(wfp))**2
	
    def gethgrams(self):
        focus = self.f_obj        
        self.img[:,:] = 0.0
        #self.den[:,:] = 20.0
        nxh = int(self.nx/2)
        self.hstack =  N.zeros((3,self.nx,self.nx),dtype=N.float32)
        angles = self.angles
        # get points
        self.Np = rd.poisson(self.nnp)
        pti = rd.randint(0,5000,self.Np)
        #self.xp = self.xps[pti]
        #self.yp = self.yps[pti]
        #self.zp = self.zps[pti]
        self.zp = N.array([3000.0])
        '''For flurophore in center'''        
        self.Np_sng = 1        
        self.xp = N.array([self.xcntr])
        self.yp = N.array([self.ycntr])       
        self.Ip = 2000.        
        #self.getfzpimg(xp,yp,3.e+3,0.0,Ip)            
        #self.Ip = rd.poisson(self.iip,self.Np)
        print self.xp
        print self.yp
        print self.zp #print focus/self.zp
        print self.Ip
        # create psfs
        for n in range(self.Nangs):
            for m in range(self.Np_sng):
                '''usually in range(self.Np)'''
                self.hol_1angle = self.getfzpimg_nonoise(self.xp[m],self.yp[m],self.zp[m],angles[n],self.Ip)
            #self.hstack[n,:,:] = rd.poisson(self.img.copy())
            self.hstack[n,:,:] = self.hol_1angle    
            self.img[:,:] = 0.0    
        #done!

    def gethgrams_oneangle(self):
        focus = self.f_obj        
        self.img[:,:] = 0.0
        #self.den[:,:] = 20.0
        nxh = int(self.nx/2)
        self.frame =  N.zeros((self.nx,self.nx),dtype=N.float32)
        #angles = self.angles
        '''get points'''
        self.Np = rd.poisson(self.nnp)
        pti = rd.randint(0,1000,self.Np)
        self.xp = self.xps[pti]
        self.yp = self.yps[pti]
        self.zp = self.zps[pti]
        #self.zp = focus
        '''For flurophore in center'''        
        #self.Np_sng = 1        
        #self.xp = self.xcntr
        #self.yp = self.ycntr        
        #self.Ip = 4000.        
        #self.getfzpimg(xp,yp,3.e+3,0.0,Ip)            
        self.Ip = rd.poisson(self.iip,self.Np)
        print self.xp
        print self.yp
        print self.zp #print focus/self.zp
        print self.Ip
        # create psfs
        for m in range(self.Np):
            self.hol_1angle = self.getfzpimg_nonoise(self.xp[m],self.yp[m],self.zp[m],0.0,self.Ip[m])
        #self.hstack[n,:,:] = rd.poisson(self.img.copy())
        self.frame = rd.poisson(self.hol_1angle)    
        self.img[:,:] = 0.0    
        #done!

    

    def getfzpimg(self,x,y,z,theta,Ip):
    #create Fresnel Zone Plates for individual single emitters
        self.indivdual = N.zeros((self.nx,self.nx),dtype=N.float32)
        nx = self.nx
        wl = self.wl
        f_o = self.f_obj
        f_d = self.f_doe
        z_h = self.z_h
        d1 = self.d1
        dx = self.dx
        nxh = int(nx/2)
        if(abs(z)) == f_o:
            self.z_r = (z_h-f_d)
            self.trans_mag = z_h/f_d
        else:
            f_e = (z*f_o)/(f_o-z)
            f_1 = (f_d*(f_e+d1))/(f_d-(f_e+d1));
            self.z_r = -(((f_1+z_h)*(f_e+d1+z_h))/(f_1-f_e-d1))
            self.trans_mag = (z_h*f_e)/(z*(f_e+d1));
        #the intensity function
        g = lambda m, n: N.exp((1j*pi/wl/self.z_r)*(((dx*(m-nxh)) - self.trans_mag*x)**2 + ((dx*(n-nxh)) - self.trans_mag*y)**2)+1j*theta).astype(N.complex64)
        self.amp = N.fromfunction(g, (nx,nx), dtype=N.complex64)
        self.individual = (2 + self.amp + N.conjugate(self.amp)).real
        #self.mult = Ip/N.sum(self.individual)
        #self.img += self.individual*self.mult
        self.img += self.individual        
        return(self.img)
        #td = '%d' % time.time()
        #tf.imsave('fzp_single' + td[-5:] + '.tif',self.img.astype(N.float32))
        #done!
     
    def getfzpimg_nonoise(self,x,y,z,theta,Ip):
        '''create Fresnel Zone Plates for individual single emitters'''
        #self.img = 300*N.ones((self.nx,self.nx),dtype=N.float32)
        #self.img = rd.poisson(self.img)
        self.individual = N.zeros((self.nx,self.nx),dtype=N.float32)        
        nx = self.nx
        wl = self.wl
        f_o = self.f_obj
        f_d = self.f_doe
        z_h = self.z_h
        d1 = self.d1
        dx = self.dx
        nxh = int(nx/2)
        r = self.r
        if(abs(z)) == f_o:
            self.z_r = (z_h-f_d)
            self.trans_mag = z_h/f_d
        else:
            f_e = (z*f_o)/(f_o-z)
            f_1 = (f_d*(f_e+d1))/(f_d-(f_e+d1));
            self.z_r = -(((f_1+z_h)*(f_e+d1+z_h))/(f_1-f_e-d1))
            self.trans_mag = (z_h*f_e)/(z*(f_e+d1));
        #the intensity function
        g = lambda m, n: N.exp((1j*pi/wl/self.z_r)*(((dx*(m-nxh)) - self.trans_mag*x)**2 + ((dx*(n-nxh)) - self.trans_mag*y)**2)+1j*theta).astype(N.complex64)
        self.amp = N.fromfunction(g, (nx,nx), dtype=N.complex64)
        self.individual = N.abs(2+ self.amp + N.conjugate(self.amp))
        #self.individual = 300*rd.poisson(self.individual)
        x_coord, y_coord = (((self.trans_mag*x)/dx)+nxh), (((self.trans_mag*y)/dx)+nxh)
        '''artificial BPP'''                    
        a,b = N.ogrid[-x_coord:nx-x_coord, -y_coord:nx-y_coord]
        self.mask = a*a + b*b <= r*r
        self.individual = self.individual*self.mask       
        self.mult = Ip/N.sum(self.individual)
        self.img += self.individual*self.mult
        return self.img
        #td = '%d' % time.time()
        #tf.imsave('fzp_single' + td[-5:] + '.tif',self.img.astype(N.float32))
        #done!
    
    def getoneimg(self):
        self.img[:,:] = 20.0
        # get points
        Np = rd.poisson(self.nnp)
        pti = rd.randint(0,1000,Np)
        #xp = self.xps[pti]
        #yp = self.yps[pti]
        xp = self.xcntr
        yp = self.ycntr
        zp = self.zps[pti]
        #Ip = rd.poisson(self.iip,Np) #[self.iip]
        Ip = rd.exponential(self.iip,Np)
        # create psfs
        for m in range(Np):
            self.addpsf(xp,yp,zp[m],Ip[m])
        # noise
        self.img = rd.poisson(self.img)
        # done!

    
    def getoneimgdrift(self,imgno):
        self.img[:,:] = 20.0
        # get points
        Np = rd.poisson(self.nnp)
        pti = rd.randint(0,1000,Np)
        xp = self.xps[pti] - (4.e-4)*imgno
        yp = self.yps[pti]
        #Ip = rd.poisson(self.iip,Np) #[self.iip]
        Ip = rd.exponential(self.iip,Np)
        # create psfs
        for m in range(Np):
            self.addpsf(xp[m],yp[m],Ip[m])
        # noise
        self.img = rd.poisson(self.img)
        # done!
        
    def getoneimgcoma(self):
        self.img[:,:] = 20.0
        Npts = 2
        # adjust aberr
        self.zarr[4] = 2.5*rd.randn()
        self.zarr[12] = 2.5*rd.randn()
        self.getaberr()
        # get points
        Np = rd.poisson(self.nnp)
        print(Np)
        pti = rd.randint(0,Npts,Np)
        print(pti)
        xp = self.xps[pti]
        yp = self.yps[pti]
        #Ip = rd.poisson(self.iip,Np) #[self.iip]
        Ip = rd.exponential(self.iip,Np)
        # create psfs
        for m in range(Np):
            self.addpsf(xp[m],yp[m],Ip[m])
        # noise
        self.img = rd.poisson(self.img)
        # done!        
        
    def runseq(self,Ns):
        ''' create sequence of raw storm data
            the data is saved to storm1.tif and
            can be analyzed with QuickPalm or RapidStorm '''
        #os.mkdir('temp')
        nxh = int(self.nx/2)    
        self.stack = N.zeros((3*Ns,512,512),dtype=N.float32)
        self.stack_bg = N.zeros((3*Ns,512,512),dtype=N.float32)
        self.stack_noisy = N.zeros((3*Ns,512,512),dtype=N.uint32)
        #nxh = int(self.nx/2)
        beg = nxh-int(nxh/2)
        edd = nxh+int(nxh/2)
        # get some stats
        #rpk = N.zeros((Ns))
        #rm = N.zeros((Ns))
        for m in range(Ns):
            #self.getoneimgdrift(m)
            #self.getoneimgcoma()
            self.gethgrams()
            self.stack[3*m:3*m+3,:,:] = self.hstack[:,beg:edd,beg:edd]            
            #self.stack[m,:,:] = self.frame
            #if self.img.max()>0:
            #    rpk[m] = self.img.max()
            #    rm[m] = self.getmetric()
        # save file
        self.stack_bg = self.stack + 0.000667
        self.stack_noisy = rd.poisson(self.stack_bg)
        td = '%d' % time.time()
        tf.imsave('fzp_' + td[-5:] + '.tif',self.stack_noisy.astype(N.uint32),photometric = 'minisblack')
        # plot stats
        #pylab.figure(1)
        #pylab.plot(rpk,'b-o')
        #pylab.hist(rpk)
        #pylab.figure(2)
        #pylab.plot(rm,'b-o')
        #pylab.hist(rm)
        #print rpk.mean()/rpk.std()
        #print rm.mean()/rm.std()
        return td
    
    def runseq2(self,Ns=500):
        ''' create sequence of raw storm data
            the data is saved to storm1.tif and
            can be analyzed with QuickPalm or RapidStorm '''
        #os.mkdir('temp')
        self.stack = N.zeros((Ns,2*self.exth,2*self.exth),dtype=N.float32)
        nxh = int(self.nx/2)
        beg = nxh-self.exth
        edd = nxh+self.exth
        # get some stats
        rpk = N.zeros((Ns))
        rm = N.zeros((Ns))
        for m in range(Ns):
            #self.getoneimgdrift(m)
            self.getoneimg()
            self.stack[m,:,:] = self.img[beg:edd,beg:edd]
            if self.img.max()>0:
                rpk[m] = self.img.max()
                #rm[m] = self.getmetric()
        # save file
        tf.imsave('storm3D.tif',self.stack.astype(N.uint16))
        # plot stats
        pylab.figure(1)
        #pylab.plot(rpk,'b-o')
        pylab.hist(rpk)
        pylab.figure(2)
        #pylab.plot(rm,'b-o')
        pylab.hist(rm)
        print rpk.mean()/rpk.std()
        print rm.mean()/rm.std()
    
    def runseqtest(self,Ns=500,Nz=4):
        ''' create a sequance of storm images with increasing
            aberrations. Ns = # of raw images per set
            Nz = # of cycles '''
        self.stack = N.zeros((Nz*Ns,2*self.exth,2*self.exth),dtype=N.float32)
        nxh = int(self.nx/2)
        beg = nxh-self.exth
        edd = nxh+self.exth
        # get some stats
        Navg = self.avg_no
        rpk = N.zeros((Nz*Ns))
        ravg = N.zeros((Nz*Ns))
        for mm in range(Nz):
            self.zarr = mm*0.1*rd.randn(self.nzarr)
            self.getaberr()
            for nn in range(Ns):
                ind = mm*Ns+nn
                self.getoneimg()
                self.stack[ind,:,:] = self.img[beg:edd,beg:edd]                
                rpk[ind] = self.img.max()
                if ind==0:
                    ravg[ind] = rpk[ind]
                elif (rpk[ind]<1.5*20):
                    ravg[ind] = ravg[ind-1]
                elif (ind<=Navg):
                    ravg[ind] = 1.0*rpk[:(ind+1)].sum()/ind
                else:
                    ravg[ind] = 1.0*rpk[ind]/Navg +ravg[ind-1] - 1.0*rpk[ind-Navg-1]/Navg
        # save file
        tf.imsave('storm1.tif',self.stack.astype(N.uint16))
        # plot stats
        pylab.plot(rpk,'b-o',ravg,'r-')
        
    def runseqcorr(self,Ns=500):
        ''' Correct aberrations '''
        # create aberrations
        self.zarr = 1.0*rd.randn(self.nzarr)
        zarr1 = N.zeros((self.nzarr))
        zarr1[:] = self.zarr
        self.getaberr()
        zp = 3
        push = 0.1
        zpush = 0.0
        zcount = 0
        zmax = 10
        oldavg = 0.0
        # initialize
        self.stack = N.zeros((Ns,2*self.exth,2*self.exth),dtype=N.float32)
        nxh = int(self.nx/2)
        beg = nxh-self.exth
        edd = nxh+self.exth
        # get some stats
        Navg = self.avg_no
        rpk = N.zeros((Ns))
        ravg = N.zeros((Ns))
        for nn in range(Ns):
            self.getoneimg()
            self.stack[nn,:,:] = self.img[beg:edd,beg:edd]                
            # Get intensity metrics
            rpk[nn] = self.img.max()
            if nn==0:
                ravg[nn] = rpk[nn]
            elif (rpk[nn]<2.5*20):
                ravg[nn] = ravg[nn-1]
            elif (nn<=Navg):
                ravg[nn] = 1.0*rpk[:(nn+1)].sum()/nn
            else:
                #ravg[nn] = 1.0*rpk[nn]/Navg +ravg[nn-1] - ravg[nn-Navg]
                pavg = rpk[(nn-Navg):nn]
                ravg[nn] = pavg[pavg>50].mean()
                #ravg[nn] = rpk[(nn-Navg):nn].mean()
            # Adjust aberration
            # try random change
            zcount += 1
            if zcount>Navg:
                if (ravg[nn]<oldavg):
                    self.zarr -= zpush
                zpush = push*rd.randn(15)
                self.zarr += zpush
                self.getaberr()
                zcount = 0
                oldavg = ravg[nn]
        # save file
        tf.imsave('storm1.tif',self.stack.astype(N.uint16))
        # plot stats
        pylab.plot(rpk,'b-o',ravg,'r-')
        print (abs(self.zarr)-abs(zarr1)).sum()
        
    def runscmodal(self,Ns=500):
        ''' Correct aberrations '''
        # create aberrations
        zarr1 = N.zeros((self.nzarr))
        zarr1[:] = self.zarr
        self.getaberr()
        zp = 3
        push = 0.2
        zpush = 0.0
        zcount = 0
        zmax = 10
        oldavg = 0.0
        # initialize
        self.stack = N.zeros((Ns,2*self.exth,2*self.exth),dtype=N.float32)
        nxh = int(self.nx/2)
        beg = nxh-self.exth
        edd = nxh+self.exth
        # get some stats
        Navg = self.avg_no
        rpk = N.zeros((Ns))
        ravg = N.zeros((Ns))
        for nn in range(0,Ns-3,3):
            z3 = self.zarr[3]
            # 1st image
            self.getaberr()
            self.getoneimg()
            self.stack[nn,:,:] = self.img[beg:edd,beg:edd]        
            p0 = self.img.max()
            # 2nd image
            self.zarr[3] = z3 + push
            self.getaberr()
            self.getoneimg()
            self.stack[nn+1,:,:] = self.img[beg:edd,beg:edd]        
            pp = self.img.max()
            # 3rd image
            self.zarr[3] = z3 - push
            self.getaberr()
            self.getoneimg()
            self.stack[nn+2,:,:] = self.img[beg:edd,beg:edd]        
            pm = self.img.max()
            # optimize
            if (p0>pp) and (p0>pm):
                self.zarr[3] = z3
#                if push>=0.02:
#                    push = push - 0.02
            elif (pp>p0) and (pp>pm):
                self.zarr[3] = z3 + push
            else:
                self.zarr[3] = z3 - push
            # don't correct if any intensity is too small
            #if (min(p0,pp,pm) < (0.5*self.iip)) or (max(p0,pp,pm)>(1.5*self.iip)):
            #    self.zarr[3] = z3
            # Get intensity metrics
            rpk[nn] = self.img.max()
            if nn==0:
                ravg[nn] = rpk[nn]
            elif (rpk[nn]<2.5*20):
                ravg[nn] = ravg[nn-1]
            elif (nn<=Navg):
                ravg[nn] = 1.0*rpk[:(nn+1)].sum()/nn
            else:
                #ravg[nn] = 1.0*rpk[nn]/Navg +ravg[nn-1] - ravg[nn-Navg]
                pavg = rpk[(nn-Navg):nn]
                ravg[nn] = pavg[pavg>50].mean()
                #ravg[nn] = rpk[(nn-Navg):nn].mean()
            # Adjust aberration
            # try random change
            zcount += 1
            if zcount>Navg:
                zcount = 0
                oldavg = ravg[nn]
        # save file
        tf.imsave('storm1.tif',self.stack.astype(N.uint16))
        # plot stats
        pylab.plot(rpk,'b-o',ravg,'r-')
        print push
        print N.sqrt((self.zarr**2).sum())
        
    def genetic(self, verbose=False):
        nxh = int(self.nx/2)
        beg = nxh-self.exth
        edd = nxh+self.exth
        # setup parameters
        population_size = 20
        gen_no = 100
        population = N.zeros((population_size,self.nzarr), dtype=N.float32)
        metric = N.zeros(population_size)
        stats = N.zeros((gen_no,3), dtype=N.float32)
        self.stack = N.zeros((population_size*gen_no,2*self.exth,2*self.exth),dtype=N.float32)
        # create initial population        
        for m in range(population_size):
            population[m] = 10*rd.randn(self.nzarr)
        # loop
        aberr = self.zarr
        ind = 0
        self.getoneimg()
        self.stack[ind,:,:] = self.img[beg:edd,beg:edd] 
        for gen in range(gen_no):
            # get metric for each member of population
            for m in range(population_size):
                self.zarr = aberr + population[m]
                self.getaberr()
                self.getoneimg()
                self.stack[ind,:,:] = self.img[beg:edd,beg:edd]
                ind += 1
                metric[m] = self.getmetric2()
            # sort
            rank = metric.argsort()
            for g in range(population_size/2):
                # select parents
                p1 = rd.randint(population_size/2,population_size)
                parent1 = population[rank[p1]]
                p2 = rd.randint(population_size/2,population_size)
                parent2 = population[rank[p2]]                
                # generate random binary template
                bin_temp = (rd.uniform(size=self.nzarr)>0.5).astype(N.float32)
                # breed
                child = parent1*bin_temp + parent2*(1-bin_temp)
                # mutate
                if (rd.rand()<0.05):
                    # maybe mutation should be stronger in some way
                    child += 6.0*rd.randn(self.nzarr)
                # replace lowest ranked members
                population[rank[g]] = child
            # keep track of stats to plot later
            stats[gen,0] = metric.max()
        # done
        return stats
        
        
    def getmetric(self,verbose=False):
        nx = self.nx
        imgf = fft2(self.img-20.0)
        den = abs(imgf).sum() #imgf[0:5,0:5].sum()
        imgf = abs(imgf/den) # normalize (intensity)
        dp = 1/(nx*self.dx)
        ext = (2*self.na/self.wl)/dp
        msk = U.discArray((nx,nx),ext/4+2,(0,0))-U.discArray((nx,nx),ext/4-2,(0,0))
        metric = (msk*imgf).sum()/msk.sum()
        #metric = abs(den)
        if verbose:
            pylab.imshow(msk,interpolation='nearest')
        return metric
        
    def getmetric2(self, verbose=False):
        ''' try gaussian filter '''
        if ((self.img-20.).sum()<3000.):
            return 0.0
        nx = self.nx
        dx = self.dx
        dp = 1./(nx*dx)
        imgf = fftshift(fft2(self.img-20.0))
        sigma = (0.20)*(self.na/self.wl)/dp
        pedge = 0.4*2.*(self.na/self.wl)/dp
        filt1 = N.fromfunction(lambda i,j: N.exp(-0.5*((i-nx/2)**2+(j-nx/2)**2)/sigma**2), (nx,nx))
        filt2 = N.fromfunction(lambda i,j: (1-N.exp(-0.5*((i-nx/2)**2+(j-nx/2)**2)/sigma**2))*(N.sqrt((i-nx/2)**2+(j-nx/2)**2)<pedge), (nx,nx))
        #filt2 = 1.0-filt1
        metric = abs(filt2*imgf).sum()/abs(filt1*imgf).sum()
        if verbose:
            pylab.figure()
            pylab.imshow(filt1,interpolation='nearest')
            pylab.figure()
            pylab.imshow(filt2,interpolation='nearest')
        return metric
        
    def metrictest(self, mode=4):
        zarr = self.zarr
        aberr = N.zeros(self.nzarr)
        self.zarr = aberr
        mags = N.arange(-10,10,0.1)
        res = N.zeros(mags.shape)
        for m, amp in enumerate(mags):
            self.zarr[mode] = amp
            self.getaberr()
            self.getoneimg()
            res[m] = self.getmetric2()
        pylab.plot(mags,res)
        return True
        
        
    def run_aberration_sequence():
        Na = 41 # no. of measurements
        ab = N.linspace(0.0, 2.0, Na)
        p = sim()
        p.iip = 1000
        p.getobj()
        nz = p.nzarr
        out = []
        for ma in ab:
            p.zarr = (ma/N.sqrt(nz-3))*rd.randn(nz)
            p.zarr[:3] = 0.0
            p.getaberr()
            arms = N.sqrt((p.zarr**2).sum())
            cd = p.runseq(2000)
            out.append((cd,arms))
        return N.array(out)
        
        
        