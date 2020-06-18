# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 16:39:42 2017

@author: abhij
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

orig_path = "F:\FINCH\Code\Python"
#orig_path = "C:\\Users\\Abhijit Marar\\Desktop\\Code\\STORMholographic_postdec"
#orig_path = "C:\\Users\\abhij\\Desktop\\Code\\Code\\STORMholographic_postdec"
import pylab

pi = N.pi
fft2 = N.fft.fft2
ifft2 = N.fft.ifft2
fftshift = N.fft.fftshift

class finch(object):
    
    
    def __init__(self,img_stack=None):
        self.nz, self.ny, self.nx = img_stack.shape
        self.img = img_stack
        self.dx = 13.0
        self.dx_mag = 0.120
        #self.nx_recon,self.ny_recon = 64
        self.N = N.linspace(-3.840,3.720,64)
        self.xx,self.yy = N.meshgrid(self.N,self.N)
        self.wl = 0.670
        self.recon_z = -150.e+3
        self.angles = N.array([0.0,2*pi/3,4*pi/3])
        self.f_doe = 300.e+3       #focal length of SLM
        self.f_obj = 3.e+3       #focal length of objective    
        #Wself.f2 = 400.e+3        
        self.d1 = 3.e+3         #distance between objective and SLM
        self.z_h = 150.e+3       #distance between SLM and CCD aperture        
        #self.Np = 20
        self.z_r = 0.
        self.z_rel_gen = N.linspace(-10.,10.,51)
        self.z_abs_gen = self.z_rel_gen + self.f_obj
        
        
    def __del__(self):
        pass
    
    def finch_recon(self,recon_dist):
        realsz = int(self.nz/3)
        wl = self.wl
        nx = self.nx
        'changes for BFLY camera'
        ny = self.ny
        nxh = int(nx/2)
        nyh = int(ny/2)
        dx = self.dx
        beg_y = nyh-int(nyh/12)
        edd_y = nyh+int(nyh/12)
        beg_x = nxh-int(nxh/12)
        edd_x = nxh+int(nxh/12)
        self.intensity_stack = N.zeros((realsz,ny,nx),dtype=N.complex64)
        self.strm_stack = N.zeros((realsz,ny,nx),dtype=N.float64)        
        angles = self.angles        
        img = self.img.astype(N.float64)
        for m in range(realsz):
            #self.final_intensity = img[m]            
            self.final_intensity = (img[3*m]*(N.exp(1j*angles[2])-N.exp(1j*angles[1])) + 
                               img[3*m+1]*(N.exp(1j*angles[0])-N.exp(1j*angles[2])) +
                               img[3*m+2]*(N.exp(1j*angles[1])-N.exp(1j*angles[0])))
            self.intensity_stack[m,:,:] = self.final_intensity
            g = lambda m, n: fftshift(ifft2(fft2(fftshift(self.final_intensity))*fft2(fftshift(N.exp((1j*pi/wl/recon_dist)*((dx*(m-nyh))**2+(dx*(n-nxh))**2)))))).astype(N.complex128)         
            self.recond = N.fromfunction(g, (ny,nx), dtype=N.complex64)
            'Entire FOV'
            self.strm_stack[m,:,:] = abs(self.recond)            
            #self.strm_stack[m,:,:] = (abs(self.recond)-abs(self.recond).min())*(2**16-1)/(abs(self.recond).max()-abs(self.recond).min())
            #self.strm_stack[m,:,:] = abs(self.recond[beg_y:edd_y,beg_x:edd_x])
        #td = '%d' % time.time()
        #tf.imsave('storm_' + td[-7:] + '.tif',self.strm_stack.astype(N.float32), photometric = 'minisblack')
        #done!
    
    def recon_dist_calc(self,zpos_abs,Np):
        self.zr_stack = N.zeros((Np,1),dtype=N.float32)
        self.mag = N.zeros((Np,1),dtype=N.float32)        
        f_o = self.f_obj
        f_d = self.f_doe
        z_h = self.z_h
        d1 = self.d1
        self.z_abs = zpos_abs
        for m in range(Np):
            if(abs(self.z_abs[m])) == f_o:
                self.z_r = (z_h-f_d)
                self.trans_mag = z_h/f_d
            else:
                f_e = (self.z_abs[m]*f_o)/(f_o-self.z_abs[m])
                f_1 = (f_d*(f_e+d1))/(f_d-(f_e+d1));
                self.z_r = -(((f_1+z_h)*(f_e+d1+z_h))/(f_1-f_e-d1))
                self.trans_mag = (z_h*f_e)/(self.z_abs[m]*(f_e+d1))
            self.zr_stack[m,:] = self.z_r
            self.mag[m,:] = self.trans_mag
        return(self.zr_stack)
        #done!

    def recon_dist_calc2(self,zpos_abs,Np):
        self.zr_stack = N.zeros((Np,1),dtype=N.float32)        
        f_o = self.f_obj
        f_d = self.f_doe
        z_h = self.z_h
        f_tl = 180.e+3
        d1 = 183.e+3
        f_4 = 120.e+3
        d2 = 300.e+3
        d3 = 240.e+3
        #f2 = self.f2/1.e+3
        self.z_abs = zpos_abs
        for m in range(Np):
            if(abs(self.z_abs[m])) == f_o:
                self.z_r = -(z_h-f_d)
                #self.trans_mag = z_h/f_o
            else:
                f_e = (self.z_abs[m]*f_o)/(f_o-self.z_abs[m])
                f_g = (f_tl*(f_e+d1))/(f_tl-(f_e+d1))
                f_h = (f_4*(f_g+d2))/(f_4-(f_g+d2))
                f_k = (f_d*(f_h+d3))/(f_d-(f_h+d3))
                self.z_r = ((f_k+z_h)*(z_h+d3+f_h))/(f_k-f_h-d3)
                #self.trans_mag = (z_h*f_e)/(self.z_abs[m]*(f_e+d1))
            self.zr_stack[m,:] = self.z_r
        return(self.zr_stack)
        #done!              
    
    def finch_recon3D(self,recon_dist,img,Np):
        self.imsz = img.shape 
        self.zpos = recon_dist        
        realsz = int(self.nz/3)
        wl = self.wl
        nx = self.nx
        ny = self.ny
        #sz = 128 # kernel size
        #szh = 64
        nyh = int(ny/2)
        nxh = int(nx/2)
        dx = self.dx
        beg = nxh-int(nxh/16)
        edd = nxh+int(nxh/16)
        #Np = self.Np
        #z = self.z
        self.intensity_stack = N.zeros((Np,ny,nx),dtype=N.complex64)
        self.strm_stack = N.zeros((Np,32,32),dtype=N.float32)        
        angles = self.angles        
        #img = self.img
        for l in  range(realsz):
            for m in range(Np):
                #self.final_intensity = img[l]
                self.final_intensity = (img[3*l]*(N.exp(1j*angles[2])-N.exp(1j*angles[1])) + 
                               img[3*l+1]*(N.exp(1j*angles[0])-N.exp(1j*angles[2])) +
                               img[3*l+2]*(N.exp(1j*angles[1])-N.exp(1j*angles[0])))
                self.intensity_stack[m,:,:] = self.final_intensity
                self.z_test = self.zpos[m]               
                g = lambda m, n: ft.fftshift(ft.ifft2(ft.fft2(ft.fftshift(self.final_intensity))*ft.fft2(ft.fftshift(N.exp((1j*pi/wl/self.z_test)*((dx*(m-nxh))**2+(dx*(n-nyh))**2)))))).astype(N.complex64)         
                self.recond = N.fromfunction(g, (ny,nx), dtype=N.complex64)
                #self.strm_stack[m,:,:] = (abs(self.recond)-abs(self.recond).min())*(2**16-1)/(abs(self.recond).max()-abs(self.recond).min())
                self.strm_stack[m,:,:] = abs(self.recond)[beg:edd,beg:edd]     
            td = '%d' % time.time()
            tf.imsave('storm_' + td[-5:] + '.tif',self.strm_stack.astype(N.float32))
            self.strm_stack[:,:,:] = 0.0
        #done!
    
    
    def finch_recon3D_oneangle(self,recon_dist,img,Np):
        self.imsz = img.shape 
        self.zpos = recon_dist        
        realsz = self.nz
        wl = self.wl
        nx = self.nx
        ny = self.ny
        #sz = self.nz # kernel size
        #szh = int(sz/2.0)
        nyh = int(ny/2)
        nxh = int(nx/2)
        dx = self.dx
        beg = nxh-int(nxh/16)
        edd = nxh+int(nxh/16)
        #Np = self.Np
        #z = self.z
        self.intensity_stack = N.zeros((Np,nx/16,nx/16),dtype=N.complex64)
        self.strm_stack = N.zeros((Np,nx/32,nx/32),dtype=N.float32)        
        angles = self.angles        
        #img = self.img
        for l in  range(realsz):
            for m in range(Np):
                self.final_intensity = img[l]
                #self.final_intensity = (img[3*l]*(N.exp(1j*angles[2])-N.exp(1j*angles[1])) + 
                #              img[3*l+1]*(N.exp(1j*angles[0])-N.exp(1j*angles[2])) +
                #               img[3*l+2]*(N.exp(1j*angles[1])-N.exp(1j*angles[0])))
                #self.intensity_stack[m,:,:] = self.final_intensity
                self.z_test = self.zpos[m]               
                g = lambda m, n: ft.fftshift(ft.ifft2(ft.fft2(ft.fftshift(self.final_intensity))*ft.fft2(ft.fftshift(N.exp((1j*pi/wl/self.z_test)*((dx*(m-nxh))**2+(dx*(n-nxh))**2)))))).astype(N.complex64)         
                self.recond = N.fromfunction(g, (ny,nx), dtype=N.complex64)
                #self.strm_stack[m,:,:] = (abs(self.recond)-abs(self.recond).min())*(2**16-1)/(abs(self.recond).max()-abs(self.recond).min())
                self.strm_stack[m,:,:] = abs(self.recond[beg:edd,beg:edd])     
            td = '%d' % time.time()
            tf.imsave('storm_' + td[-5:] + '.tif',self.strm_stack.astype(N.float32))
            self.strm_stack[:,:,:] = 0.0
        #done!
    
    
    def finch_recon3D_finer(self,recon_dist,img,Np):
        '''reconstructing the holograms to finer slices'''        
        self.imsz = img.shape 
        self.zpos = recon_dist        
        realsz = int(self.imsz[0]/3)
        wl = self.wl
        nx = self.nx
        ny = self.ny
        nxh = int(nx/2)
        dx = self.dx
        beg = nxh-int(nxh/2)
        edd = nxh+int(nxh/2)
        #Np = self.Np
        #z = self.z
        self.intensity_stack = N.zeros((Np,nx,nx),dtype=N.complex64)
        self.strm_stack = N.zeros((Np,nx,nx),dtype=N.float32)        
        angles = self.angles        
        #img = self.img
        for l in  range(realsz):
            for m in range(Np):
                #self.final_intensity = img            
                self.final_intensity = (img[3*l]*(N.exp(1j*angles[2])-N.exp(1j*angles[1])) + 
                                        img[3*l+1]*(N.exp(1j*angles[0])-N.exp(1j*angles[2])) +
                                        img[3*l+2]*(N.exp(1j*angles[1])-N.exp(1j*angles[0])))
                self.intensity_stack[m,:,:] = self.final_intensity
                self.z_test = self.zpos[m]               
                g = lambda m, n: ft.fftshift(ft.ifft2(ft.fft2(ft.fftshift(self.final_intensity))*ft.fft2(ft.fftshift(N.exp((1j*pi/wl/self.z_test)*((dx*(m-nxh))**2+(dx*(n-nxh))**2)))))).astype(N.complex64)         
                self.recond = N.fromfunction(g, (nx,nx), dtype=N.complex64)
                #self.strm_stack[m,:,:] = abs(self.recond[beg:edd,beg:edd]) 
                self.strm_stack[m,:,:] = abs(self.recond) 
        i,j,k = N.unravel_index(self.strm_stack.argmax(),self.strm_stack.shape)
        self.strm_stack = self.strm_stack[:,j-16:j+16,k-16:k+16]
        return(self.strm_stack)
         #done!!!       
    
    def wavelet(self,path):
        '''Wavelet filtering with kernel using B-Spline Basis
        function of the 3rd order with a scaling factor of 2'''
        self.k1 = N.array([0.0625,0.250,0.375,0.250,0.0625],dtype=N.float32)
        self.k2 = N.array([0.0625,0.0,0.250,0,0.375,0,0.250,0,0.0625],dtype=N.float32)        
        #path_in = "Z:\\pythonfiles\\storm\\STORMholographic_postdec\\3D\\Helix\\5000photons_noise_1angle\\Reconstructions"        
        path_in = path        
        os.chdir(path_in)          
        for filename in glob.glob('*.tif'):
            self.img = tf.imread(filename)
            self.img = self.img +20.0            
            self.V0 = self.img
            self.nz,self.ny,self.nx = self.img.shape
            self.fltrd_img = N.zeros((self.nz,self.ny,self.nx),dtype=N.float32)        
            '''kernels'''        
            for m in range(self.nz):
                '''first wavelet level'''
                self.V1_inner = nd.filters.convolve1d(self.V0[m],self.k1,axis=1,mode='reflect')
                self.V1 = nd.filters.convolve1d(self.V1_inner,self.k1,axis=0,mode='reflect')
                '''second wavelet level'''            
                self.V2_inner = nd.filters.convolve1d(self.V1,self.k2,axis=1,mode='reflect')
                self.V2 = nd.filters.convolve1d(self.V2_inner,self.k2,axis=0,mode='reflect')
                '''Watershed'''            
                self.W = self.V1 - self.V2
                self.fltrd_img[m,:,:] = self.W
            outname = 'Filtered_' + filename[6:11] + '.tif'
            print(outname)
            tf.imsave(outname, self.fltrd_img.astype(N.float32),photometric = 'minisblack')
        os.chdir(orig_path)  

    
    def crop_sm(self,path):
        '''Crop smaller region aroung single molecule'''
        path_in = path
        os.chdir(path_in)
        self.crpd_imgs = N.zeros((80,32,32),dtype=N.float32)
        for frame,filename in enumerate(glob.glob('*.tif')):
            self.img_init = tf.imread(filename)
            i,j,k = N.unravel_index(self.img_init.argmax(),self.img_init.shape)
            self.img_crpd = self.img_init[:,j-16:j+16,k-16:k+16]
            tf.imsave(filename[6:11] + '_crpd_recon.tif', self.img_crpd.astype(N.float32),photometric = 'minisblack')
            print(frame)
        os.chdir(orig_path)
    
    def max_pos(self,path,coords):
        path_in = path
        os.chdir(path_in)
        self.max_stack_fltrd = N.zeros((100,32,32),dtype=N.float32)
        for frame,filename in enumerate(glob.glob('*.tif')):
            self.img3D_fltrd_crpd = tf.imread(filename)
            self.max_stack_fltrd[frame,:,:] = self.img3D_fltrd_crpd[coords[frame,1],:,:]
        tf.imsave('3D_fltrd_max_stack'+'.tif',self.max_stack_fltrd.astype(N.float32),photometric='minisblack')
        os.chdir(orig_path)
        
        
        
    def approx_pos(self,thrsh,path):
        '''Finding approximate positions of molecules using local intensity maximum
        and 26-connected neighborhood'''
        path_in = path
        os.chdir(path_in)
        self.coords = [[0,0,0,0]]
        self.coords_a = [[0,0,0,0]]
        self.coords_b = [[0,0,0,0]]
        for frame,filename in enumerate(glob.glob('*.tif')):
            print(frame)
            self.fltrd = tf.imread(filename)
            for l in range(1,self.fltrd.shape[0]-1):
                for m in range(1,self.fltrd.shape[1]-1):
                    for n in range(1,self.fltrd.shape[2]-1):
                        cube = self.fltrd[l-1:l+2,m-1:m+2,n-1:n+2]            
                        i,j,k = N.unravel_index(cube.argmax(),cube.shape)
                        if (self.fltrd[l,m,n] >= thrsh) & ([i,j,k] == [1,1,1]):
                            self.coords = N.append(self.coords,[[frame,l,m,n]],axis=0)
        self.coords = self.coords[1:,:]
        self.fin_coords_approx = self.frame_del(self.coords,self.coords[:,0])
        os.chdir(orig_path)
    
    def approx_pos_2D(self,thrsh,img):
        '''Finding approximate positions of molecules using local intensity maximum
        and 8-connected neighborhood'''
        #path_in = "C:\\Users\\Abhijit Marar\\Desktop\\Code\\STORMholographic_postdec\\3D\\Helix\\Filtered"        
        #path_in = "C:\\Users\\abhij\\Desktop\\Code\\Code\\STORMholographic_postdec\\3D\\Helix\\Filtered"      
        #path_in = "Z:/pythonfiles/storm/STORMholographic_postdec/500ph/450 holograms/Filtered"
        #os.chdir(path_in)
        self.coords = [[0,0,0]]
        self.coords_a = [[0,0,0,0]]
        self.coords_b = [[0,0,0,0]]
        for frame in range(img.shape[0]):        
            for m in range(1,img.shape[1]-1):
                for n in range(1,img.shape[2]-1):
                    self.square = img[frame][m-1:m+2,n-1:n+2]            
                    i,j = N.unravel_index(self.square.argmax(),self.square.shape)
                    if (img[frame][m,n] >= thrsh) & ([i,j] == [1,1]):
                        self.coords = N.append(self.coords,[[frame,m,n]],axis=0)
        self.coords = self.coords[1:,:]
        #self.fin_coords_approx = self.frame_del(self.coords,self.coords[:,0])
        #os.chdir(orig_path)
    
    def list_duplicates(self,seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return ((key,locs) for key,locs in tally.items() 
                                if len(locs)>1)    
                                
    def frame_del(self,coords,frame_list):
       '''find fluorophores too close to each other in z and delete them''' 
       self.coords_bad = [[0,0,0,0]]
       self.indices = [0]
       for dup in sorted(self.list_duplicates(frame_list)):
           self.test = dup[0]
           self.dup_indices = dup[1]
           self.dup_coords = coords[self.dup_indices]
           if ((N.abs(N.diff(self.dup_coords[:,1]))).min() <= 4):
               #or (N.abs(N.diff(self.dup_coords[:,3]))).min() <= 4                
               self.coords_bad = N.append(self.coords_bad,self.dup_coords,axis=0)
               self.indices = N.append(self.indices,self.dup_indices,axis=0)
               continue
       self.coords_bad = self.coords_bad[1:,:]
       self.indices = self.indices[1:]
       self.reduced_coords_list = N.delete(coords,self.indices,axis=0)
       return(self.reduced_coords_list)
       
    def finer_z(self,coord,slices):
        '''creates finer slice distances'''
        #Change linspace to arange to ensure constant distances between slices
        self.zfine_stack = N.zeros((slices,1),dtype=N.float32)
        f_o = self.f_obj
        '''Reconstruction distance corresponding to given above frames'''
        self.zpt = self.z_rel_gen[coord]
        if (coord == 1 or coord == 49):
            nslices = 2*slices
            self.zpt_above = self.z_rel_gen[coord-1]
            self.zpt_below = self.z_rel_gen[coord+1]
            self.finer_zstack = N.linspace(self.zpt_above,self.zpt_below,nslices)
            self.z_abs_finer = self.finer_zstack + f_o
            self.zfine_stack = self.recon_dist_calc(self.z_abs_finer,nslices)
        else:
            nslices = 2*slices
            self.zpt_above = self.z_rel_gen[coord-2]
            self.zpt_below = self.z_rel_gen[coord+2]
            self.finer_zstack = N.linspace(self.zpt_above,self.zpt_below,nslices)
            self.z_abs_finer = self.finer_zstack + f_o
            self.zfine_stack = self.recon_dist_calc(self.z_abs_finer,nslices)
        return(self.zfine_stack, self.finer_zstack, nslices)        
        
    def sub_region(self,sz,img,xcoord,ycoord):
        ''' cut out subregions from original image'''
        self.img_finer = img
        self.xpos = xcoord
        self.ypos = ycoord
        '''frame of interest'''
        self.ROI_finer = self.img_finer[:,self.xpos-int(N.floor(sz/2)):self.xpos+int(N.floor(sz/2)+1),self.ypos-int(N.floor(sz/2)):self.ypos+int(N.floor(sz/2)+1)]
        self.xx_roi = self.xx[:,self.xpos-int(N.floor(sz/2)):self.xpos+int(N.floor(sz/2)+1)]
        self.yy_roi = self.yy[self.ypos-int(N.floor(sz/2)):self.ypos+int(N.floor(sz/2)+1),:]        
        return(self.ROI_finer,self.xx_roi,self.yy_roi)        
    
    def finer_stack(self,slices):
        '''creates stack based on the above finer slices'''
        coords = self.coords
        for i in range(len(coords)):
            self.frame = coords[i,0]
            self.z_approx = coords[i,1]
            self.img_set = self.img[3*self.frame:3*self.frame+3]
            self.dist = self.finer_z(self.z_approx,slices)                  
            self.finer_img_stck = self.finch_recon3D_finer(self.dist,self.img_set,slices)
            td = '%d' % time.time()
            tf.imsave('finer_' + td[-5:] + '.tif',self.finer_img_stck.astype(N.float32))
    #done!  
            
    def loc_coords(self,sz,slices,Sigma,iterations):
        '''Main function to call after approx positions have been found'''
        coords = self.fin_coords_approx
        self.final_res = [[0,0,0,0,0,0,0]]
        #self.CRLB_final = [[0,0,0,0,0,0,0]]      
        for i in range(len(coords)):
            print i
            self.mid_res = [[0,0,0,0,0]] 
            self.frame = coords[i,0]
            self.z_approx = coords[i,1]
            self.img_set = self.img[3*self.frame:3*self.frame+3]
            self.dist = self.finer_z(self.z_approx,slices)                  
            self.finer_img_stck = self.finch_recon3D_finer(self.dist[0],self.img_set,self.dist[2])
            td = '%d' % time.time()
            tf.imsave('finer_' + td[-5:] + '.tif',self.finer_img_stck.astype(N.float32))
            '''Cut out ROI's for each finer stack'''            
            self.x_approx = coords[i,2]
            self.y_approx = coords[i,3]
            self.finer_subrgn = self.sub_region(sz,self.finer_img_stck,self.x_approx,self.y_approx)
            self.real_spacex = self.finer_subrgn[1]
            self.real_spacey = self.finer_subrgn[2]                
            tf.imsave('roits_' + str(i) + '.tif',self.finer_subrgn[0].astype(N.float32))
            '''Localize each frame in the ROI'''
            for num in range(len(self.finer_subrgn[0])):
                self.loc_single = self.MLEfit_sigma(self.finer_subrgn[0][num,:,:],Sigma,sz,iterations)
                self.mid_res = N.append(self.mid_res,self.loc_single,axis = 0)
            self.mid_res = self.mid_res[1:,:]
            self.max_intensity_pixel = N.array(N.where(self.finer_subrgn[0] == self.finer_subrgn[0].max())) 
            self.z_intensity_line_profile = self.finer_subrgn[0][:,self.max_intensity_pixel[1],self.max_intensity_pixel[2]]
            self.z_intensity_line_profile = N.reshape(self.z_intensity_line_profile,[len(self.z_intensity_line_profile)])
            #self.z_fit_res = self.z_fit(self.dist[1],self.mid_res[:,2])            
            self.z_fit_res = self.z_fit(self.dist[1],self.z_intensity_line_profile)
            self.z_loc_um = self.z_fit_res[0]
            #self.sgm = self.z_fit_res[1]            
            self.I = self.z_fit_res[1]
            self.intrpltd_res = self.x_y_pos(self.dist[1],self.mid_res[:,0],
                                                  self.mid_res[:,1],self.mid_res[:,4],
                                                  self.mid_res[:,3],self.z_loc_um)                 
            self.x_loc_roi = self.intrpltd_res[0]
            self.y_loc_roi = self.intrpltd_res[1]
            #self.final_coords_roi = N.append(self.final_coords_roi,[[self.x_loc_roi,self.y_loc_roi]],axis =0)                                     
            #self.x_loc = (self.real_spacex[0,0]+(self.dx_mag*self.intrpltd_res[0]))*1.e+3
            #self.y_loc = (self.real_spacey[0,0]+(self.dx_mag*self.intrpltd_res[1]))*1.e+3
            self.z_loc_nm = 1.e+3*(self.z_fit_res[0])            
            self.I_loc = self.intrpltd_res[2]
            self.bg_loc = self.intrpltd_res[3]
            '''****************I_loc is sigma************************'''
            self.final_res = N.append(self.final_res,[[self.frame,self.x_loc_roi,self.y_loc_roi,self.z_loc_nm[0],self.I_loc[0],self.I[0],self.bg_loc[0]]],axis=0)
        self.final_res = self.final_res[1:,:]
        #self.final_coords_roi = self.final_coords_roi[1:,:]
        
    def loc_coords2(self,sz,slices,Sigma,iterations):
        '''find fluorophore position by gaussian fitting'''
        coords = self.fin_coords_approx
        self.final_res = [[0,0,0,0,0,0,0]]
        for i in range(len(coords)):
            print i
            self.mid_res = [[0,0,0,0,0]] 
            self.frame = coords[i,0]
            self.z_approx = coords[i,1]
            self.img_set = self.img[3*self.frame:3*self.frame+3]
            self.dist = self.finer_z(self.z_approx,slices)
            self.finer_img_stck = self.finch_recon3D_finer(self.dist[0],self.img_set,self.dist[2])
            td = '%d' % time.time()
            tf.imsave('finer_' + td[-5:] + '.tif',self.finer_img_stck.astype(N.float32))
            '''Cut out ROI's for each finer stack'''            
            self.x_approx = coords[i,2]
            self.y_approx = coords[i,3]
             

    def x_y_pos(self,z_array,x_array,y_array,I_array,bg_array,zpos):
        ''' interpolate x,y,I and bg to find result corresponding to z_loc'''        
        '''z_array: dist[1],x_array = inter_res[:,0],y_array = inter_res[:,1]''' 
        '''zpos = z_loc, I_array = inter_res[:,2], bg_array = inter_res[:3]'''            
        self.min_dist = N.abs(z_array[0]-z_array[1])
        self.idx = N.where(N.abs(z_array-zpos) < self.min_dist)
        if (self.idx[0].size == 0):
            return(N.array([0]),N.array([0]),N.array([0]),N.array([0]))
        self.idx_l = self.idx[0][0]
        if (self.idx_l == 0 or self.idx_l == 39):
            self.x_loc_interp = N.array([x_array[self.idx_l]])
            self.y_loc_interp = N.array([y_array[self.idx_l]])
            self.I_loc_interp = N.array([I_array[self.idx_l]])
            self.bg_loc_interp = N.array([bg_array[self.idx_l]])
     
        else:
            self.idx_r = self.idx[0][1]
            self.x_loc_l = x_array[self.idx_l]
            self.x_loc_r = x_array[self.idx_r]
            self.y_loc_l = y_array[self.idx_l]
            self.y_loc_r = y_array[self.idx_r]
            self.I_loc_l = I_array[self.idx_l]
            self.I_loc_r = I_array[self.idx_r]
            self.bg_loc_l = bg_array[self.idx_l]
            self.bg_loc_r = bg_array[self.idx_r]            
            self.abscissa =  N.array([z_array[self.idx_l],z_array[self.idx_r]])
            self.ord_x = N.array([self.x_loc_l,self.x_loc_r])
            self.ord_y = N.array([self.y_loc_l,self.y_loc_r])         
            self.ord_I = N.array([self.I_loc_l,self.I_loc_r])
            self.ord_bg = N.array([self.bg_loc_l,self.bg_loc_r])
            f_x = interpolate.interp1d(self.abscissa,self.ord_x)
            f_y = interpolate.interp1d(self.abscissa,self.ord_y)
            f_I = interpolate.interp1d(self.abscissa,self.ord_I)
            f_bg = interpolate.interp1d(self.abscissa,self.ord_bg)        
            self.x_loc_interp = f_x(zpos)
            self.y_loc_interp = f_y(zpos)
            self.I_loc_interp = f_I(zpos)
            self.bg_loc_interp = f_bg(zpos)        
        return(self.x_loc_interp,self.y_loc_interp,self.I_loc_interp,self.bg_loc_interp)        
                 
    
    def z_fit(self,x,y):
        '''Performs polynomial fit of distance vs sigma/intensity and finds minimum'''
        print(x)
        print(y)
        self.coeff = N.polyfit(x,y,2)
        self.pol = N.poly1d(self.coeff)
        self.crit = self.pol.deriv().r
        self.r_crit = self.crit[self.crit.imag==0].real
        #self.bst_sgm = self.pol(self.r_crit)
        self.bst_I = self.pol(self.r_crit)
        #return(self.r_crit,self.bst_sgm)
        return(self.r_crit,self.bst_I)

    def sub_region_2D(self,sz,img,xcoord,ycoord):
        ''' cut out subregions from original image'''
        self.img = img
        self.xpos = xcoord
        self.ypos = ycoord
        '''frame of interest'''
        self.ROI_finer = self.img[self.xpos-int(N.floor(sz/2)):self.xpos+int(N.floor(sz/2))+1,self.ypos-int(N.floor(sz/2)):self.ypos+int(N.floor(sz/2)+1)]
        self.xx_roi = self.xx[:,self.xpos-int(N.floor(sz/2)):self.xpos+int(N.floor(sz/2))+1]
        self.yy_roi = self.yy[self.ypos-int(N.floor(sz/2)):self.ypos+int(N.floor(sz/2))+1,:]        
        return(self.ROI_finer,self.xx_roi,self.yy_roi)    
        
    def loc_coords_2D(self,img,sz,Sigma,iterations):
        '''Main function to call after approx positions have been found'''
        coords = self.coords
        '''deleting first column for 2D lolcalization after 3D approx pos'''
        coords = N.delete(coords,1,1)
        image = img
        self.final_res = [[0,0,0,0,0,0]]
        for i in range(len(coords)):
            print i
            #self.mid_res = [[0,0,0,0,0]] 
            self.frame = coords[i,0]
            self.x_approx = coords[i,1]
            self.y_approx = coords[i,2]
            self.roi = self.sub_region_2D(sz,image[self.frame],self.x_approx,self.y_approx)
            self.real_spacex = self.roi[1]
            self.real_spacey = self.roi[2]                
            '''Localize each frame in the ROI'''
            self.loc_single = self.MLEfit_sigma(self.roi[0],Sigma,sz,iterations)
            self.x_loc = self.loc_single[0,0]
            self.y_loc = self.loc_single[0,1]
            #self.x_loc = (self.real_spacex[0,0]+(self.dx_mag*self.loc_single[0,0]))*1.e+3
            #self.y_loc = (self.real_spacey[0,0]+(self.dx_mag*self.loc_single[0,1]))*1.e+3            
            self.final_res = N.append(self.final_res,[[self.frame,self.x_loc,self.y_loc,self.loc_single[0,2],self.loc_single[0,3],self.loc_single[0,4]]],axis = 0)            
        self.final_res = self.final_res[1:,:]             
          
    def MLEfit_sigma(self,data,PSFsigma,sz,iterations):
        params = 5
        self.M = N.zeros((params,params),dtype=N.float)
        self.Minv = N.zeros((params,params),dtype=N.float)
        self.CRLB = N.zeros((params,1),dtype=N.float)
        self.dudt = N.zeros((params,1),dtype=N.float)
        self.d2udt2 = N.zeros((params,1),dtype=N.float)
        self.theta = N.zeros((params,1),dtype=N.float)
        self.maxjump = N.array([1.0, 1.0, 100.0, 2.0, 0.5])
        self.gamma = N.array([1.0,1.0,0.5,1.0,1.0])
        ''' Calculating Center of Mass'''
        self.init_coords = nd.measurements.center_of_mass(data)
        #self.init_coords = G.com_2d(data,sz)            
        ''' Initializing fitting parameters, theta = {x,y,I,bg,sigma}'''            
        self.theta[0] = self.init_coords[0]
        self.theta[1] = self.init_coords[1]
        #self.theta[3] = N.minimum(10.0e+10,N.min(data))
        self.int_bg = G.GaussFMaxMin2D(sz,PSFsigma,data)        
        self.theta[3] = self.int_bg[1]
        self.theta[2] = N.maximum(0.0, (self.int_bg[0]-self.theta[3])*2*N.pi*pow(PSFsigma,2))
        #self.theta[3] = 10.
        #self.theta[2] = 500.        
        self.theta[4] = PSFsigma
        '''main iterative loop'''
        for itr in range(iterations):
            self.num = N.zeros((params,1),dtype=N.float)
            self.den = N.zeros((params,1),dtype=N.float)
            for i in range(sz):
               for j in range(sz):
                   self.PSFx = G.IntGauss1D(i,self.theta[0],self.theta[4])
                   self.PSFy = G.IntGauss1D(j,self.theta[1],self.theta[4])
                   self.model = self.theta[3]+self.theta[2]*self.PSFx*self.PSFy 
                   '''Calculating the Derivatives'''
                   self.x_derivative = G.DerivativeIntGauss1D(i,self.theta[0],PSFsigma,self.theta[2],self.PSFy)
                   self.dudt[0] = self.x_derivative[0]
                   self.d2udt2[0] = self.x_derivative[1]
                   self.y_derivative = G.DerivativeIntGauss1D(j,self.theta[1],PSFsigma,self.theta[2],self.PSFx)
                   self.dudt[1] = self.y_derivative[0]
                   self.d2udt2[1] = self.y_derivative[1]
                   self.sigma_derivative = G.DerivativeIntGauss2DSigma(i,j,self.theta[0],self.theta[1],self.theta[4],self.theta[2],self.PSFx,self.PSFy)
                   self.dudt[4] = self.sigma_derivative[0]
                   self.d2udt2[4] = self.sigma_derivative[1]
                   self.dudt[2] = self.PSFx*self.PSFy
                   self.d2udt2[2] = 0.0
                   self.dudt[3] = 1.0
                   self.d2udt2[3] = 0.0
                   '''Newton-Raphson Iteration'''
                   self.cf = 0.0
                   self.df = 0.0
                   if (self.model>10.0e-3):
                       self.cf = data[i,j]/self.model-1
                       self.df = data[i,j]/pow(self.model,2)
                   self.cf = N.minimum(self.cf,10.0e4)
                   self.df = N.minimum(self.df,10.0e4)
                   for ll in range(params):
                       self.num[ll]+=self.dudt[ll]*self.cf
                       self.den[ll]+=self.d2udt2[ll]*self.cf-pow(self.dudt[ll],2)*self.df
            '''The update'''
            if(itr<5):
               for ll in range(params):
                   self.theta[ll]-=self.gamma[ll]*N.minimum(N.maximum(self.num[ll]/self.den[ll], -self.maxjump[ll]), self.maxjump[ll])    
                   print self.theta[0]
                   print self.theta[1]
            else:
               for ll in range(params):
                   self.theta[ll]-=N.minimum(N.maximum(self.num[ll]/self.den[ll], -self.maxjump[ll]), self.maxjump[ll])
                   print self.theta[0]
                   print self.theta[1]            
            '''Any other constraints'''
            self.theta[2]=N.maximum(self.theta[2], 1.0)
            self.theta[3]=N.maximum(self.theta[3], 0.01)
            self.theta[4]=N.maximum(self.theta[4], 0.5)
            self.theta[4]=N.minimum(self.theta[4], sz/2.0)
        return(N.transpose(self.theta))
#        '''Calculating CRLB and Log-Likelihood'''
#        self.Div = 0.0
#        for i in range(sz):
#            for j in range(sz):
#                self.PSFx = G.IntGauss1D(i,self.theta[0],PSFsigma)
#                self.PSFy = G.IntGauss1D(j,self.theta[1],PSFsigma)
#                self.model = self.theta[3]+self.theta[2]*self.PSFx*self.PSFy
#                '''Calculating the Derivatives'''
#                self.x_derivative = G.DerivativeIntGauss1D(i,self.theta[0],PSFsigma,self.theta[2],self.PSFy)
#                self.dudt[0] = self.x_derivative[0]
#                self.y_derivative = G.DerivativeIntGauss1D(j,self.theta[1],PSFsigma,self.theta[2],self.PSFx)
#                self.dudt[1] = self.y_derivative[0]
#                self.sigma_derivative = G.DerivativeIntGauss2DSigma(i,j,self.theta[0],self.theta[1],self.theta[4],self.theta[2],self.PSFx,self.PSFy)
#                self.dudt[4] = self.sigma_derivative[0]
#                self.dudt[2] = self.PSFx*self.PSFy
#                self.dudt[3] = 1.0
#                '''Building the Fisher Information Matrix'''
#                self.M+=self.dudt*(N.transpose(self.dudt)/self.model)
#                '''Log-Likelihood'''
#                if (self.model>0):
#                    if (data[i,j]>0):
#                        self.Div+=data[i,j]*N.log(self.model)-self.model-data[i,j]*N.log(data[i,j])+data[i,j]
#                    else:
#                        self.Div-=self.model
#        '''Matrix Inverse'''
#        self.Minv =  N.linalg.inv(self.M)
#        for kk in range(params):        
#            self.CRLB[kk] = self.Minv[kk,kk]
#        '''Log-Likelihood'''
#        self.LL = self.Div
#        return(N.transpose(self.theta),N.transpose(self.CRLB),self.LL)
           
        
        
        
                                