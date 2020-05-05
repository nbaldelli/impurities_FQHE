# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:36:56 2020

@author: niccobal
"""

import numpy as np
from numba import njit,jit
import pfaffian as pfunc

#%%
@njit
def wavefunction(r,filling,Nb,Ntotal):
  gausfac=(-np.sum(r[0]**2+r[1]**2)/4) #overall gaussian factor
  #Laughlin Liquid
  laughsea=0
  for i in range(Nb,Ntotal):
    for j in range (i+1,Ntotal):
      laughsea=laughsea+(1./filling)*np.log(r[0,i]+1j*r[1,i]-r[0,j]-1j*r[1,j])
  # vandermonde determinant / Laughlin of filling 1/2  for the impurities (fermionic / bosonic)
  # if power =1 we are constructing a vandermonde
  # if power =2 we are building a Laughlin nu=1/2
  lauimpu=0
  for i in range(0,Nb-1):
    for j in range(i+1,Nb):
      lauimpu=lauimpu+np.log(r[0,i]+1j*r[1,i]-r[0,j]-1j*r[1,j])
  # maj-min interactions
  inter=0
  for i in range(0,Nb):
    for j in range(Nb,Ntotal):
      inter=inter+np.log(r[0,i]+1j*r[1,i]-r[0,j]-1j*r[1,j])
  wf=gausfac+laughsea+lauimpu+inter
  return wf
#%%
@njit
def matrixpf(r):
    L=len(r[0,:])
    pf=np.zeros((L,L),dtype=np.complex128)
    for i in range(L):
        for j in range(L):
            if (i!=j):
                pf[i,j]=1./(r[0,i]+1j*r[1,i]-r[0,j]-1j*r[1,j])
    return pf

#%%
#@njit
def wavefunctionMR(r,filling,Nb,Ntotal):
  gausfac=(-np.sum(r[0]**2+r[1]**2)/4) #overall gaussian factor
  #Laughlin Liquid
  laughsea=0
  for i in range(Nb,Ntotal):
    for j in range (i+1,Ntotal):
      laughsea=laughsea+(1./filling)*np.log(r[0,i]+1j*r[1,i]-r[0,j]-1j*r[1,j])
  # vandermonde determinant / Laughlin of filling 1/2  for the impurities (fermionic / bosonic)
  # if power =1 we are constructing a vandermonde
  # if power =2 we are building a Laughlin nu=1/2
  lauimpu=0
  for i in range(0,Nb-1):
    for j in range(i+1,Nb):
      lauimpu=lauimpu+np.log(r[0,i]+1j*r[1,i]-r[0,j]-1j*r[1,j])
  # maj-min interactions
  inter=0
  for i in range(0,Nb):
    for j in range(Nb,Ntotal):
      inter=inter+np.log(r[0,i]+1j*r[1,i]-r[0,j]-1j*r[1,j])
  #pfaffian factor typical of moore read
  pfaf=np.log(pfunc.pfaffian(matrixpf(r),method='P')) 
  wf=pfaf+gausfac+laughsea+lauimpu+inter
  return wf

#%%
@njit
def deriva(r,coord,index,filling,Nb,Ntotal): #first derivative operator
  hh=0.0000001
  rplus=r.copy()
  rminus=r.copy()
  rplus[coord,index]=rplus[coord,index]+hh
  rminus[coord,index]=rminus[coord,index]-hh
  deri=(wavefunction(rplus,filling,Nb,Ntotal)-wavefunction(rminus,filling,Nb,Ntotal))/(2*hh)
  return deri
#%%
@njit
def LzTotal(r,filling,Nb,Ntotal):
  lz=0
  for i in range(0,Ntotal):
    lz+=deriva(r,0,i,filling,Nb,Ntotal)*(-r[1,i])+deriva(r,1,i,filling,Nb,Ntotal)*r[0,i]
  return -1j*lz

@njit
def LzImpu(r,filling,Nb,Ntotal):
  lz=0
  for i in range(0,Nb):
    lz=lz+deriva(r,0,i,filling,Nb,Ntotal)*(-r[1,i])+deriva(r,1,i,filling,Nb,Ntotal)*r[0,i]
  return -1j*lz

@njit
def Lz2Impu(r,Nb):
  lz2=0
  for i in range(0,Nb):
    for j in range(0,Nb):
      lz2=lz2+r[0,i]*r[0,j]*deriva2(r,i,1,j,1)+r[1,i]*r[1,j]*deriva2(r,i,0,j,0)-r[0,i]*r[1,j]*deriva2(r,i,1,j,0)-r[1,i]*r[0,j]*deriva2(r,i,0,j,1)
      if i==j: 
        lz2=lz2-r[0,i]*deriva(r,i,0)-r[1,i]*deriva(r,i,1)
  return -lz2
#%%
@jit
def MCrun(step,nblk,nmov,nterm,filling,Nb,Ntotal):
    lz=0
    lzimpu=0
    #lzimpu2=0
    r=10*step*(np.random.rand(2,Ntotal)-0.5) #initialize particles positions
    wfold=wavefunction(r,filling,Nb,Ntotal)
    #starts integration
    count=0
    for iblk in range(1,nblk+1): #number of runs
      for jmov in range(1,nmov+1): #moves per run
        for kterm in range(1,nterm+1): #thermalization steps
          randomito=np.random.rand() 
          dr=step*(np.random.rand(2,Ntotal)-0.5) #random update step
          rnew=r+dr
          wf=wavefunction(rnew,filling,Nb,Ntotal)
          difwa=np.exp(2*(np.real(wf)-np.real(wfold))) #metropolis update rule
          if difwa>randomito:
            count=count+1
            wfold=wf
            r=rnew.copy()
        dlz=LzTotal(r,filling,Nb,Ntotal) #Total Lz, to cross check  
        dlzimpu=LzImpu(r,filling,Nb,Ntotal)  #<Lb>
        # Update of observables
        lz=lz+dlz #angular momentum
        lzimpu=lzimpu+dlzimpu 
        #lzimpu2=lzimpu2+dlzimpu**2
        #  lz2impu=lz2impu+dlzimpu2
      #Prints partial result for everyblock
      print(iblk,lz/(iblk*nmov),lzimpu/(iblk*nmov))
    
    lzfinal=lz/(nblk*nmov)
    lzimpufinal=lzimpu/(nblk*nmov)
    #lzimpu2final=lzimpu2/(nblk*nmov)
    #lz2impufinal=lz2impu/nblk/nmov
    #errorlzimpu=np.sqrt((np.real(lzimpu2final)-np.real(lzimpufinal)**2)/(nblk-1))
    errorlzimpu=0
    lzimpu2final=0
    return lzfinal,lzimpufinal,lzimpu2final,errorlzimpu,count

#%%
#monte carlo parameters
upd=0.9 #metropolis step
bks=200 #number of montecarlo runs
mvs=200 #moves per run
therm=10 #termalization steps

#wavefunction parameters
fil=1/3 #laughlin filling
Na=8 #number of majority particles
Nimp=2 #number of impurities 
power=1 #parameter to express impurities state
Ntot=Na+Nimp #total number of particles

#%%
lzout,lzimp,lzimp2,errorlz,cc=MCrun(upd,bks,mvs,therm,fil,Nimp,Ntot)

print("Results:")
print("Filling:",fil)
print("Particles in sea:",Na)
print("Impurities:",Nimp)
print("total number of sample points:",therm*bks*mvs)
print("Lztotal=",lzout)
print("<Lb>=",lzimp,"error:",errorlz)
print("Acceptance=",cc/(bks*mvs*therm))
#print("Delta Lb=",np.sqrt(np.real(lz2impufinal)-np.real(lzimpufinal)**2))


'''
#
#@njit
#def deriva2(wf,r,coord1,index1,coord2,index2): #icoi=index, i=coord
#  hh=0.0000001
#  rpirpj=r.copy()
#  rpirmj=r.copy()
#  rmirpj=r.copy()
#  rmirmj=r.copy()
#  rpirpj[icoi,i]=r[icoi,i]+hh
#  rpirpj[icoj,j]=rpirpj[icoj,j]+hh
#
#  rpirmj[icoi,i]=r[icoi,i]+hh
#  rpirmj[icoj,j]=rpirmj[icoj,j]-hh
#
#  rmirpj[icoi,i]=r[icoi,i]-hh
#  rmirpj[icoj,j]=rmirpj[icoj,j]+hh
#
#  rmirmj[icoi,i]=r[icoi,i]-hh
#  rmirmj[icoj,j]=rmirmj[icoj,j]-hh
#
#  wfpipj=wavefunction(rpirpj)
#  wfpimj=wavefunction(rpirmj)
#  wfmipj=wavefunction(rmirpj)
#  wfmimj=wavefunction(rmirmj)
#
#  d2=(wfpipj+wfmimj-wfpimj-wfmipj)/(2*hh)**2+deriva(r,i,icoi)*deriva(r,j,icoj)
#  return d2
'''