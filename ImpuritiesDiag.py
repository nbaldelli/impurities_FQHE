# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:02:01 2020

@author: niccobal
"""

import numpy as np
from scipy import sparse as spr
from scipy.sparse import linalg
import itertools
from numba import njit,jit

# calculate logarithm of factorial
def logfac(num):
    if (num==0):
        return 0
    if (num==1):
        return 0
    if (num>1):
        return logfac(num-1)+np.log(num)

# LLL orbital wave functions
def LLL(x,y,m):
    return np.sqrt(1/(2*np.pi))*2**(-m/2)*np.exp(-logfac(m)/2)*np.exp(-(x**2+y**2)/4)*(x+1j*y)**m

#calculate pseudopotential matrix elements 
#(here only for two pseudopotentials vo and v1, at filling 1/5 we need also v2 and v3, different code)

def vhal(m1,m2,m3,m4,v0,v1):
    #define the haldane pseudopotentials
    l=2
    v=np.array([0,0])
    v[0]=v0
    v[1]=v1
    out=0
    if (m1+m2-m3-m4==0):
        for i in range(l):
            ll=i;
            n=m1+m2-ll;
            if (n>=0):           
                # C_m1,m2^n,l       
                B=0
                for k in range(n+1):
                    if (ll>=(m1-k))and(m1-k>=0):        
                        A = logfac(n)+logfac(ll)-logfac(k)-logfac(n-k)-logfac(m1-k)-logfac(ll-m1+k)
                        B = B+np.exp(A)*(-1)**(ll-m1+k);
                C1=(logfac(m1)+logfac(m2)-logfac(n)-logfac(ll))/2-(n+ll)*(np.log(2)/2)+np.log(np.abs(B))+1j*np.pi*np.heaviside(B,0)
                C1 =np.exp(C1)
                # C_m3,m4^n,l
                B=0
                for k in range(n+1):
                    if  (ll>=(m4-k))and(m4-k>=0):
                        A = logfac(n)+logfac(ll)-logfac(k)-logfac(n-k)-logfac(m4-k)-logfac(ll-m4+k)
                        B = B+np.exp(A)*(-1)**(ll-m4+k)
                C2 = (logfac(m4)+logfac(m3)-logfac(n)-logfac(ll))/2-(n+ll)*(np.log(2)/2)+np.log(np.abs(B))+1j*np.pi*np.heaviside(B,0)
                C2 = np.exp(C2);
            # V
                out=out+C1*C2*v[i];
    return round(np.real(np.double(out)),11)

L=21   
Ne=4
Nh=1

gee=1; geh=1; we=1; wh=1.001

lmax=3*(Ne-1)+7; #truncation of single particle angumom WHY
Lmin=np.int(max(Ne*(Ne-1)/2,L-Nh*(2*lmax-Nh+1)/2)); # smallest possible total angular momentum of electrons
Lmax=np.int(L-Nh*(Nh-1)/2); # largest allowed total angular momentum of electrons

d=Lmax-Lmin+1 # number of angular momentum sectors
De=np.zeros(d,dtype=np.int64); # dimension of electronic Hilbert space sectors
Dh=np.zeros(d,dtype=np.int64); # dimension of the impurity Hilbert space
De=np.zeros(d,dtype=np.int64)

# special case: there are no impurities 
if (Nh==0):
    Lmin=L;
    Lmax=L;
    d=1;
    Dh=np.array([1]); #is this wrong?
    De=np.array([0]);

#CONFIGURATIONS FOR MAJORITY PARTICLES
larr=np.arange(0,lmax+1)
allchoices=np.array(list(itertools.combinations(larr,Ne)))

ebase=np.zeros((len(allchoices),d)) #python cannot create dynamical arrays
elist=np.zeros((sum(2**(allchoices[-1,:]))+1,2)) 


def configmaj(allchoices,Lmin,d):
    for i in range(len(allchoices[:,0])):
        Le=sum(allchoices[i,:]); # angular momentum of a given configuration i
        sector=Le-Lmin; # to which sector does it correspond
        if (sector>=0)and(sector<d):
            De[sector]=De[sector]+1 # then Hilbert space dimension increases by 1
            num=0
            for j in range(Ne):
                num=num+2.**allchoices[i,j]
            ebase[De[sector]-1,sector]=int(num) #the first index goes as long as I have states to put!!
            elist[int(num),:]=[sector,De[sector]] #understand how I could do this
    return ebase,elist

#ebase=[[0]*d]*len(allchoices)
#elist=[[0,0]]*sum(2**(allchoices[-1,:]))
#
#@njit
#def configmaj2(allchoices,Lmin,d):
#    De=[0]*d
#    for i in range(len(allchoices[:,0])):
#        Le=sum(allchoices[i,:]); # angular momentum of a given configuration i
#        sector=Le-Lmin; # to which sector does it correspond
#        if (sector>=0)and(sector<d):
#            De[sector]=De[sector]+1 # then Hilbert space dimension increases by 1
#            num=0
#            num=sum(2.**allchoices[i,:])
#            ebase[De[sector]-1][sector]=int(num) #the first index goes as long as I have states to put!!
#            elist[int(num)][:]=[sector,De[sector]] #understand how I could do this
#    return ebase,elist

# CONFIGURATIONS FOR MINORITY PARTICLES
# (Essentially the same as for majority particles, as we take both of them
# to be fermionic)

if (Nh>0):  
    allchoices=np.array(list(itertools.combinations(larr,Nh)))
    hbase=np.zeros((len(allchoices),d)) #python cannot create dynamical arrays
    hlist=np.zeros((sum(2**(allchoices[-1,:]))+1,2)) 
    for i in range(len(allchoices)):
        Lh=sum(allchoices[i,:])
        sector=L-Lh-Lmin; #why this? becaus it is exactly Le-Lmin
        if (sector>=0)and(sector<d):
            Dh[sector]=Dh[sector]+1
            num=0
            for j in range(Nh):
                num=num+2.**allchoices[i,j]
            hbase[Dh[sector]-1,sector]=int(num)
            hlist[int(num),:]=[sector,Dh[sector]]
            
D=De*Dh; # This is a vector with Hilbert space dimension in each sector, being the product De(sector) X Dh(sector)
Dtot=sum(D) # summing over all sector: total Hilbert space dimension
print(Dtot)
print(d)

# this is a vector with powers of 2, we will use it to relate
# configurations to binary numbers
pots=2.**(lmax-np.arange(0,lmax+1))

#% that turns out to be faster than calling the function whenever we need it
vmaj=np.zeros((lmax+1,lmax+1,lmax+1,lmax+1))
vmin=np.zeros((lmax+1,lmax+1,lmax+1,lmax+1))
for k1 in range(lmax+1):
    for k2 in range(lmax+1):
        for k3 in range(lmax+1):
            for k4 in range(lmax+1):
                vmaj[k1,k2,k3,k4]=vhal(k1,k2,k3,k4,0,1)
                vmin[k1,k2,k3,k4]=vhal(k1,k2,k3,k4,1,0)

# matrix elements for e-e interactions
iee=[]; jee=[]; vee=[];
# matrix elements for e-h interactions
ieh=[]; jeh=[]; veh=[];
# matrix element traps
htrap=[]
etrap=[]

for sec in range(d): # loop over all sectors
    print(sec)
    for ih in range(Dh[sec]): # loop over all impurity configuration in the sector
        
        if (Nh>0): # if we have no impurities, the impurity has not been defined, therefore the "if" 
            hbin=np.fromstring(np.binary_repr(int(hbase[ih,sec])), dtype='S1').astype(int)
            hbin=np.pad(hbin,(lmax-len(hbin)+1,0),'constant',constant_values=(0,0))
            
    #        % this transforms the number of the state (saved in hbase) into the binary number which characterizes the occupation 
    #        % It is a binary with lmax+1 digits (for smaller numbers, the first digits are zero)
    #        % the command -'0' transform the number into a vector, 
    #        % i.e 101-'0'  -->  (1,0,1)
            
            hoccs=lmax-np.flip(np.argwhere(hbin)[:,0]) 
    #        % e.g. (1,0,0,1,1) -> (0,1,4) [we have chosen the smaller levels to
    #        % be on the right, so level on the right has angular momentum 0,
    #        % increasing to lmax=4, being the level on the left]
            
        for ie in range(De[sec]):  # loop over all majority configurations
            ebin=np.fromstring(np.binary_repr(int(ebase[ie,sec])), dtype='S1').astype(int)
            ebin=np.pad(ebin,(lmax-len(ebin)+1,0),'constant',constant_values=(0,0))
            eoccs=lmax-np.flip(np.argwhere(ebin)[:,0]);  # occupied angular momentum levels
            eemps=np.setdiff1d(larr,eoccs);  # empty angular momentum levels

            inn=sum(D[:sec])+(ih)*De[sec]+ie; #penso che sto numero sia abbastanza arbitrario
#            % here we define a unique number of the state in the matrix, 
#            % i.e. in a combined basis for majority and impurity.
#            % obviously, this number is between 1 and D, and it's uniquely
#            % given by ie, ih, sec
                         
#            % diagonal parts of the matrix from the harmonic trap - depends
#            % only on the total angular momentum of majority and impurity
#            % particles
            etrap.append(sec+Lmin)
            htrap.append(L-Lmin-sec)
            
#            % interactions between majority particles (block diagonal within a sector)
            choices=np.array(list(itertools.combinations(eoccs,2))); # choose two occupied levels to be annihilated          
            for j in range(len(choices)):
                l1=choices[j,0];       l2=choices[j,1];   # note that l2>l1 CHIEDI A TOBI MA PENSO CI SIA ERRORE
                eemps2=np.sort(np.append(eemps,choices[j,:])) # empty levels after annihilation
                choices2=np.array(list(itertools.combinations(eemps2,2))); # choose two levels to be created
                for jj in range(len(choices2)):
                    ll1=choices2[jj,0]; ll2=choices2[jj,1];  #note that ll2>ll1 is it true for python?
                    if (ll1+ll2-l1-l2==0): # angular momentum is conserved     
                        #define the new configuration, newebin, after scattering
                        newebin=np.zeros(len(ebin))
                        newebin[:]=ebin[:] 
                        newebin[lmax-l1]=0; newebin[lmax-l2]=0; newebin[lmax-ll1]=1; newebin[lmax-ll2]=1;
                        # determine sign of matrix element (assuming order ll1,ll2,l1,l2)
                        aa=len(np.where(eoccs<l1)[0]) # how many occupied levels must right annihilator pass                
                        bb=len(np.where(eoccs<l2)[0]) # how many occupied levels must left annihilator pass
                        eoccs2=np.setdiff1d(eoccs,choices[j,:])
                        cc=len(np.where(eoccs2<ll1)[0]) # how many occupied levels must right creator pass
                        dd=len(np.where(eoccs2<ll2)[0]) # how many occupied levels must left creator pass
                        sig=(-1)**(aa+bb+cc+dd);
    #                    % interaction matrix element, from the array vmaj,
    #                    % times the sign sig
    #                    % all four combinations: +1234, -1243, -2134, +2143
                        v2=sig*(vmaj[ll1,ll2,l1,l2]-vmaj[ll1,ll2,l2,l1]-vmaj[ll2,ll1,l1,l2]+vmaj[ll2,ll1,l2,l1]);
    #                    % what is the position number of this newebin
    #                    % configuration ?? Look up in elist:
                        ieout=int(elist[int(np.dot(newebin,pots)),1]);
    #                    % This is the number of the new state in the matrix
    #                    % (i.e. combined basis of majority and impurity)
                        outt=sum(D[:sec])+(ih)*De[sec]+ieout-1 #CAPIRE QUESTO
    #                    % we save the matrix in a sparse format, we need three lists
                        iee.append(inn) # this is the index for the incoming state
                        jee.append(outt) # this is the inces for the outcoming state
                        vee.append(v2); # this is the interaction strength of the scatter process connecting the states
            # Interaction between impurities and majority particles
            if (Nh>0):
                for jj1 in range(len(eoccs)):
                    l1=eoccs[jj1]
                    for jj2 in range(len(hoccs)):
                        l2=hoccs[jj2]
                        for l4 in range(lmax+1):
                           newhbin=np.zeros(len(hbin))
                           newebin=np.zeros(len(ebin))               
                           newebin[:]=ebin[:];     newhbin[:]=hbin[:];       newebin[lmax-l1]=0;   newhbin[lmax-l2]=0;
                           l3=l1+l2-l4; # conservation of angular momentum ISN'T THIS REDUNDANT
                           if (l3>-1)and(l3<lmax+1)and(newebin[lmax-l4]==0)and(newhbin[lmax-l3]==0):    
                               newebin[lmax-l4]=1;    newhbin[lmax-l3]=1; 
                               if ((np.dot(lmax-larr,newebin))<=Lmax): 
                                   sig=(-1)**(jj1+jj2) # annihilators have to pass over jjx-1 occupied levels
                                   occs2=np.setdiff1d(eoccs,l1)
                                   cc=len(np.where(occs2<l4)[0]) # how many occupied levels must electron creator pass
                                   occs2=np.setdiff1d(hoccs,l2)
                                   dd=len(np.where(occs2<l3)[0]) # how many occupied levels must impurity creator pass
                                   sig=sig*(-1)**(cc+dd)
                                   v2=sig*vmin[l4,l3,l2,l1] # vhal(l4,l3,l2,l1,1,0);
                                   ieout=int(elist[int(np.dot(newebin,pots)),1])
                                   ihout=int(hlist[int(np.dot(newhbin,pots)),1])
                                   secout=int(elist[int(np.dot(newebin,pots)),0])
                                   
                                   outt=sum(D[:(secout)])+(ihout-1)*De[secout]+ieout-1;
                                   ieh.append(inn) # this is the index for the incoming state
                                   jeh.append(outt) # this is the inces for the outcoming state
                                   veh.append(v2) # this is the interaction strength of the scatter process connecting the states

ieh=np.array(ieh); jeh=np.array(jeh); veh=np.array(veh);
iee=np.array(iee); jee=np.array(jee); vee=np.array(vee);

# Build the sparse matrices from the lists we have produced
vehmat=spr.coo_matrix((veh,(ieh.astype(np.int),jeh.astype(np.int))),shape=(Dtot,Dtot))
veemat=spr.coo_matrix((vee,(iee.astype(np.int),jee.astype(np.int))),shape=(Dtot,Dtot))
etrapmat=spr.coo_matrix((etrap,(np.arange(0,Dtot),np.arange(0,Dtot))))
htrapmat=spr.coo_matrix((htrap,(np.arange(0,Dtot),np.arange(0,Dtot))),shape=(Dtot,Dtot))

hmat=gee*veemat+geh*vehmat+we*etrapmat+wh*htrapmat

del( vehmat, veemat, etrapmat, htrapmat, ieh, jeh, veh, iee, jee, vee)

#diagonalize the matrix
nol=20
vals1,vecs=linalg.eigsh(hmat,k=nol,which='SM') 

#% evaluate the total angular momentum of the majority particles within
#% each of the eigenstates
i=0;Lel=np.zeros(nol)
for sec in range(d):
    for e in range(nol):
        Lel[e]=Lel[e]+(Lmin+sec)*np.dot(vecs[i:i+D[sec],e],vecs[i:i+D[sec],e]);
    i=i+D[sec];
    
print(Lel)
print(L-Lel) 



