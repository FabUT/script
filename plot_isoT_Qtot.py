# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:24:36 2020

@author: ftholin
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def scinot(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    return a + "$\\times 10^{" + str(b) + "}$"


# 1.0000 * 10^4

# Data for plotting

#---Lecture :
folder='EOS-Data/'

file_Q='Al.Rho-Qtot.ist'

if('Rho-P' in file):
    RhoP=True
    print("PLOT Rho vs P...")
else:
    RhoP=False

###============================ PREMIERE LECTURE =============================
f=open(folder+file_Q, 'r')
chars = set('#&')
Nr=0
NT=1
found=False

for line in f:
    
    if any((c in chars) for c in line):
        if('&' in line):
            found=True
            NT=NT+1
    else: 
        if not found:
            Nr=Nr+1

print ('Nr=', Nr)
print ('NT=', NT)
 
f.close()

###============================ DEUXIEME LECTURE =============================
f=open(folder+file_Q, 'r')
chars = set('#&')
r=np.zeros(Nr)
T=np.zeros(Nr)
Q=np.zeros((Nr,NT))

found=False
j=0
i=0

for line in f:
    
    if any((c in chars) for c in line):
        if('&' in line):
            found=True
            i=0
            j=j+1
        if('T =' in line):
            line = line.strip()
            columns = line.split()   
            T[j]=float(columns[3])                 
    else: 
        line = line.strip()
        columns = line.split()   
        Q[i,j]=float(columns[1])  
        if not found:
            r[i]=1.0*float(columns[0])
        i=i+1  
      
rmin=np.min(r)
rmax=np.max(r)
Tmin=np.min(T)
Tmax=np.max(T)
Qmin=np.min(Q)
Qmax=np.max(Q)

print('rmin=', rmin)
print('rmax=', rmax)
print('Tmin=', Tmin)
print('Tmax=', Tmax)
print('Qmin=', Qmin)
print('Qmax=', Qmax)

print('Q=', Q[Nr-2,NT-2])
print('Q=', Q[Nr-1,NT-1])

T=T*11604.0

#for i in range(NT):
#    print("T[", i, "]=", T[i])

#===================== PLOTS ========================= 


#=========================== Q vs rho ======================================

fig, ax = plt.subplots(figsize=(6, 3.75))#plt.xlim(0, 0.03)

#=== FONTS:
plt.rc('font',   size      = 12) # controls default text sizes
plt.rc('axes',   titlesize = 12) # fontsize of the axes title
plt.rc('axes',   labelsize = 12) # fontsize of the x and y labels
plt.rc('xtick',  labelsize = 12) # fontsize of the xtick labels
plt.rc('ytick',  labelsize = 12) # fontsize of the ytick labels
plt.rc('legend', fontsize  = 12) # legend fontsize
plt.rc('figure', titlesize = 12) # fontsize of the figure title

plt.xscale("log")
#plt.yscale("log")

#=== labels:
ax.set_xlabel('$\\rho$ [ g cm$^{-3}$ ]')
ax.set_ylabel('$Q$', color='black')

#=== range:
#ax.set_xlim(auto=False, xmin=0, xmax=100)

ax.set_xlim(auto=False, xmin=1e-3, xmax=1e1)
ax.set_ylim(auto=False, ymin=0, ymax=Qmax)

#=== plots:

colors=["black","red","blue","green","orange","grey","purple","brown","gold","cadetblue"]
dash=[[1,0],[1,1],[2,1],[2,2],[3,1],[3,2],[3,3],[4,1],[4,2],[4,3]]

for j in range(NT):
    line0,=ax.plot(r, Q[:,j], dashes=[1,0],  lw=0.5, c="lightgray", alpha=1,zorder=1, label='')

#=== Major tics:
#ax.tick_params(axis='y', labelcolor='black')

#=== Minor ticks
#ax.minorticks_on()
#ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

j=9
line1,=ax.plot(r, Q[:,j], dashes=dash[0],  lw=1, c=colors[0], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=19
line2,=ax.plot(r, Q[:,j], dashes=dash[1],  lw=1, c=colors[1], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=23
line3,=ax.plot(r, Q[:,j], dashes=dash[2],  lw=1, c=colors[2], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=25
line4,=ax.plot(r, Q[:,j], dashes=dash[3],  lw=1, c=colors[3], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=28
line5,=ax.plot(r, Q[:,j], dashes=dash[4],  lw=1, c=colors[4], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=33
line6,=ax.plot(r, Q[:,j], dashes=dash[5],  lw=1, c=colors[5], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=66
line7,=ax.plot(r, Q[:,j], dashes=dash[6],  lw=1, c=colors[6], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
#label='$T={:.1e} [eV]$'.format(T[j])




#=== legende
plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7],loc='center left', frameon=False,
           bbox_to_anchor=(1.0, 0.5), shadow=False, ncol=1)


#=== Grille:
#ax.grid()

#=== sortie
fig.savefig("isoT_Qrho.png",bbox_inches="tight",dpi=200)
plt.show()


#=========================== Q vs V ======================================
V=np.zeros(Nr)
for i in range(Nr):
    V[i]=1.0/r[i]
#V=r 

#np.reciprocal(r)

fig, ax = plt.subplots(figsize=(6, 3.75))#plt.xlim(0, 0.03)

#=== FONTS:
plt.rc('font',   size      = 12) # controls default text sizes
plt.rc('axes',   titlesize = 12) # fontsize of the axes title
plt.rc('axes',   labelsize = 12) # fontsize of the x and y labels
plt.rc('xtick',  labelsize = 12) # fontsize of the xtick labels
plt.rc('ytick',  labelsize = 12) # fontsize of the ytick labels
plt.rc('legend', fontsize  = 12) # legend fontsize
plt.rc('figure', titlesize = 12) # fontsize of the figure title

plt.xscale("log")
#plt.yscale("log")

#=== labels:
ax.set_xlabel('$V$ [ cm$^{3}$ g$^{-1}$ ]')
ax.set_ylabel('$Q$', color='black')

#=== range:
#ax.set_xlim(auto=False, xmin=0, xmax=100)

ax.set_xlim(auto=False, xmin=0.1, xmax=1e3)
ax.set_ylim(auto=False, ymin=0, ymax=Qmax)

#=== plots:

colors=["black","red","blue","green","orange","grey","purple","brown","gold","cadetblue"]
dash=[[1,0],[1,1],[2,1],[2,2],[3,1],[3,2],[3,3],[4,1],[4,2],[4,3]]

for j in range(NT):
    line0,=ax.plot(V, Q[:,j], dashes=[1,0],  lw=0.5, c="lightgray", alpha=1,zorder=1, label='')

#=== Major tics:
#ax.tick_params(axis='y', labelcolor='black')

#=== Minor ticks
#ax.minorticks_on()
#ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

j=9
line1,=ax.plot(V, Q[:,j], dashes=dash[0],  lw=1, c=colors[0], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=19
line2,=ax.plot(V, Q[:,j], dashes=dash[1],  lw=1, c=colors[1], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=23
line3,=ax.plot(V, Q[:,j], dashes=dash[2],  lw=1, c=colors[2], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=25
line4,=ax.plot(V, Q[:,j], dashes=dash[3],  lw=1, c=colors[3], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=28
line5,=ax.plot(V, Q[:,j], dashes=dash[4],  lw=1, c=colors[4], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=33
line6,=ax.plot(V, Q[:,j], dashes=dash[5],  lw=1, c=colors[5], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
j=66
line7,=ax.plot(V, Q[:,j], dashes=dash[6],  lw=1, c=colors[6], alpha=1,zorder=1, label='$T=$'+scinot(T[j],1)+' K')
#label='$T={:.1e} [eV]$'.format(T[j])




#=== legende
plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7],loc='center left', frameon=False,
           bbox_to_anchor=(1.0, 0.5), shadow=False, ncol=1)


#=== Grille:
#ax.grid()

#=== sortie
fig.savefig("isoT_QV.png",bbox_inches="tight",dpi=200)
plt.show()


#print(scinot(10000, sig_fig=1))


"""


t = np.arange(0.0, 2.0, 0.01)

s = 1 + np.sin(2 * np.pi * t)

A=100.0
r=0.005
beta=2.0

x = np.arange(0.0, 0.03, 0.0001)
QEM =A*np.exp(-(x/r)**beta)

fig, ax = plt.subplots()
ax.plot(x, QEM)

plt.xlim(0, 0.03)
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()

"""
# SUBROUTINE ResolutionRK4(x,y,dx,NbEquation,Derivee)
# C x : abscisse
# C y() : A l'appel ce tableau contient les valeurs y initiales. Au retour, il contient les
# C valeurs calculees
# C dx : pas de calcul
# C NbEquation : nombre d'equations dans le systeme dif.
# C Derivee : nom de la fonction decrivant le systeme dif.
# IMPLICIT NONE
# EXTERNAL Derivee ! fonction de calcul de la derivee
# C
# C Typage de l'interface de ResolutionRK4
# C
# INTEGER NbEquation
# REAL x, y(NbEquation), dx
# C
# C Declaration des variables locales
# REAL pred1(NbEquation),pred2(NbEquation),pred3(NbEquation),1 pred4(NbEquation), ytemp(NbEquation), halfdx
# INTEGER i
# halfdx = dx/2
# C Premiere prediction
# CALL Derivee(x,y,pred1,NbEquation)
# DO i=1, NbEquation
# ytemp(i) = y(i) + pred1(i)*halfdx
# ENDDO
# C Seconde prediction
# CALL Derivee(x+halfdx,ytemp,pred2,NbEquation)
# DO i=1, NbEquation
# ytemp(i) = y(i) + pred2(i)*halfdx
# ENDDO
# C Troisieme prediction
# CALL Derivee(x+halfdx,ytemp,pred3,NbEquation)
# DO i=1, NbEquation
# ytemp(i) = y(i) + pred3(i)*dx
# ENDDO
# C Quatrieme prediction
# CALL Derivee(x+dx,ytemp,pred4,NbEquation)
# C Estimation de y pour le pas suivant
# DO i=1, NbEquation
# y(i)=y(i)+(dx/6.0)*(pred1(i)+2.0*pred2(i)+2.0*pred3(i) + pred4(i))
# ENDDO
# END 
