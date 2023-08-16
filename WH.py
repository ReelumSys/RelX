# -*- coding: utf-8 -*-
#import packages
import powerxrd as xrd
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
#import subprocess
import sys
sys.path.append('C:/Users/Vladi/Desktop/RelX/powerxrd')
#C:/Users/Vladi/Desktop/RelX/powerxrd
import csv
from itertools import repeat
import os
import time

from numpy import*
from scipy import*
from matplotlib import pylab as plt
from scipy.optimize import fmin
from scipy.optimize import curve_fit

from powerxrd.main import scherrer as beta
from powerxrd.main import Chart as SchPeak
import contextlib

global ksev1
ksev1 = 'ksev1.csv'

global Ee2
global r

global m
global d

def myfiledel():
    #filename0 = 'myfilesize.txt'
    #if os.path.exists(filename0): os.remove(filename0)
    filename = 'myfile.txt'
    if os.path.exists(filename): os.remove(filename)
    filename2 = 'myfile2.txt'
    if os.path.exists(filename2): os.remove(filename2)

myfiledel()

def test_sch():
    
    #Wurst = pd.read_csv(ksev1, names=['Theta','Int'])  
    #np.savetxt('ksev1.xy', Wurst, fmt='%s', delimiter='\t')
    #data = xrd.Data('ksev1.xy').importfile()
    #chart = xrd.Chart(*data)

    #chart.backsub(tol=1.0,show=True)
    #chart.SchPeak(xrange=[10,50],verbose=True,show=True)
    plt.xlabel('2 $\\theta$')
    plt.title('backsub and Scherrer width calculation')
    #plt.show()

test_sch()




def test_allpeaks():
    Wurst = pd.read_csv(ksev1, names=['Theta','Int'])    
    np.savetxt('ksev1.xy', Wurst, fmt='%s', delimiter='\t')

    data = xrd.Data('ksev1.xy').importfile()
    chart = xrd.Chart(*data)

    chart.backsub(tol=1,show=False)
    #chart.SchPeak
    chart.allpeaks(tols=(0.09,0.8), verbose=False, show=True)
    #Printfunc2 = xrd.Chart.Printfunc(chart)
    #np.savetxt('FWHMfinal.xy', Printfunc2, fmt='%f', delimiter='\t')
    plt.xlabel('2 $\\theta$')
    #plt.suptitle('backsub & Automated Scherrer width calculation of all peaks*')
    #plt.show()



test_allpeaks()

def reel_Williamson_Hall():
    
    global m
    global d
    data = loadtxt ('myfile.txt', usecols=(0))
    data2 = loadtxt ('myfile2.txt', usecols=[0])
    
    print(np.radians)

    x = (data2)
    y = (data)
    wl = 0.15406 # Ang 

    theta  =  np.radians(x/2)

    x = sin(theta)

    y1= np.radians(y)
    
    y = y1*cos(theta) 



    def func(x,m,c):
        return m * x + c
    
    popt,pcov= curve_fit(func,x,y)
    #print(popt)

    m = popt[0]
    c = popt[1]
    k = 0.94
    d = k*wl/c

    y_max = amax(y)
    print('y_max', y_max)



    print('crystallite size = ',d,'nm')
    print('Lattice Strain =    ',m)

    
    #d.to_csv('CrystSize.csv')

    xx = arange(0.0,0.7,0.1)
    yy = m*xx + c

    plt.xlim(0,0.7)
    plt.ylim(0,y_max+0.003)

    plt.title('Size = %.2f nm %s Strain = %.4f'%(d,'', m))


    plt.plot((x),func(x,*popt),'b-')
    plt.plot(xx,yy,'b--')
    #plt.title ('Williamson-Hall Plot',fontsize = 14)
    plt.plot(x,y,'ro')
    np.savetxt('WH-realx.txt', x, fmt='%f', delimiter='')
    np.savetxt('WH-realy.txt', y, fmt='%f', delimiter='')
    datanew = pd.read_csv('WH-realx.txt', names=['sinTheta'])
    datanew2 = pd.read_csv('WH-realy.txt', names=['BetaCosTheta'])
    


    plt.xlabel(r'sin $\theta$',fontsize = 14)
    plt.ylabel(r'$\beta$ cos $\theta$',fontsize = 14)
    plt.grid(True)
    plt.savefig('WH-PLOT.png')
    #plt.show()

    y = y1*cos(theta) 
    x = sin(theta)

    Sum = (y/x)**2
    plt.close()

    return data,data2,m,d,
reel_Williamson_Hall()

def reel_Halder_Wagner():
    global Ee2
    global r
    beta = loadtxt ('myfile.txt', usecols=(0))
    theta = loadtxt ('myfile2.txt', usecols=[0])

    x10 = beta/(tan(theta)*sin(theta))

    y11 = 1000*(((beta/(tan(theta)))**2))

    Klamba = 0.15406 * (4/3)
    EePre = 16 * Klamba

    Ee = np.sqrt(Klamba/(beta*EePre))
    print(Ee)
    #print(Ee)
    Ee2 = Ee.mean()
    Ee2 = Ee2 *100
    #x12 = x10.transpose()
    #print(x12)
    x13 = pd.DataFrame(x10)
    #print(x13)

    #y13 = y11.transpose()
    #print(y13)
    y14 = pd.DataFrame(y11)
    #print(y14)


    frames = [x13,y14]

    result = pd.concat(frames, axis=1)

    #print(result)


    
    def func(x10,r,j):
        return r * x10 + j
    
    popt,pcov= curve_fit(func,x10,y11)

    r = popt[0]/1
    j = popt[1]
    
    xx = arange(0.0,7,0.1)
    yy = r*xx + j



    y_max = amax(y11)

    
    
    plt.xlim(-2,7)
    plt.ylim(-2,y_max+0.003)

    plt.title('Size = %.2f nm %s Strain = %.4f'%(Ee2,'', r))
   
    plt.plot(x13,y14, '.')
    plt.plot(xx,yy,'b--')
    
    plt.ylabel(r'1000*($\beta$ / tan ($\theta$))$^2$',fontsize = 14)
    plt.xlabel(r'$\beta$ (tan ($\theta$) * sin ($\theta$))',fontsize = 14)
    plt.grid(True)
    plt.savefig('HW-PLOT.png')

    #plt.show()
    plt.close()

reel_Halder_Wagner()




#data = xrd.Data('myfile.txt').importfile()
    #df1={'myfile.txt'}
    #df2={'myfile2.txt'}
    #df1 = open('myfile.txt')
    #df2 = open('myfile2.txt') 
    #Trans = np.transpose(df1)
    

    #datacrest = np.array([*df1])
    #dataset = pd.DataFrame({df1})
    #print(dataset)
    #np.savetxt('FWHMdef_search.xy', dataset, fmt='%s', delimiter='')

    #datacrest2 = np.array([*df2])
    #dataset2 = pd.DataFrame({df2})
    #print(dataset2)
    
  
    #np.savetxt('FWHMdef1.xy', dataset2, fmt='%s', delimiter='')
    #np.savetxt('FWHMstep.csv', dataset2, fmt='%s', delimiter='')
    #np.savetxt('FWHMstep2.csv', dataset, fmt='%s', delimiter='')

    #df1 = open('FWHMstep.csv')
    #df2 = open('FWHMstep2.csv')
    #colnames=['Theta']
    #colnames2=['FEWHM']
    #datanew2 = pd.read_csv('FWHMstep2.csv', skiprows=1, names=colnames2)
    #datanew = pd.read_csv('FWHMstep.csv', skiprows=1, names=colnames)


    #reader_file = csv.reader(datanew)
    #value = len(list(datanew))
    #print(datanew)

    #reader_file = csv.reader(datanew2)
    #value = len(list(datanew2))
    #print(datanew2)
    #value2=value/2

    #datanew['FWHM'] = datanew2
    #datanew.to_csv('FWHMStep10.csv')

    #lines = list()
    #remove= [value2]
    
    #datanew = pd.read_csv('FWHMStep10.csv')
    #values = len(list(datanew))
    #print(datanew)

    #colnames=['No', 'Theta', 'FWHM'] 
    #user1 = pd.read_csv('FWHMStep10.csv', header=None)
    #print(user1)
   


    #datanew = datanew.astype(str)
    #datasmo = '.'.join(datanew[i:i+2] for i in range(0, len(datanew), 2))
    #a=datanew
    #b=datanew2
    #np.savetxt('FWHMstep3.csv', datasmo, fmt='%s', delimiter='')

    #datanew[*datanew2] = datanew2
    #datanew.shape[0]
    #datanew.to_csv('FWHMStep4.txt')


    #a = np.array([*df1])
    #b = np.array([*df2])
    #print(a)
    #np.savetxt('FWHMstep3.csv', , fmt='%s', delimiter='')
    #c = np.stack((a,b),axis=0)

    #d = np.transpose(c)
    


    #np.savetxt('FWHMstep111.xy', a, fmt='%s', delimiter='')
    #np.savetxt('FWHMstep222.xy', b, fmt='%s', delimiter='')
    #np.savetxt('FWHMstep333.xy', d, fmt='%s', delimiter='')

    #np.savetxt('FWHMstep.xy', df1, fmt='%s', delimiter='')

    #dataset = pd.DataFrame({'Theta': df1})
    #dataset2 = pd.DataFrame({'FWHM': df2})

    
    
    #df6 = np.column_stack([dataset, dataset2])
    #df6 = np.c_[dataset, dataset2]
    
    #df6 = np.hstack((dataset, dataset2))   #WORKING
    #df6 = np.append(df1, dataset2, axis=1)
    #np.savetxt('FWHMstep4.txt', df6, fmt='%s', delimiter='')
    #print(df6)
    

    #dataset = pd.DataFrame({'Theta': df1})
    #dataset2 = pd.DataFrame({'FWHM': df2})
    #datasetx = np.transpose(dataset)
    #dataset2y = np.transpose(dataset2)
    #print(datasetx)

    

    #df3 = pd.concat([dataset, dataset2], axis=1)
    #np.savetxt('FWHMstep3.txt', df3, fmt='%s')


    
    
    

    #df3 = dataset.assign(e=pd.Series(dataset2).values)
    #df3 = dataset2y.join(datasetx, lsuffix='Theta', rsuffix='FWHM')
    #Crackset = pd.DataFrame({'Theta': dataset, 'Int': dataset})
   

    #df3 = pd.concat([df1, df2],axis=1, ignore_index=True, sort=False)    
    #fmt=['%s','%s']
    #np.savetxt('FWHMdef2.txt', df3, fmt=fmt, delimiter=',')
    #df3.to_csv('FWHMdef10.txt', header=None, index=None, sep='\t', mode='a')


    #df4 = np.transpose(df3)
    #np.savetxt('FWHMdef3.txt', dataset, fmt='%s')











    #df_merged = pd.concat([df1, df2], ignore_index=True, sort=False)
    #dataset = pd.DataFrame({'FWHM': df1[:, 0]})
    #dataset2 = pd.DataFrame({'Theta': df2[:, 0]})
    
    #df_merged = pd.concat([df1, df2], ignore_index=True)
   
    
    #data = xrd.Data('FWHMdef10.txt').importfile()
    #data = loadtxt ('FWHMStep10.csv', usecols=(0,1))
    #datanew.to_csv('FWHMStep19.csv')
    #dfX = datanew[['Theta']]
    #dfY = datanew[['FWHM']]
    #dfX.Theta = datanew.Theta.astype(float64)
    #dfY.FWHM = datanew.FWHM.astype(float64)
    #np.savetxt('FWHMnew.txt', dataset, fmt='%')

    #print(dfX)
    #print(dfY)
