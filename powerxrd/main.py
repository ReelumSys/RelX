import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import pandas
import pandas as pd
import csv

from numpy import*
from scipy import*
from matplotlib import pylab as plt
from scipy.optimize import fmin
from scipy.optimize import curve_fit

from lmfit import CompositeModel, Model
from lmfit.lineshapes import gaussian, step


def braggs(twotheta,lmda=1.54):
    '''interplanar spacing "d_hkl" from Braggs law'''

    'lambda in Angstroms'
    twothet_rad=twotheta*np.pi/180
    
#    dhkl = lmda /(2*np.sin(twothet_rad/2))
    
    if twotheta.any() < 5:
        L =len(twotheta)
        dhkl = np.zeros(L)
        dhkl[0] = 'inf'
        
        k =1
        while k < L:
            dhkl[k] = lmda /(2*np.sin(twothet_rad[k]/2))
            k+=1
    else:
        dhkl = lmda /(2*np.sin(twothet_rad/2))
    
    dhkl = np.round(dhkl,2)
    return dhkl


def braggs_s(twotheta,lmda=1.54):
    'lambda in Angstroms'
    twothet_rad=twotheta*np.pi/180
    
    
    if twotheta < 5:
        dhkl = 'inf'
    else:
        dhkl = lmda /(2*np.sin(twothet_rad/2))
        dhkl = np.round(dhkl,2)
    
    return dhkl
    #print(dhlk)


def scherrer(K,lmda,beta,theta):
    '''Scherrer equation'''
    # print('Scherrer Width == K*lmda / (FWHM*cos(theta))')
    return K*lmda / (beta*np.cos(theta))    #tau


def funcgauss(x,y0,a,mean,sigma):
    '''Gaussian equation'''
    return y0+(a/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mean)**2/(2*sigma*sigma))


def Rietveld_func(x, HKL, atomic_positions, s, m_K, TwoTheta_M, K, N_j, f_j, M_j, phi, Theta_k, P_K, A, y_bi ):
        """
        Calculate the Rietveld equation for crystal structure analysis.

        Parameters
        ----------
        x : array(float)
            Array with x-data 2-theta values.
        HKL : array((num_indices,3)) 
            Miller indices matrix (more than one hkl).
        atomic_positions : array((3, num_atoms)) 
            xj, yj and zj atomic positions.
        s : float
            Scale factor (constant).
        TwoTheta_M : float
            The Bragg angle of the reflection from a monochromator (it is a constant for a fixed wavelength).         # Lorentz-Polarization Factor (L_pK) sub-group
        K : float
            Fractional polarization of the beam (Pecharsky).
        N_j : array((num_atoms))
            'Nj is the site occupancy divided by the site multiplicity'.         # Structure Factor (F_K) sub-group
        f_j : array((num_atoms))
            'fj is the atomic form factor'.
        M_j : array((num_atoms))
            'M j contains the thermal contributions (atomic displacements)'.
        phi : str
            Reflection profile function (e.g. 'pseudo-voight', 'voight', 'gauss', etc.).
        Theta_k : float
            '2#\#theta_k: the calculated position of the Bragg peak corrected for the zero-point shift of the counter (Rietveld 1969)' # to fit.
        P_K : float
            Preferred orientation i.e. it is a multiplier, which accounts for possible deviations from a complete randomness in the distribution of grain orientations (Pecharsky 2005). # to fit.
        A : float
            Absorption factor (formula).
        y_bi : float
            Background.
        """
        
        #h,k,l =                [1,1,0,1,1,0,3,2,1,2,2,1],[0,1,1,0,0,3,0,2,2,1,2,1],[1,0,2,2,4,2,2,0,3,3,1,4],      

        #HKL = pd.read_csv('HKL.csv', names=['H', 'K', 'L'], index_col=0)
        #dfHKL = [HKL]
        #print(dfHKL)
        
        #dfHKL = pd.read_csv('HKL.csv', names=['H', 'K', 'L'], index_col=False)

        
        #print(dfhkl)
        #df1HKL = dfHKL(np.float64)
        #dfHKL = np.asarray(dfHKL, dtype = 'int')
            

        #print(dfHKL)
        #dfHKL = pd.read_excel("HKL.txt", dtype = {(np.float)}) 
        
        dfHKL = np.loadtxt("HKL.txt")
        dfHKL = np.array(dfHKL)
        np.savetxt('HKL1.txt', dfHKL, fmt='%i', delimiter=' ')

       
        print(dfHKL)
        
        HKL =                  dfHKL        
        """


        
        HKL =                  [[1,0,1],      
                               [1,1,0],         
                               [0,1,2],     
                               [1,0,2],
                               [1,0,4],
                               [0,3,2],
                               [3,0,2],
                               [2,2,0],
                               [1,2,3],
                               [2,1,3], 
                               [2,2,1], 
                               [1,1,4]]
        """


        atomic_positions = [[0.48764,0.00000,0.33330],[0.43868,0.31187,0.14752]]                       
        Mj = [[0.5],[0.1]]             

        def LorentzPol_Factor(Theta, TwoTheta_M = 1,K=1 ):
            'Lorentz-Polarization factor (this is complex)'

            # CTHM = coefficient for monochromator polarization 

            
            CTHM = np.cos(TwoTheta_M)**2
            #L_pK = ( 1 - K +  (K*CTHM*np.cos(2*Theta)**2) ) \
            #            / ( ( 2 * (np.sin(Theta))**2 ) * np.cos(Theta) )
            Teta = dot(Theta,2)
                        
            #CTHM = np.cos(TwoTheta_M)**2
            L_pK = ( 1 - K +  (K*CTHM*np.cos(Teta)**2) ) \
                        / ( ( 2 * (np.sin(Theta))**2 ) * np.cos(Theta) )  

            return L_pK

        def Structure_Factor_K(Theta, Miller_indices_K,atomic_positions,N_j,f_j):
            'Structure Factor'
            imag_i = 1j
            u_s = 1
            lmbda = 1
           
           

            h,k,l = Miller_indices_K
            M_j = 8 * (np.pi**2) * (u_s**2) * np.sin(Theta)**2 / (lmbda**2)

            F_K = []
            for a in atomic_positions: 
                x,y,z  = a
                F_K.append( N_j * f_j * np.exp ( 2 * np.pi * imag_i ) * (h*x + k*y + l*z)  * np.exp(1) - M_j )
            
            return F_K
        
        sum_component = []
        for HKL_K in HKL:
            L_pK = LorentzPol_Factor(x,TwoTheta_M,K)
            F_K = Structure_Factor_K(x, HKL_K,atomic_positions,N_j,f_j)
            sum_component.extend( m_K * L_pK *  np.abs(F_K)**2  * phi * (x - Theta_k) * P_K * A  + y_bi )

        return s * sum_component
            

class Funk:
    def __init__(self):
        self.cnt=0
        
    def __call__(self, arg):
        self.cnt+=1

class Data:
    def __init__(self,file):
        '''
        Data structure.

        Parameters
        ----------
        file : str
            file name and/or path for XRD file in .xy format
        '''
        self.file = file

    
    def importfile(self):

        if self.file.split(".")[1]=='xy':
            df = pandas.read_csv(self.file, sep='\t', header=None)   #'https://www.statology.org/pandas-read-text-file/'

        if self.file.split(".")[1]=='csv':
            df = pandas.read_csv(self.file, header=None)   

        x,y = np.array(df).T
        return x,y 


class Rietveld:
    def __init__(self, x_exp=[],y_exp=[]):
        '''
        Rietveld structure. Loader of Rietveld equation for refinement.

        Parameters
        ----------
        x_exp : list(float) / np.array(float)
            x-data theta values (experimental)
        y_exp : list(float) / np.array(float)
            y-data              (experimental)
        HKL : np.array(int)
            array containing all h,k,l Miller indices.
        atoms : np.array(float)
            atomic positions xj yj zj 
        model : object
            lmfit.Model object for Rietveld function
        pars : object
            lmfit.Model.Parameter objects for Rietveld function   (all initially set to value of 1 as default)      
        params : list(str)
            list of Rietveld function parameters 
        fixed : list(str)
            list of Rietveld function parameters to fix in Rietveld refinement (default: only 's' is fixed)
        '''
        
        dfHKL = pd.read_csv('HKL.csv', names=['H', 'K', 'L'], index_col=False)
        HKL =                  dfHKL
        """
                               [[1,0,1],      
                               [1,1,0],         
                               [0,1,2],     
                               [1,0,2],
                               [1,0,4],
                               [0,3,2],
                               [3,0,2],
                               [2,2,0],
                               [1,2,3],
                               [2,1,3], 
                               [2,2,1], 
                               [1,1,4]]
        """

        atompos = [[0.48764,0.00000,0.33330],[0.43868,0.31187,0.14752]]                       
        Mj = [[0.5],[0.1]]  


        self.x_exp = x_exp
        self.y_exp = y_exp
        self.HKL =  HKL
        self.atoms = atompos
        self.model = Model(Rietveld_func)
        self.pars = self.model.make_params()
        self.params = self.model.param_names
        for i in self.params:
            self.pars[i].value = 1


        self.fixed  = ['s']


    def refine(self):
        '''
        Performs Rietveld refinement on the experimental data.

        This function uses the fixed parameters specified by the user to perform a Rietveld refinement on the experimental data. 
        It then generates a report of the fit results and plots the data with the initial and best fits. Data is then saved in a format to be loaded to the Chart class for 
        additional plot processing. 

        Example Usage:
        To refine the data from 'my_data.xy' file:

        .. code-block:: python

            import powerxrd as xrd

            x, y = xrd.Data('my_data.xy').importfile()      # Import data from file
            model = xrd.Rietveld(x, y)                      # Create Rietveld model
            .
            .
            .            
            model.refine()                                  # Perform Rietveld refinement
        '''


        # params to fix
        # params to fix
        for i in self.fixed:
            
            dfHKL = pandas.read_csv('HKL.csv', names=['H', 'K', 'L'], index_col=False)
            #dfHKL['index1'] = dfHKL.index
            #st.dataframe(dfHKL)
        
            global HKL
            HKL =              dfHKL
                                

            """
                               [[1,0,1],      
                               [1,1,0],         
                               [0,1,2],     
                               [1,0,2],
                               [1,0,4],
                               [0,3,2],
                               [3,0,2],
                               [2,2,0],
                               [1,2,3],
                               [2,1,3], 
                               [2,2,1], 
                               [1,1,4]]
            """
            self.pars[i].vary = False

        # fit this model to data array y
        result = self.model.fit(self.y_exp, params=self.pars, x=self.x_exp)

        
        print(result.fit_report())
        np.savetxt

        #print(C) 
        PreFit = self.x_exp
        PreFit2 = array([PreFit])
        #print(PreFit)
        
        #A=np.stack([PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit, PreFit])
        np.savetxt('RietveldFit2.txt', PreFit, fmt='%f', delimiter=',')
        df2 = pandas.read_csv('RietveldFit2.txt', header=None)
        df2 = df2.transpose()
        
        

        np.savetxt('RietveldFit.txt', result.init_fit, fmt='%f', delimiter=',')
        df = pandas.read_csv('RietveldFit.txt', header=None)
        
        dfmean = df.mean()
        np.savetxt('RietveldFitMean.txt', dfmean, fmt='%f', delimiter=',')
        dfmean = pandas.read_csv('RietveldFitMean.txt', header=None)
        #dfmean = dfmean.transpose()
        #print(dfmean)


        np.savetxt('RietveldFit3.txt', result.best_fit, fmt='%f', delimiter=',')
        df2 = pandas.read_csv('RietveldFit3.txt', header=None)
        #print(df2)

        dfmean2 = df2.mean()
        
        np.savetxt('RietveldFitMean2.txt', dfmean2, fmt='%f', delimiter=',')
        dfmean2 = pandas.read_csv('RietveldFitMean2.txt', header=None)
        #dfmean2 = dfmean.transpose()
        dfmean2.rename(columns={df.columns[0]: 'Int'},inplace=True)
        #dfmean2.transpose()
        dfmean2.rename(columns={df.columns[0]: 'Int'},inplace=True)
        #print(dfmean2)
       

        np.savetxt('RietveldFitMean3.txt', dfmean2, fmt='%f', delimiter=' ')



        fig, axes = plt.subplots(1, 2, figsize=(12.8, 10))
        
        dfmean2num = dfmean2.to_numpy()
        #print(dfmean2num)

        selfxtrans = self.x_exp.transpose()
        print(selfxtrans)

        filename = 'RietveldFitMean.txt'
        with open(filename) as f:
            lines = [line.rstrip('\n') for line in f]
            #print(lines)

        filename2 = 'RietveldFitMean3.txt'
        with open(filename2) as f:
            lines2 = [line.rstrip('\n') for line in f]
            
        lines = np.array(lines)
        lines2 = np.array(lines2)
   
     

        my_int_list = [float(i) for i in lines]
        my_int_list2 = [float(i) for i in lines2]

       
        #a = ''.join(str(my_int_list).split(','))
        #b = ''.join(str(my_int_list2).split(','))

        list_1 = selfxtrans.tolist()
        

        #print(list_1)
        #print(my_int_list)

        axes[1].plot(self.x_exp, self.y_exp, 'bo', label='bo', )
        axes[0].plot(self.x_exp, self.y_exp, 'bo', label='bo', )

        #print(self.x_exp)
        #axes[0].plot(list_1, my_int_list, 'k', label='initial fit')
        axes[0].plot(list_1, my_int_list2, 'r', label='best fit')
        axes[0].legend()
        plt.xlabel(r'$2Theta$',fontsize = 14)
        plt.ylabel(r'$Int$',fontsize = 14)
        #plt.show()
        plt.savefig('RietveldRef.png')
        #plt.show()
        # <end examples/doc_model_composite.py>
        

class Chart:

    def __init__(self,x,y):
        '''
        Chart structure. Constructs x-y XRD data to manipulate and analyze. 

        Parameters
        ----------
        x : np.array(float)
            array with x-data 2-theta values
        y : np.array(float)
            array with y-data peak intensity values
        K : float
            dimensionless shape factor for Scherrer equation (default 0.9)
        lambdaKa : float
            X-ray wavelength of alpha radiation
        lambdaKi : float
            X-ray wavelength of "i" radiation (beta, gamma, other)
        '''
        self.x          = x          # x values
        self.y          = y          # y values
        self.K          = 0.9       
        self.lambdaKa   = 0.15406
        self.lambdaKi   = 0.139

    def local_max(self,xrange=[12,13]):
        '''Maximum finder in specified xrange

        Parameters
        ----------
        xrange_Ka : [](float)
            range of x to find globalmax
        '''

        i_l = self.x.searchsorted(xrange[0], 'left')
        i_r = self.x.searchsorted(xrange[1], 'right')

        'segments of x-y data within specified xrange'
        xseg = self.x[i_l:i_r]
        yseg = self.y[i_l:i_r]
        
        'find maximum y value within specified range and corresponding x loc.'
        imax = np.argmax(yseg)
        max_x = xseg[imax]
        max_y = yseg[imax]

        print('local_max -- max x: {} max y: {}'.format(max_x,max_y))
        return max_x, max_y

    def emission_lines(self, xrange_Ka=[10,20], show = True):
        '''Emission lines arising from different types of radiation i.e. K_beta radiation
        wavelength of K_beta == 0.139 nm
        
        Parameters
        ----------
        show: bool
            show plot of XRD chart
        xrange_Ka : [](float)
            range of x-axis (2-theta) for K_alpha radiation
        '''
        twothet_Ka_deg, int_Ka = Chart(self.x, self.y).local_max(xrange=xrange_Ka)
        twothet_Ka=twothet_Ka_deg*np.pi/180

        twothet_Ki = 2*np.arcsin((self.lambdaKi/self.lambdaKa)*np.sin(twothet_Ka/2))
        twothet_Ki_deg = twothet_Ki*180/np.pi

        # return twothet_Ka_deg, int_Ka, twothet_Ki_deg

        if show:
            plt.vlines(twothet_Ka_deg,0,int_Ka, colors='k', linestyles='solid', \
                    label=r'K$\alpha$; $\theta$ = {} '.format(round(twothet_Ka_deg,2)))
            plt.vlines((twothet_Ka_deg+twothet_Ki_deg)/2,0,int_Ka, colors='k', linestyles='--', label='')
            plt.vlines(twothet_Ki_deg,0,int_Ka, colors='r', linestyles='solid',\
                    label=r'K$\beta$; $\theta$ = {} '.format(round(twothet_Ki_deg,2)))
        else:

            return twothet_Ki_deg


    def gaussfit(self, verbose=True):
        '''Fit of a Gaussian curve ("bell curve") to raw x-y data'''
        meanest = self.x[list(self.y).index(max(self.y))]
        sigest = meanest - min(self.x)
        popt, pcov = optimize.curve_fit(funcgauss,self.x,self.y,p0 = [min(self.y),max(self.y),meanest,sigest])
        
        if verbose:
            print('\n-Gaussian fit results-')
            print('y-shift {}\namplitude {}\nmean {}\nsigma {}'.format(*popt))
            print('covariance matrix \n{}'.format(pcov))
        return popt

    

        

    def SchPeak(self,xrange=[12,13],verbose=True, show=True):

        
        '''Scherrer width calculation for peak within a specified range
        
        Parameters
        ----------
        xrange : [](float)
            range of x-axis (2-theta) where peak to be calculated is found
        show: bool
            show plot of XRD chart
        '''

        # print('\nSchPeak: Scherrer width calc. for peak in range of [{},{}]'.format(*xrange))

        'xseg and yseg:x and y segments of data in selected xrange'
        xseg,yseg = [],[]
        for n, j in zip(self.x,self.y):
            if n >= xrange[0] and n <= xrange[1]:
                xseg.append(n)
                yseg.append(j) 

        
        y0,a,mean,sigma = Chart(xseg,yseg).gaussfit(verbose)
        ysegfit = funcgauss(np.array(xseg),y0,a,mean,sigma)

        'FULL WIDTH AT HALF MAXIMUM'
        global FWHM_deg
        FWHM_deg = sigma*2*np.sqrt(2*np.log(2))
        
        FWHM = FWHM_deg*np.pi/180
        #np.savetxt('FWHM.xy', FWHM, fmt='%f', delimiter='\t')
        #print('\nFWHM == sigma*2*sqrt(2*ln(2)): {} degrees'.format(FWHM_deg))

        HWMIN = sigma*np.sqrt(2*np.log((50)))
        # print('\nHalf-width Minimum (HWMIN) (1/50 max) == sigma*sqrt(2*ln(50)): {} degrees'.\
        #     format(HWMIN))




        'scherrer width peak calculations'
        max_x = xseg[list(yseg).index(max(yseg))]

        theta=max_x/2
        theta=theta*np.pi/180

        Dict = [[max_x],[FWHM_deg]]
        data = np.transpose(Dict)
        


        def myfun(s,i=[0]):
            print(s)
            i[0]+=1
            input("Press Enter to continue...")
            return i[0]#np.savetxt('FWHM.txt', data, fmt='%f', delimiter=' ')
        
        



        # print('K (shape factor): {}\nK-alpha: {} nm \nmax 2-theta: {} degrees'.\
        #     format(self.K,self.lambdaKa,max_x))
        
        Sch=scherrer(self.K,self.lambdaKa,FWHM,theta)
        X,Y = xseg,ysegfit

        # print('\nSCHERRER WIDTH: {} nm'.format(Sch))

        if verbose:
            print('\nSchPeak: Scherrer width calc. for peak in range of [{},{}]'.\
                                    format(*xrange))
            print('\nFWHM == sigma*2*sqrt(2*ln(2)): {} degrees'.\
                                    format(FWHM_deg))
            print('K (shape factor): {}\nK-alpha: {} nm \nmax 2-theta: {} degrees'.\
                                    format(self.K,self.lambdaKa,max_x))
            print('\nSCHERRER WIDTH: {} nm'.\
                                    format(Sch))

        
        if show:
            plt.plot(X,Y,'c--')             # gauss fit 
            plt.plot(xseg,yseg,color='m')   # fitted segment

        left = mean - HWMIN
        right = mean + HWMIN 

        #filename = 'myfile.txt'
        #if os.path.exists(filename): os.remove(filename)
        #filename2 = 'myfile2.txt'
        #if os.path.exists(filename2): os.remove(filename2)

        filename = 'myfile.txt'
        filename2 = 'myfile2.txt'

        with open("myfile.txt", "a") as file_prime:
            for num in range(0,20):
                prime = True
                for i in range(1,num):
                    
                        prime = False
                if prime:
                    print(num)
                    file_prime.write(str(FWHM_deg)+ '\n')

        with open("myfile.txt", "a") as file_prime:
            for num in range(0,20):
                prime = True
                for i in range(1,num):
                    
                        prime = False
                if prime:
                    print(num)
                    file_prime.write(str(FWHM_deg)+ '\n')

        with open("myfile2.txt", "a") as file_prime:
            for num in range(0,20):
                prime = True
                for i in range(1,num):
                    
                        prime = False
                if prime:
                    print(num)
                    file_prime.write(str(max_x)+ '\n')
        

            with open("myfile2.txt", "a") as file_prime:
                for num in range(0,20):
                    prime = True
                    for i in range(1,num):
                        
                            prime = False
                    if prime:
                        print(num)
                        file_prime.write(str(max_x)+ '\n')
               
         
        #with open('myfile.txt') as sorting
        #np.sort(max_x, axis=None)
        

        print('\nFWHM == ln(2)): {} degrees'.\
                                    format(FWHM_deg))

        # return Sch,X,Y
        return max_x, max(yseg), Sch, left,right
        
        

    def allpeaks_recur(self,left=0, right=1, tols_=(2e5,0.8),schpeaks=[],verbose = False, show = True):
        '''recursion component function for main allpeaks function below'''
        # print('left right',left,right)
        max_x, max_y = Chart(self.x, self.y).local_max(xrange=[left,right])
        maxpeak_height, peaktrough_d = tols_ 
        peak_max = max_y     

        if peak_max > maxpeak_height:
            xrange = [ max_x - peaktrough_d, max_x + peaktrough_d ]
            Sch_x, Sch_y, Sch, l,r = Chart(self.x, self.y).\
                        SchPeak(xrange,verbose,show)
            schpeaks.append([Sch_x,Sch_y,Sch])
            
          
           


            Chart(self.x, self.y).allpeaks_recur(left, l,tols_,schpeaks,verbose,show)
            Chart(self.x, self.y).allpeaks_recur(r, right,tols_,schpeaks,verbose,show)

          


    def allpeaks(self, tols=(0.2,0.8), verbose=False, show = True):
        '''Driver code for allpeaks recursion : Automated Scherrer width calculation of all peaks
        
        Parameters
        ----------
        tols : (float, float)
            tolerances for recursion 
            tol[0]: Minimum peak height to be calculated as a percent of the chart's global maximum (default=0.2 [20% of global maximum])
            tol[1]: Average distance from peak (top) to trough (bottom) of all peak (default=0.8)
        show: bool
            show plot of XRD chart
        '''
        print('\n-------------------------------------------\nALLPEAKS: '+\
            'Automated Scherrer width calculations with a recursive search of local maxima\n')

        #init xrange [left, right]
        left = min(self.x)
        right = max(self.x)
        schpeaks_ = []

        max_x, max_y = Chart(self.x, self.y).local_max(xrange=[left,right])
        print('\n')
        maxpeak_height = max_y*tols[0]
        peaktrough_d = tols[1]


        tols_ = (maxpeak_height, peaktrough_d)
        Chart(self.x, self.y).allpeaks_recur(left, right, tols_, schpeaks_,verbose,show)


        



        print('\nSUMMARY (.csv format):')
        print('2-theta / deg, \t Intensity, \t Sch width / nm')

        for i in schpeaks_:
            print('2-theta: {} deg - Sch width: {} nm'.format(*i))


        sortidcs = np.argsort(np.array(schpeaks_).T[0])
        # print(sortidcs)
        for i in sortidcs:
            print('{}, \t  {}, \t  {} '.format(*schpeaks_[i]))
            np.savetxt('FWHMFirst.txt', schpeaks_, fmt='%f', delimiter=' ')
          


        


    def mav(self,n=1,show=False):
        '''Function for an "n" point moving average. '''
        L=int(len(self.x)//n)
        newy=np.zeros(L)
        for i in range(L):
            k=0
            while k < n:
                newy[i] += self.y[(i*n)+k]
                k += 1
    #           print(i)
            newy[i]=newy[i]/n

        newx=np.zeros(L)
        for i in range(L):
            newx[i] = self.x[i*n]

        'update'
        self.x, self.y = newx,newy

        if show:
            plt.plot(self.x,self.y)

        return newx,newy


    def XRD_int_ratio(self,xR1=[8.88,9.6],xR2=[10.81,11.52]):
        '''Calculate relative peak intensity (i.e. comparing one peak to another)'''
        # 'XRD b/t two intensities ratio'
        return Chart(self.x, self.y).local_max(xR2)[1]/Chart(self.x, self.y).local_max(xR1)[1]



    def backsub(self,tol=1,show=False):
        '''
        Background subtraction operation.This function is a running conditional statement 
        which evaluates whether a small increase in the x-direction will increase the magnitude of the 
        y variable beyond a certain tolerance.

        Parameters
        ----------
        tol : float, optional
            Tolerance for background subtraction.
            The function evaluates whether a small increase in the x-direction will increase the magnitude
            of the y variable beyond a certain tolerance value. This tolerance value is defined as a
            percentage of the y variable at each point. Peaks above this tolerance are considered and
            their background is removed.
            Default value is 1.
        show : bool, optional
            Whether to show the plot of the XRD chart.
            Default value is False.
        '''
        
        L=len(self.y)
        lmda = int(0.50*L/(self.x[0]-self.x[L-1]))         #   'approx. # points for half width of peaks'

        backsub_y=np.zeros(L)
        for i in range(L):
            if self.y[(i+lmda)%L] > tol*self.y[i]:          #tolerance 'tol'
                backsub_y[(i+lmda)%L] = self.y[(i+lmda)%L] - self.y[i]
            else:
                if self.y[(i+lmda)%L] < self.y[i]:
                    backsub_y[(i+lmda)%L] = 0
        
        'update'
        self.x = self.x
        self.y = backsub_y

        b = np.array(self.x)
        a = np.array(backsub_y)
        c = np.append(a, b)
        #c =  np.r_[ a, b[1] ] 
        
        d = np.c_[b,a]

        with open('test1.txt','x') as f:
            

            
            np.savetxt('test1.xy', d, fmt='%f', delimiter='\t')
            f.close()
            os.remove('test1.txt')

        #f=open('test1.txt','x')   

       
        #np.savetxt('test1.txt', a, fmt='%f')

        #b = np.array(self.x)
        #np.append('test1.txt', b)

        #f=open('test1.txt','a')                



        #with open('ye.txt', 'a') as f:
        #    f.write('\n'.join(ye))

        if show:
            plt.plot(self.x,self.y)
        


        #b = np.array(ex)
        #a = np.array(backsub_y)
        #with open('test1.txt','x') as f:
        #    np.savetxt(f, a, fmt='%f', newline=", ")
        #    np.savetxt(f, b, fmt='%f', newline=", ")
        #f.close()

        return self.x,backsub_y



def WHPeak(self,xrange=[12,13],verbose=True, show=True):

        
        '''Scherrer width calculation for peak within a specified range
        
        Parameters
        ----------
        xrange : [](float)
            range of x-axis (2-theta) where peak to be calculated is found
        show: bool
            show plot of XRD chart
        '''

        # print('\nSchPeak: Scherrer width calc. for peak in range of [{},{}]'.format(*xrange))

        'xseg and yseg:x and y segments of data in selected xrange'
        xseg,yseg = [],[]
        for n, j in zip(self.x,self.y):
            if n >= xrange[0] and n <= xrange[1]:
                xseg.append(n)
                yseg.append(j) 

        
        y0,a,mean,sigma = Chart(xseg,yseg).gaussfit(verbose)
        ysegfit = funcgauss(np.array(xseg),y0,a,mean,sigma)

        'FULL WIDTH AT HALF MAXIMUM'
        global FWHM_deg
        FWHM_deg = sigma*2*np.sqrt(2*np.log(2))
        
        FWHM = FWHM_deg*np.pi/180

        #np.savetxt('FWHM.xy', FWHM, fmt='%f', delimiter='\t')
        #print('\nFWHM == sigma*2*sqrt(2*ln(2)): {} degrees'.format(FWHM_deg))

        HWMIN = sigma*np.sqrt(2*np.log((50)))
        # print('\nHalf-width Minimum (HWMIN) (1/50 max) == sigma*sqrt(2*ln(50)): {} degrees'.\
        #     format(HWMIN))




        'scherrer width peak calculations'
        max_x = xseg[list(yseg).index(max(yseg))]

        theta=max_x/2
        theta=theta*np.pi/180

        Dict = [[max_x],[FWHM_deg]]
        data = np.transpose(Dict)
        np.savetxt('Williamson4.xy', data, fmt='%f', delimiter='\t') 


        def myfun(s,i=[0]):
            print(s)
            i[0]+=1
            input("Press Enter to continue...")
            return i[0]#np.savetxt('FWHM.txt', data, fmt='%f', delimiter=' ')
        
        



        # print('K (shape factor): {}\nK-alpha: {} nm \nmax 2-theta: {} degrees'.\
        #     format(self.K,self.lambdaKa,max_x))
        
        Sch=scherrer(self.K,self.lambdaKa,FWHM,theta)
        X,Y = xseg,ysegfit

        # print('\nSCHERRER WIDTH: {} nm'.format(Sch))

        if verbose:
            print('\nSchPeak: Scherrer width calc. for peak in range of [{},{}]'.\
                                    format(*xrange))
            print('\nFWHM == sigma*2*sqrt(2*ln(2)): {} degrees'.\
                                    format(FWHM_deg))
            print('K (shape factor): {}\nK-alpha: {} nm \nmax 2-theta: {} degrees'.\
                                    format(self.K,self.lambdaKa,max_x))
            print('\nSCHERRER WIDTH: {} nm'.\
                                    format(Sch))

        
        if show:
            plt.plot(X,Y,'c--')             # gauss fit 
            plt.plot(xseg,yseg,color='m')   # fitted segment

        left = mean - HWMIN
        right = mean + HWMIN 


        
        FWHMsuper = FWHM_deg
        

        print('\nFWHM == ln(2)): {} degrees'.\
                                    format(FWHM_deg))

        # return Sch,X,Y
        return FWHM_deg

def reel_delete():
    
        with open('myfile.txt','w') as f:

            
            print('myfile.txt')
            f.close()
            os.remove('myfile.txt')


reel_delete()

        

        
def WH():
        
         

        
        

        x = data[:,0]
        y = data[:,1]
        wl = 0.15406 # Ang 

        theta  =  radians(x/2)

        x = sin(theta)

        y1= radians(y)

        y = y1*cos(theta) 


        def func(x,m,c):
            return m * x + c

        popt,pcov= curve_fit(func,x,y)
        #print popt
        m = popt[0]
        c = popt[1]
        k = 0.94
        d = k*wl/c

        y_max = amax(y)
        print('y_max', y_max)



        print('crystallite size = ',d,'nm')
        print('Lattice Strain =    ',m)

        xx = arange(0.0,0.7,0.1)
        yy = m*xx + c

        plt.xlim(0,0.7)
        plt.ylim(0,y_max+0.003)

        plt.title('Size = %.2f nm %s Strain = %.4f'%(d,'', m))


        plt.plot((x),func(x,*popt),'b-')
        plt.plot(xx,yy,'b--')
        #plt.title ('William son-Hall Plot',fontsize = 14)
        plt.plot(x,y,'ro')
        plt.xlabel(r'sin ($\theta$)',fontsize = 14)
        plt.ylabel(r'$\beta$ cos($\theta$)',fontsize = 14)
        plt.grid(True)
        plt.savefig('WH-PLOT.png')
        plt.show()



        return WH()