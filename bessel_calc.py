__author__ = 'pablogsal'

import numpy as np
import os
import logging

#Create logger for module
module_logger = logging.getLogger('HADES.bessel_calc')

# Class to obtain values for the Bessel functions.
# When instantiating this class, if no paths are given,
# the class looks in the same directory as the class file is.
#
# A little inception, I know :).

class bessel_integrals(object):

    def __init__(self,gamsp,importfileF=os.path.dirname(os.path.realpath(__file__))+'/integral_data/F_vals.dat',
                 importfileG=os.path.dirname(os.path.realpath(__file__))+'/integral_data/G_vals.dat'):

        #Get class logger

        self.logger = logging.getLogger('HADES.bessel_calc.bessel_integrals')
        self.logger.info('Creating an instance of bessel_integrals')

        self.logger.info('Reading bessel integral data from files: '+importfileF.split('/')[-1]+' , '
                                                                    +importfileG.split('/')[-1]+'.')


        self.f_array=np.loadtxt(importfileF)
        self.g_array=np.loadtxt(importfileG)

        self.xvals=self.f_array.transpose()[0]


        self.f_val=self.f_array.transpose()[1]
        self.g_val=self.g_array.transpose()[1]


        # Now we can generate the integral data

        #Initialize to 0 the values
        self.int_f1=np.zeros(len(self.xvals))
        self.int_f2=np.zeros(len(self.xvals))
        self.int_g1=np.zeros(len(self.xvals))
        self.int_g2=np.zeros(len(self.xvals))


        # First run

        self.int_f1[0] = self.xvals[0] ** ((gamsp-3.0)/2.0) * self.f_val[0] * (self.xvals[1]-self.xvals[0])
        self.int_f2[0] = self.xvals[0] ** ((gamsp-2.0)/2.0) * self.f_val[0] * (self.xvals[1]-self.xvals[0])
        self.int_g1[0] = self.xvals[0] ** ((gamsp-3.0)/2.0) * self.g_val[0] * (self.xvals[1]-self.xvals[0])
        self.int_g2[0] = self.xvals[0] ** ((gamsp-2.0)/2.0) * self.g_val[0] * (self.xvals[1]-self.xvals[0])

        #Middle run

        for j in range(len(self.xvals)-2):
            k=j+1
            self.int_f1[k] = self.xvals[k] ** ((gamsp-3.0)/2.0) * self.f_val[k] * (self.xvals[k+1]-self.xvals[k-1])
            self.int_f2[k] = self.xvals[k] ** ((gamsp-2.0)/2.0) * self.f_val[k] * (self.xvals[k+1]-self.xvals[k-1])
            self.int_g1[k] = self.xvals[k] ** ((gamsp-3.0)/2.0) * self.g_val[k] * (self.xvals[k+1]-self.xvals[k-1])
            self.int_g2[k] = self.xvals[k] ** ((gamsp-2.0)/2.0) * self.g_val[k] * (self.xvals[k+1]-self.xvals[k-1])


         # End run

        k=len(self.xvals)-1
        self.int_f1[k] = self.xvals[k] ** ((gamsp-3.0)/2.0) * self.f_val[k] * (self.xvals[k]-self.xvals[k-1])
        self.int_f2[k] = self.xvals[k] ** ((gamsp-2.0)/2.0) * self.f_val[k] * (self.xvals[k]-self.xvals[k-1])
        self.int_g1[k] = self.xvals[k] ** ((gamsp-3.0)/2.0) * self.g_val[k] * (self.xvals[k]-self.xvals[k-1])
        self.int_g2[k] = self.xvals[k] ** ((gamsp-2.0)/2.0) * self.g_val[k] * (self.xvals[k]-self.xvals[k-1])



        self.logger.info('Data readed correctly from files: '+importfileF.split('/')[-1]+' , '
                                                             +importfileG.split('/')[-1]+'.')






    def F(self,x):
         return np.interp(x, self.xvals, self.f_val)

    def G(self,x):
         return np.interp(x, self.xvals, self.g_val)



