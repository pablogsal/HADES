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

    def __init__(self,importfileF=os.path.dirname(os.path.realpath(__file__))+'/integral_data/F_vals.dat',
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

        self.logger.info('Data readed correctly from files: '+importfileF.split('/')[-1]+' , '
                                                             +importfileG.split('/')[-1]+'.')

    def F(self,x):
         return np.interp(x, self.xvals, self.f_val)

    def G(self,x):
         return np.interp(x, self.xvals, self.g_val)



