__author__ = 'pablogsal'

import numpy as np
import ConfigParser
import os, sys
import tabulate
from tools import stype



#Class for output colors

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'





class imput_parameters(object):

    def __init__(self,importfile):

        #Some units

        c=2.9979e10

        #We have to read the information from the INI file to configure the parameters.

        Config = ConfigParser.ConfigParser()

        # Read from the supported filename, if the name is empty string then
        # read from the ACTUAL directory of the file to aboid path-related problems.

        if importfile =="" :
            Config.read(os.path.dirname(os.path.realpath(__file__))+'/conf.INI')
        else:
            Config.read(importfile)

        #Read the variables from the config file.

        stype("Reading input values....")

        self.hubble= float( Config.get('STANDARD_KEYS',"Hubble") )
        self.rate_ene=float( Config.get('STANDARD_KEYS',"Rate_ene") )
        self.gam_sp=float( Config.get('STANDARD_KEYS',"Gam_sp") )
        self.beam_radius=float( Config.get('STANDARD_KEYS',"Beam_radius") )
        self.beam_radius_cells=int( Config.get('STANDARD_KEYS',"Beam_radius_cells") )
        self.tracer_limit=float( Config.get('STANDARD_KEYS',"Tracer_limit") )
        self.redshift=float( Config.get('STANDARD_KEYS',"Redshift") )
        self.viewing_angle=float( Config.get('STANDARD_KEYS',"Viewing_angle") )
        self.RMHD_data_file=Config.get('STANDARD_KEYS',"RMHD_data_file")
        self.x_var_dim=int( Config.get('STANDARD_KEYS',"x_var_dim") )
        self.y_var_dim=int( Config.get('STANDARD_KEYS',"y_var_dim") )
        self.x_grid_dim=int( Config.get('STANDARD_KEYS',"x_grid_dim") )
        self.y_grid_dim=int( Config.get('STANDARD_KEYS',"y_grid_dim") )
        self.non_thermal_fraction=float( Config.get('STANDARD_KEYS',"Non_termal_fraction") )
        self.non_thermal_density=float(Config.get('STANDARD_KEYS',"Non_termal_density") )
        self.external_density=float(Config.get('STANDARD_KEYS',"External_density") )
        self.observation_freq=float(Config.get('STANDARD_KEYS',"Observation_freq") )*1.e9 # Pass to Hz the values
        self.result_file=Config.get('STANDARD_KEYS',"Result_file")
        self.min_energy=1.0e-04 # Standar min energy of non-thermal electrons
        self.q0= 0.5 # Cosmological parameter
        self.h0=self.hubble*1.e5/3.0856e18/1.e6
        self.luminosity_distance = c/self.h0/self.q0/self.q0*(self.q0*self.redshift+(self.q0-1.e0) *(np.sqrt(
            1.e0+2.e0*self.q0*self.redshift)-1))


        #Now we check that the variables for correct values

        stype("Checking input values....")

        if self.gam_sp < 1 or self.gam_sp == 2:
            print(bcolors.FAIL +"ERROR - The spectral index must be >1 and =! 2."+ bcolors.ENDC)
            sys.exit("We have found some errors. \nTry looking at the previous lines to find some clue. :("+'\n')

        if self.x_grid_dim > self.x_var_dim:
            print(bcolors.FAIL +"ERROR - Variable x_grid_dim must be smaller than x_var_dim"+ bcolors.ENDC)
            sys.exit("We have found some errors. \nTry looking at the previous lines to find some clue. :("+'\n')

        if self.y_grid_dim > self.y_var_dim:
            print(bcolors.FAIL +"ERROR - Variable x_grid_dim must be smaller than x_var_dim"+ bcolors.ENDC)
            sys.exit("We have found some errors. \nTry looking at the previous lines to find some clue. :("+'\n')

        if self.viewing_angle < 0 or self.viewing_angle > 90:
            print(bcolors.FAIL +"ERROR - Viewing angle must be between 0 and 90 degs."+ bcolors.ENDC)
            sys.exit("We have found some errors. \nTry looking at the previous lines to find some clue. :("+'\n')


        stype(bcolors.OKGREEN +"All input values are correct."+ bcolors.ENDC)



    # This function calculates the energy factor of the non thermal electrons
    # and write the value in self.non_thermal_density. The actual formulas for these
    # calculatons can be found in NEED SOURCE!!!  Jose Luis Thesis ?

    #Notice that if the non_thermal_density =! 0 then the self.min_energy must be calculated!

    #INPUT:
    #
    # EPS value at (0,0) from the RMHD class.

    def calculate_energy_factor(self,eps):
        # Now, we can calculate the energy factor for the non-thermal electrons
        c=2.9979e10 #speed of light
        e_mass=9.1094e-28 #rest-mass of the electron

        if self.non_thermal_density == 0:
            self.non_thermal_density=self.min_energy/eps/e_mass/c/c*(self.gam_sp-1e0)/(self.gam_sp-2e0)*(
                1e0-self.rate_ene**(
                2e0-self.gam_sp))/(1e0-self.rate_ene**(1e0-self.gam_sp))

            if  self.non_thermal_density < 1 :

                print(bcolors.FAIL +"ERROR - Energy factor for non-thermal e- is <1 "+ bcolors.ENDC)
                print(bcolors.FAIL +'The actual value is = '+str(self.non_thermal_density)+"."+'\n'+ bcolors.ENDC)
                sys.exit("We have found some errors. \nTry looking at the previous lines to find some clue. :("+'\n')

        else:
            self.min_energy=self.non_thermal_density*eps*e_mass*c*c*(self.gam_sp-2.e0)/(self.gam_sp-1.e0)*(1.e0-rate_ene**(
                1.e0-self.gam_sp))/(1.e0-rate_ene**(2.e0-self.gam_sp))




    #Function to write the values to the screen
    def print_values(self):

            headers=["Hubble constant","Energy ratio of e-","Expectral index","Radius of the beam","Cells of the "
                                                                                                   "beam",
                     "Tracer lim . value","Redshift","Viewing angle","RMHD file","Variable dimensions",
                     "Grid dimensions","Non thermal e- fraction","Non thermal e- density","External density",
                     "Observation frequency","Result file","Min_energy","Luminosity distance"]
            printdata=[self.hubble,self.rate_ene,self.gam_sp,self.beam_radius,self.beam_radius_cells,
                       self.tracer_limit,self.redshift,self.viewing_angle, self.RMHD_data_file,[self.x_var_dim,
                       self.y_var_dim],[self.x_grid_dim,self.y_grid_dim],self.non_thermal_fraction,
                       self.non_thermal_density,
                       self.external_density, self.observation_freq,self.result_file,self.min_energy,
                       self.luminosity_distance]

            stype ('\n'+"-----Input parameters--------"+'\n')
            print (tabulate.tabulate(zip(headers,printdata), headers=['Variable Name', 'Value'],
                             tablefmt='orgtbl') )
            stype ('\n'+"-----End Input parameters----")



    #Function to write the parameters to a file
    def print_values_to_file(self):


            outfile=open(self.result_file, 'a')

            headers=["Hubble constant","Energy ratio of e-","Expectral index","Radius of the beam","Cells of the "
                                                                                                   "beam",
                     "Tracer lim . value","Redshift","Viewing angle","RMHD file","Variable dimensions",
                     "Grid dimensions","Non thermal e- fraction","Non thermal e- density","External density",
                     "Observation frequency","Result file","Min_energy","Luminosity distance"]
            printdata=[self.hubble,self.rate_ene,self.gam_sp,self.beam_radius,self.beam_radius_cells,
                       self.tracer_limit,self.redshift,self.viewing_angle, self.RMHD_data_file,[self.x_var_dim,
                       self.y_var_dim],[self.x_grid_dim,self.y_grid_dim],self.non_thermal_fraction,
                       self.non_thermal_density,
                       self.external_density, self.observation_freq,self.result_file,self.min_energy,
                       self.luminosity_distance]

            outfile.write('\n'+"-----Input parameters--------"+'\n')
            outfile.write (tabulate.tabulate(zip(headers,printdata), headers=['Variable Name', 'Value'],
                             tablefmt='orgtbl') )
            outfile.write ('\n'+"-----End Input parameters----")










# This class carries the definition of some constants in cgs

class constants(object):

    def __init__(self,gam_sp):
        self.c=2.9979e10 #speed of light
        self.e_mass=9.1094e-28 #rest-mass of the electron
        self.e_charge= 4.8032e-10 #charge of the electron in 'esu'
        self.c1 =3.e0*self.e_charge/4.e0/np.pi/self.e_mass/self.e_mass/self.e_mass/self.c/self.c/self.c/self.c/self.c
        self.c_emiss= np.sqrt(3.0e0)*self.e_charge*self.e_charge*self.e_charge/16.e0/np.pi/self.e_mass/self.c/self.c*self.c1**((gam_sp-1.e0)/2.e0)
        self.c_absor= np.sqrt(3.0e0)*self.e_charge*self.e_charge*self.e_charge/16.e0/np.pi/self.e_mass*self.c1**(gam_sp/2.e0)*(gam_sp+2.e0)











class rmhd_data(object):

    def __init__(self,importfile):

        stype("Reading RMHd data from file '"+str(importfile.name)+"'....")

        #The file has two values in the header that act as Fortran records (to be confirmed)
        #so, we have to read this values and do nothing with them

        np.fromfile(file=importfile, dtype=np.int32,count= 1).byteswap()
        np.fromfile(file=importfile, dtype=np.int32,count= 1).byteswap()

        # First let's get the dimensions of the arrays and other dimension-related information to read correctly
        # the rest of the arrays.

        self.nsdim = self.readint(importfile)
        self.mnx= self.readint(importfile)
        self.mny = self.readint(importfile)
        self.mnz = self.readint(importfile)
        n=self.mnx*self.mnz

        #Now, we start to read the values taking account of the types, and we transfer those into suitable variables

        self.mn = self.readint(importfile)
        self.nstep = self.readint(importfile)
        self.time = self.readdp(importfile)
        self.igeomx = self.readint(importfile)
        self.igeomy = self.readint(importfile)
        self.igeomz = self.readint(importfile)
        self.gridlx = self.readdp(importfile)
        self.gridly = self.readdp(importfile)
        self.gridlz = self.readdp(importfile)
        self.nx = self.readint(importfile)
        self.ny = self.readint(importfile)
        self.nz = self.readint(importfile)
        self.density =( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.eps = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.tracer = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.velx = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.vely = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.velz =( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.bx = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.by =( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.bz = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.bxs = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.bys = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")
        self.bzs = ( self.readarr(importfile,n) ).reshape((self.mnx,self.mnz),order="FORTRAN")

        stype(bcolors.OKGREEN +"RMHD data read and organized correctly."+ bcolors.ENDC+'\n')


    # Utility funtion to read one integer from a binary file when little endian

    def readint(self,fileobj):
        return (np.fromfile(file=fileobj, dtype=np.int32,count=1).byteswap())[0]

    # Utility funtion to read one double p. float from a binary file when little endian

    def readdp(self,fileobj):
        return (np.fromfile(file=fileobj, dtype=np.float64,count=1).byteswap())[0]

    # Utility funtion to read an array of floats from a binary file when little endian

    def readarr(self,fileobj,n):
        return np.fromfile(file=fileobj, dtype=np.float64,count=n).byteswap()

    def correction(self,external_density):
        c=2.9979e10
        self.bx= self.bx*np.sqrt(4.e0*np.pi*external_density*c*c)
        self.by= self.bx*np.sqrt(4.e0*np.pi*external_density*c*c)
        self.bz= self.bx*np.sqrt(4.e0*np.pi*external_density*c*c)
        self.density= self.density*self.tracer
        self.eps=self.eps*self.tracer



def mirror(array):
    return np.append(array[::-1,::],array,0)



