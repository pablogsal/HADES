#!/usr/bin/env python

__author__ = 'pablogsal'

import numpy as np

import importer,geometry,bessel_calc,cuda_driver
from tools import *
from progress_lib import *


#  _____      _ _   _       _ _          _   _
# |_   _|    (_) | (_)     | (_)        | | (_)
#   | | _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __
#   | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
#  _| || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#  \___/_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#



# Open the file as binary imput
f=open("JMC2RAU", "rb")

# Instanciate the class rmhd_data with the binary information from the file
data=importer.rmhd_data(f)

# Instanciate the class parameters with the input file
par=importer.imput_parameters("")

#Open the output_file

f=open(par.result_file, "w")

#Pass the magnetic field to cgs and correct the density and eps with the tracer values
data.correction(par.external_density)

#Calculate the energy factor of the non-thermal electrons with the eps value at (0,0).
par.calculate_energy_factor( data.eps[0][0] )

#Print values in the screen
par.print_values()
par.print_values_to_file()






# ___  ___      _
# |  \/  |     (_)
# | .  . | __ _ _ _ __
# | |\/| |/ _` | | '_ \
# | |  | | (_| | | | | |
# \_|  |_/\__,_|_|_| |_
#

#Create the image object as instantiation of the image class

obs_map=geometry.image(par)

    #Print the image object info to screen

obs_map.print_to_screen()

    #Print the image object info to file

obs_map.print_to_file()

#Get the indexes of the cells where the jet starts

jet_limits=geometry.tracer_limits(data.tracer,par.tracer_limit)

#Get the bessel integrals

bessel=bessel_calc.bessel_integrals()


#Cool progress to prevent the user of the CUDA freeze!
stype('Charging CUDA kernel.....')
print(bcolors.WARNING +"Warning: The cuda kernel will use max capacity of graphics procesors,\nthe screen could "
                       "become unresponsible during the process."+ bcolors.ENDC)

charge()



cosa=cuda_driver.kernel_driver(data,par,obs_map,jet_limits)

