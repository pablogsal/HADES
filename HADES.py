#!/usr/bin/env python
# -*- coding: utf8 -*-

__author__ = 'pablogsal'

import importer
import geometry
import bessel_calc
from pycuda_files import cuda_driver
from tools import *
import logging, argparse
import __builtin__

import os




#  _____      _ _   _       _ _          _   _
# |_   _|    (_) | (_)     | (_)        | | (_)
#   | | _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __
#   | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
#  _| || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#  \___/_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#



# Config the parser------------------------------------------------------


######## START OF PARSER CREATION #########

parser = argparse.ArgumentParser(description='Simulate synchrotron emission from RMHD data file.')

parser.add_argument('RMHDfile', metavar='RMHDfile', type=str, nargs=1,
                   help='A path to the file with the RMHD data.')

parser.add_argument('-v', dest='verbose_flag', action='store_const',
                   const=True, default=False,
                   help='Prints more info (default: False)')

parser.add_argument('-quiet', dest='quiet_flag', action='store_const',
                   const=True, default=False,
                   help='Prints no data-related info (default: False)')


args = parser.parse_args()

#Bad idea of making global the quiet flag:

__builtin__.qflag = args.quiet_flag


######## END OF PARSER CREATION #########

# Create logger-------------------------------------------------

######## START OF LOGGER CREATION #########
# create logger with 'spam_application'
logger = logging.getLogger('HADES')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('Hades_log.log')
fh.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()

if args.verbose_flag:
    ch.setLevel(logging.DEBUG)
else:
    ch.setLevel(logging.WARNING)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
#Custom classes (from tools)
logging.Logger.error = log_error
logging.Logger.ok = log_OK
logging.Logger.fail_check=log_WARNING
######## END OF LOGGER CREATION #########

# Now, let's do really fun stuff with the program------------------------------


#Show program LOGO
logger.info('Showing program logo')

logo=open(os.path.dirname(os.path.realpath(__file__))+"/logo.txt", "r").read()
stype(logo)

logger.info('End of program logo')


# Open the file as binary imput
logger.info('Opening the file as binary input')

try:
    f=open( os.path.abspath(args.RMHDfile[0])    , "rb")
    logger.info('File '+str(f.name)+' opened correctly')
except IOError:
    logger.error('File not found')
    exit()



# Instanciate the class rmhd_data with the binary information from the file
data=importer.rmhd_data(f)
data.rmhd_test()

# Instanciate the class parameters with the input file
par=importer.imput_parameters("")


#Instanciate the constants class

const=importer.constants(par.gam_sp)

#Open the output_file
logger.info('Opening the output file '+par.result_file+'.')
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


# charge()


logger.info('Starting the CUDA driver.')

cuda_driver.kernel_driver(data,par,obs_map,jet_limits,const,bessel)


