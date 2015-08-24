#!/usr/bin/env python
# -*- coding: utf8 -*-

__author__ = 'pablogsal'

import importer
import geometry
import bessel_calc
import outputer
from pycuda_files import cuda_driver
from tools import *
import logging
import argparse
import __builtin__
import os


def main():
    """Main function of the HADES code. Run without parameters to execute"""



    #  _____      _ _   _       _ _          _   _
    # |_   _|    (_) | (_)     | (_)        | | (_)
    #   | | _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __
    #   | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
    #  _| || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
    #  \___/_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
    #

    # Config the parser------------------------------------------------------

    ######## START OF PARSER CREATION #########

    parser = argparse.ArgumentParser(
        description='Simulate synchrotron emission from RMHD data file.')

    parser.add_argument('RMHDfile', metavar='RMHDfile', type=str, nargs=1,
                        help='A path to the file with the RMHD data.')

    parser.add_argument('-v', dest='verbose_flag', action='store_const',
                        const=True, default=False,
                        help='Prints more info (default: False)')

    parser.add_argument(
        '-images',
        dest='image_flag',
        action='store_const',
        const=True,
        default=False,
        help='Construct polarization map and background images (default: False)')

    parser.add_argument(
        '--keys',
        dest='ini_keys',
        default='STANDARD_KEYS',
        help='Use the keys in the .ini file (default: STANDARD_KEYS)')

    parser.add_argument('-quiet', dest='quiet_flag', action='store_const',
                        const=True, default=False,
                        help='Prints no data-related info (default: False)')

    args = parser.parse_args()

    # Bad idea of making global the quiet flag:

    __builtin__.qflag = args.quiet_flag

    ######## END OF PARSER CREATION #########

    # Create logger-------------------------------------------------

    ######## START OF LOGGER CREATION #########
    # create logger with 'spam_application'
    logger = logging.getLogger('HADES')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(
        os.path.dirname(
            os.path.realpath(__file__)) +
        '/results/Hades_log.log')
    fh.setLevel(logging.DEBUG)
    # create console handler
    ch = logging.StreamHandler()

    if args.verbose_flag:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.WARNING)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Custom classes (from tools)
    logging.Logger.error = log_error
    logging.Logger.ok = log_OK
    logging.Logger.fail_check = log_WARNING
    ######## END OF LOGGER CREATION #########

    # ___  ___      _
    # |  \/  |     (_)
    # | .  . | __ _ _ _ __
    # | |\/| |/ _` | | '_ \
    # | |  | | (_| | | | | |
    # \_|  |_/\__,_|_|_| |_
    #

    # Show program LOGO
    logger.info('Showing program logo')

    logo = open(
        os.path.dirname(
            os.path.realpath(__file__)) +
        "/logo.txt",
        "r").read()
    stype(logo)

    logger.info('End of program logo')

    # Open the file as binary imput
    logger.info('Opening the file as binary input')

    try:
        f = open(os.path.abspath(args.RMHDfile[0]), "rb")
        logger.info('File ' + str(f.name) + ' opened correctly')
    except IOError:
        logger.error('File not found')
        exit()

    # Instanciate the class parameters with the input file
    par = importer.input_parameters("", args.ini_keys)

    # Instanciate the class rmhd_data with the binary information from the file

    if args.ini_keys == 'STANDARD_KEYS':
        data = importer.rmhd_data(f)
    else:
        data = importer.rmhd_data_y(f, par)

    data.rmhd_test()

    # Instanciate the constants class

    const = importer.constants(par.gam_sp)

    # Open the output_file
    logger.info('Opening the output file ' + par.result_file + '.')
    
    # Pass the magnetic field to cgs and correct the density and eps with the
    # tracer values
    data.correction(par.external_density)

    # Calculate the energy factor of the non-thermal electrons with the eps
    # value at (0,0).
    par.calculate_energy_factor(data.eps[0][0])

    # Print values in the screen
    par.print_values()

    # Create the image object as instantiation of the image class

    obs_map = geometry.image(par)

    # Print the image object info to screen

    obs_map.print_to_screen()

    # Get the indexes of the cells where the jet starts

    jet_limits = geometry.tracer_limits(data.tracer, par.tracer_limit)

    # Get the bessel integrals

    bessel = bessel_calc.bessel_integrals(par.gam_sp)

    # Start cuda driver to procese the image

    logger.info('Starting the CUDA driver.')

    results = cuda_driver.kernel_driver(
        data, par, obs_map, jet_limits, const, bessel)

    logger.info('CUDA driver terminated successfully.')

    # Create a object with the results using output data class

    output_data = outputer.output_data(*results)

    # Make cosmological corrections to the data

    output_data.make_corrections(obs_map, par)

    # Output the data

    output_data.make_hdf5_file(par, obs_map)

    if args.image_flag:
        output_data.make_polarization_map()
        output_data.make_polarization_background()

    logger.ok('Program terminated successfully.')


if __name__ == '__main__':

    main()
