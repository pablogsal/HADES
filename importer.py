__author__ = 'pablogsal'

import ConfigParser
import os

import tabulate

from tools import *

"""This module provides an interface for I/O
of the relevant MagnetoHidrodinamic files and
parameters defined in the configuration files.
Also, it cantains usefull clases of constants.
"""


# Create logger for module
module_logger = logging.getLogger('HADES.importer')


# Class for output colors

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class input_parameters(object):
    """This class provides an interface to read the input parameter from
    the configuration files.

        The class has mainly properties, that are the parameters readed
        from the configuration files or the ones calculated from them.

        To instantiate the class you must provide

        - **INITIALIZATION**:

              :param importfile: A string containing the path of the config file.
                                 If an empty string is provided, a standard path
                                 is used instead.

              :param key: A key from the config files indicating which section we
                          must read to obtain the parameters. This is provided using
                          the --key= option in the parser
              :type importfile: String
              :type key: String

        - **OUTPUT**:

        A object of the class input_parameters."""

    def __init__(self, importfile, key):
        # Get class logger

        self.logger = logging.getLogger('HADES.importer.input_parameters')
        self.logger.info('Creating an instance of input_parameters')

        # Some units

        c = 2.9979e10

        # We have to read the information from the INI file to configure the
        # parameters.

        Config = ConfigParser.ConfigParser()

        # Read from the supported filename, if the name is empty string then
        # read from the ACTUAL directory of the file to aboid path-related
        # problems.

        if importfile == "":
            Config.read(
                os.path.dirname(
                    os.path.realpath(__file__)) +
                '/conf.INI')
        else:
            Config.read(importfile)

        # Read the variables from the config file.
        self.logger.info('Reading input values')

        self.hubble = float(Config.get(key, "Hubble"))
        self.rate_ene = float(Config.get(key, "Rate_ene"))
        self.gam_sp = float(Config.get(key, "Gam_sp"))
        self.beam_radius = float(Config.get(key, "Beam_radius"))
        self.beam_radius_cells = int(Config.get(key, "Beam_radius_cells"))
        self.tracer_limit = float(Config.get(key, "Tracer_limit"))
        self.redshift = float(Config.get(key, "Redshift"))
        self.viewing_angle = float(Config.get(key, "Viewing_angle"))
        self.RMHD_data_file = Config.get(key, "RMHD_data_file")
        self.x_var_dim = int(Config.get(key, "x_var_dim"))
        self.y_var_dim = int(Config.get(key, "y_var_dim"))
        self.x_grid_dim = int(Config.get(key, "x_grid_dim"))
        self.y_grid_dim = int(Config.get(key, "y_grid_dim"))
        self.scalex = int(Config.get(key, "scalex"))
        self.scaley = int(Config.get(key, "scaley"))
        self.non_thermal_fraction = float(
            Config.get(key, "Non_termal_fraction"))
        self.non_thermal_density = float(Config.get(key, "Non_termal_density"))
        self.external_density = float(Config.get(key, "External_density"))
        self.observation_freq = float(Config.get(
            key, "Observation_freq")) * 1.e9  # Pass to Hz the values
        self.result_file = Config.get(key, "Result_file")
        self.min_energy = 1.0e-04  # Standar min energy of non-thermal electrons
        self.q0 = 0.5  # Cosmological parameter
        self.h0 = self.hubble * 1.e5 / 3.0856e18 / 1.e6
        self.luminosity_distance = c / self.h0 / self.q0 / self.q0 * \
            (self.q0 * self.redshift + (self.q0 - 1.e0) * (np.sqrt(1.e0 + 2.e0 * self.q0 * self.redshift) - 1))

        self.logger.info('Input values readed correctly')

        # Now we check that the variables for correct values
        self.logger.info('Checking input values.')

        if self.gam_sp < 1 or self.gam_sp == 2:
            self.logger.error('The spectral index must be >1 and =! 2.')
            exit()

        if self.x_grid_dim > self.x_var_dim:
            self.logger.error(
                'Variable x_grid_dim must be smaller than x_var_dim')
            exit()

        if self.y_grid_dim > self.y_var_dim:
            self.logger.error(
                'Variable x_grid_dim must be smaller than x_var_dim')
            exit()

        if self.viewing_angle < 0 or self.viewing_angle > 90:
            self.logger.error('Viewing angle must be between 0 and 90 degs')
            exit()

        self.logger.ok('All input values are correct.')

    def calculate_energy_factor(self, eps):
        """ This function calculates the energy factor of the non thermal electrons
         and write the value in self.non_thermal_density. The actual formulas for these
         calculatons can be found in NEED SOURCE!!!  Jose Luis Thesis ?

        Notice that if the non_thermal_density =! 0 then the self.min_energy must be calculated!

        - **INITIALIZATION**:

              :param eps: A real number indicating the eps value at (0,0) from the RMHD class.
              :type arg1: Real Number

        - **OUTPUT**:

        None - Recalculate some parameters like non_thermal_fraction and min_energy."""

        self.logger.info('Requested the calculation of energy factor.')
        # Now, we can calculate the energy factor for the non-thermal electrons
        c = 2.9979e10  # speed of light
        e_mass = 9.1094e-28  # rest-mass of the electron

        if self.non_thermal_fraction == 0:
            self.non_thermal_fraction = self.min_energy / eps / e_mass / c / c * (self.gam_sp - 1e0) / (
                self.gam_sp - 2e0) * (1e0 - self.rate_ene**(2e0 - self.gam_sp)) / (1e0 - self.rate_ene**(1e0 - self.gam_sp))

            if self.non_thermal_fraction < 1:
                self.logger.error('Energy factor for non-thermal e- is <1.')
                self.logger.error('The actual value is = ' +
                                  str(self.non_thermal_density) + ".")
                exit()

        else:
            self.min_energy = self.non_thermal_density * eps * e_mass * c * c * (self.gam_sp - 2.e0) / (self.gam_sp - 1.e0) * (
                1.e0 - self.rate_ene**(1.e0 - self.gam_sp)) / (1.e0 - self.rate_ene**(2.e0 - self.gam_sp))

        self.logger.info('Energy factor correctly calculated.')

    def print_values(self):
        """Function to write the values to the screen"""

        headers = [
            "Hubble constant",
            "Energy ratio of e-",
            "Expectral index",
            "Radius of the beam",
            "Cells of the "
            "beam",
            "Tracer lim . value",
            "Redshift",
            "Viewing angle",
            "RMHD file",
            "Variable dimensions",
            "Grid dimensions",
            "Non thermal e- fraction",
            "Non thermal e- density",
            "External density",
            "Observation frequency",
            "Result file",
            "Min_energy",
            "Luminosity distance"]
        printdata = [self.hubble,
                     self.rate_ene,
                     self.gam_sp,
                     self.beam_radius,
                     self.beam_radius_cells,
                     self.tracer_limit,
                     self.redshift,
                     self.viewing_angle,
                     self.RMHD_data_file,
                     [self.x_var_dim,
                      self.y_var_dim],
                     [self.x_grid_dim,
                      self.y_grid_dim],
                     self.non_thermal_fraction,
                     self.non_thermal_density,
                     self.external_density,
                     self.observation_freq,
                     self.result_file,
                     self.min_energy,
                     self.luminosity_distance]

        stype('\n' + 'Input data from config.INI')
        stype(tabulate.tabulate(zip(headers, printdata), headers=[
              'Variable Name', 'Value'], tablefmt='rst', stralign="left") + '\n')

    def print_values_to_file(self):
        """Function to write the parameters to a file"""

        self.logger.info(
            'Requested print value table to file ' +
            self.result_file +
            '.')
        outfile = open(self.result_file, 'a')

        headers = [
            "Hubble constant",
            "Energy ratio of e-",
            "Expectral index",
            "Radius of the beam",
            "Cells of the "
            "beam",
            "Tracer lim . value",
            "Redshift",
            "Viewing angle",
            "RMHD file",
            "Variable dimensions",
            "Grid dimensions",
            "Non thermal e- fraction",
            "Non thermal e- density",
            "External density",
            "Observation frequency",
            "Result file",
            "Min_energy",
            "Luminosity distance"]
        printdata = [self.hubble,
                     self.rate_ene,
                     self.gam_sp,
                     self.beam_radius,
                     self.beam_radius_cells,
                     self.tracer_limit,
                     self.redshift,
                     self.viewing_angle,
                     self.RMHD_data_file,
                     [self.x_var_dim,
                      self.y_var_dim],
                     [self.x_grid_dim,
                      self.y_grid_dim],
                     self.non_thermal_fraction,
                     self.non_thermal_density,
                     self.external_density,
                     self.observation_freq,
                     self.result_file,
                     self.min_energy,
                     self.luminosity_distance]

        outfile.write('\n' + "-----Input parameters--------" + '\n')
        outfile.write(tabulate.tabulate(zip(headers, printdata), headers=[
                      'Variable Name', 'Value'], tablefmt='rst'))
        outfile.write('\n' + "-----End Input parameters----" + '\n')
        self.logger.info(
            'Value table correctly written to file ' +
            self.result_file +
            '.')


class constants(object):

    """Class that carries some constant definitions in cgs

        To instantiate the class you must provide:

        - **INITIALIZATION**:

              :param gam_sp: A Real number containing the value of the parameter gamma from the parameters class.

              :type gam_sp: Real number

        - **OUTPUT**:

        A object of the class constants."""

    def __init__(self, gam_sp):
        # Get class logger
        self.logger = logging.getLogger('HADES.importer.constants')
        self.logger.info('Creating an instance of constants')
        self.c = 2.9979e10  # speed of light
        self.e_mass = 9.1094e-28  # rest-mass of the electron
        self.e_charge = 4.8032e-10  # charge of the electron in 'esu'
        self.c1 = 3.e0 * self.e_charge / 4.e0 / np.pi / self.e_mass / \
            self.e_mass / self.e_mass / self.c / self.c / self.c / self.c / self.c
        self.c_emiss = np.sqrt(3.0e0) * self.e_charge * self.e_charge * self.e_charge / \
            16.e0 / np.pi / self.e_mass / self.c / self.c * self.c1**((gam_sp - 1.e0) / 2.e0)
        self.c_absor = np.sqrt(3.0e0) * self.e_charge * self.e_charge * self.e_charge / \
            16.e0 / np.pi / self.e_mass * self.c1**(gam_sp / 2.e0) * (gam_sp + 2.e0)


class rmhd_data(object):

    """This class provides an interface to read and organize the rmhd data
    from Chema's code.

    The class has mainly properties, that are readed from the binary file (ejem, ejem),
    and organized in numpy arrays. The reading is somewhat magical because we have to deal
    with the FORTRAN unformatted format (yes, **unformatted format**).

    To instantiate the class you must provide

    - **INITIALIZATION**:

          :param importfile: A string containing the path of the config file.
                             If an empty string is provided, a standard path
                             is used instead.

          :type importfile: String

    - **OUTPUT**:

    A object of the class rmhd_data."""

    def __init__(self, importfile):
        # Get class logger
        self.logger = logging.getLogger('HADES.importer.rmhd_data')
        self.logger.info('Creating an instance of rmhd_data')

        self.logger.info("Reading RMHd data from file '" +
                         str(importfile.name) + "'....")

        # The file has two values in the header that act as Fortran records (to be confirmed)
        # so, we have to read this values and do nothing with them

        np.fromfile(file=importfile, dtype=np.int32, count=1).byteswap()
        np.fromfile(file=importfile, dtype=np.int32, count=1).byteswap()

        # First let's get the dimensions of the arrays and other dimension-related information to read correctly
        # the rest of the arrays.

        self.nsdim = self.readint(importfile)
        self.mnx = self.readint(importfile)
        self.mny = self.readint(importfile)
        self.mnz = self.readint(importfile)
        n = self.mnx * self.mnz

        # Now, we start to read the values taking account of the types, and we
        # transfer those into suitable variables

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
        self.density = (
            self.readarr(
                importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.eps = (self.readarr(importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.tracer = (
            self.readarr(
                importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.velx = (
            self.readarr(
                importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.vely = (
            self.readarr(
                importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.velz = (
            self.readarr(
                importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.bx = (self.readarr(importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.by = (self.readarr(importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.bz = (self.readarr(importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.bxs = (self.readarr(importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.bys = (self.readarr(importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")
        self.bzs = (self.readarr(importfile, n)).reshape(
            (self.mnx, self.mnz), order="FORTRAN")

        self.logger.ok("RMHD data read and organized correctly ")

    # Test function to RMHD data
    def rmhd_test(self):
        """This method test the RMHD data for strange values.

        No initialization parameters

        - **OUTPUT**:

        None, only confirmation message in the logger."""

        self.logger.info("Starting tests to RMHD data ")

        flag = False
        if (self.density < 0).any():
            self.logger.fail_check("Negative values for density founded")
            flag = True
        else:
            self.logger.info("All values of density are correct")

        if (self.eps < 0).any():
            self.logger.fail_check("Negative values for energy founded")
            flag = True
        else:
            self.logger.info("All values of energy are correct")

        if (self.velx > 1).any():
            self.logger.fail_check("Found some values of velx > c")
            flag = True
        else:
            self.logger.info("All values for velx are correct")

        if (self.vely > 1).any():
            self.logger.fail_check("Found some values of vely > c")
            flag = True
        else:
            self.logger.info("All values for vely are correct")

        if (self.velz > 1).any():
            self.logger.fail_check("Found some values of velz > c")
            flag = True
        else:
            self.logger.info("All values for velz are correct")

        if flag:
            self.logger.error("Some test on the RMHD data filed ")
        else:
            self.logger.ok("RMHD data tests passed correctly. ")

        self.logger.info("End of tests to RMHD data.")

    def readint(self, fileobj):
        """Utility funtion to read one integer from a binary file when little endian"""
        return (
            np.fromfile(
                file=fileobj,
                dtype=np.int32,
                count=1).byteswap())[0]

    def readdp(self, fileobj):
        """Utility funtion to read one double p. float from a binary file when ittle endian"""
        return (
            np.fromfile(
                file=fileobj,
                dtype=np.float64,
                count=1).byteswap())[0]

    def readarr(self, fileobj, n):
        """Utility funtion to read an array of floats from a binary file when little endian"""
        return np.fromfile(file=fileobj, dtype=np.float64, count=n).byteswap()

    def correction(self, external_density):
        """This method corrects some arrays of the RMHD data with the tracer
        and converts the values to CGD (Oh my!).

         - **INITIALIZATION**:

            :param external_density: A real number with the value of the external density from the parameters class.

            :type importfile: Real number

         - **OUTPUT**:

            None, some changes in the arrays of B and density

            """

        self.logger.info(
            "Requested correction of magnetic field, density and energy.")

        c = 2.9979e10
        self.bx = self.bx * np.sqrt(4.e0 * np.pi * external_density * c * c)
        self.by = self.by * np.sqrt(4.e0 * np.pi * external_density * c * c)
        self.bz = self.bz * np.sqrt(4.e0 * np.pi * external_density * c * c)
        self.density = self.density * self.tracer
        self.eps = self.eps * self.tracer

        self.logger.info(
            "Correctly corrected the values of magnetic field, density and energy.")


def mirror(array):
    """Utility funtion to mirror an anrray"""
    return np.append(array[::-1, ::], array, 0)


class rmhd_data_y(object):

    """This class provides an interface to read and organize the rmhd data
    from Yosuke's code.

    The class has mainly properties, that are readed from the binary file (ejem, ejem),
    and organized in numpy arrays. The reading is somewhat magical because we have to deal
    with the FORTRAN unformatted format (yes, **unformatted format**).

    To instantiate the class you must provide

    - **INITIALIZATION**:

          :param importfile: A string containing the path of the config file.
                             If an empty string is provided, a standard path
                             is used instead.

          :type importfile: String

    - **OUTPUT**:

    A object of the class rmhd_data."""

    def __init__(self, importfile, parameters):
        # Get class logger
        self.logger = logging.getLogger('HADES.importer.rmhd_data')
        self.logger.info('Creating an instance of rmhd_data')

        self.logger.info("Reading RMHd data from file '" +
                         str(importfile.name) + "'....")

        self.mnx = parameters.x_grid_dim
        self.mny = parameters.y_grid_dim
        self.mnz = parameters.y_grid_dim
        n = self.mnx * self.mnz

        data = np.loadtxt(importfile.name)

        data = data.transpose()

        # Now, we start to read the values taking account of the types, and we
        # transfer those into suitable variables

        "rho, v^r, v^\\phi, v^z, p, B^r, B^\\phi, B^z"
        self.density = (data[2]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.eps = (
            (3.0 *
             data[6]) /
            data[2]).reshape(
            (self.mnx,
             self.mnz),
            order="FORTRAN")
        self.tracer = (data[10]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.velx = (data[3]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.vely = (data[4]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.velz = (data[5]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.bx = (data[7]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.by = (data[8]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.bz = (data[9]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.bxs = (data[7]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.bys = (data[8]).reshape((self.mnx, self.mnz), order="FORTRAN")
        self.bzs = (data[9]).reshape((self.mnx, self.mnz), order="FORTRAN")

        self.logger.ok("RMHD data read and organized correctly ")

    # Test function to RMHD data
    def rmhd_test(self):
        """This method test the RMHD data for strange values.

        No initialization parameters

        - **OUTPUT**:

        None, only confirmation message in the logger."""

        self.logger.info("Starting tests to RMHD data ")

        flag = False
        if (self.density < 0).any():
            self.logger.fail_check("Negative values for density founded")
            flag = True
        else:
            self.logger.info("All values of density are correct")

        if (self.eps < 0).any():
            self.logger.fail_check("Negative values for energy founded")
            flag = True
        else:
            self.logger.info("All values of energy are correct")

        if (self.velx > 1).any():
            self.logger.fail_check("Found some values of velx > c")
            flag = True
        else:
            self.logger.info("All values for velx are correct")

        if (self.vely > 1).any():
            self.logger.fail_check("Found some values of vely > c")
            flag = True
        else:
            self.logger.info("All values for vely are correct")

        if (self.velz > 1).any():
            self.logger.fail_check("Found some values of velz > c")
            flag = True
        else:
            self.logger.info("All values for velz are correct")

        if flag:
            self.logger.error("Some test on the RMHD data filed ")
        else:
            self.logger.ok("RMHD data tests passed correctly. ")

        self.logger.info("End of tests to RMHD data.")

    def readint(self, fileobj):
        """Utility funtion to read one integer from a binary file when little endian"""
        return (
            np.fromfile(
                file=fileobj,
                dtype=np.int32,
                count=1).byteswap())[0]

    def readdp(self, fileobj):
        """Utility funtion to read one double p. float from a binary file when ittle endian"""
        return (
            np.fromfile(
                file=fileobj,
                dtype=np.float64,
                count=1).byteswap())[0]

    def readarr(self, fileobj, n):
        """Utility funtion to read an array of floats from a binary file when little endian"""
        return np.fromfile(file=fileobj, dtype=np.float64, count=n).byteswap()

    def correction(self, external_density):
        """This method corrects some arrays of the RMHD data with the tracer
        and converts the values to CGD (Oh my!).

         - **INITIALIZATION**:

            :param external_density: A real number with the value of the external density from the parameters class.

            :type importfile: Real number

         - **OUTPUT**:

            None, some changes in the arrays of B and density

            """

        self.logger.info(
            "Requested correction of magnetic field, density and energy.")

        c = 2.9979e10
        self.bx = self.bx * np.sqrt(4.e0 * np.pi * external_density * c * c)
        self.by = self.by * np.sqrt(4.e0 * np.pi * external_density * c * c)
        self.bz = self.bz * np.sqrt(4.e0 * np.pi * external_density * c * c)
        self.density = self.density * self.tracer
        self.eps = self.eps * self.tracer

        self.logger.info(
            "Correctly corrected the values of magnetic field, density and energy.")
