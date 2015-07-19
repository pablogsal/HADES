#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
This program is created as a test to use CUDA to sum 2 arrays of numpy float32.
"""


import numpy as np
from pycuda import driver, compiler, gpuarray, tools




import importer,geometry,bessel_calc


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
























# This code is for pretty printing numpy arrays
import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.


import pycuda.autoinit


# First, we start defining the Kernel. The kernel must accept
# linearized arrays, so, when we construct the indexes, we
# must take this into account. This sample kernel only
# add the two arrays.
#
#This Kernel must be written in C/C++.
#
# The Kernel is a string with the C code in python so beware of using
# % and \n in the code because python will interpret these as scape
# characters. Use %% and \\n instead

cuda_geometry=open("./Device_src/cuda_geometry.cc", "r").read()
cuda_kernel=open("./Device_src/kernel.cc", "r").read()

code_string= cuda_geometry+cuda_kernel

kernel_code_template = code_string

# Define the (square) matrix size and the TILE size.
# The TILE must divide the matrix size in order to run correctly.

# The TILE size will be the dimx and dimy of threads per block.
# So, the grid will be created dividing the MATRIX_SIZE per the TILE_SIZE
# and then using TILE_SIZE* TILE_SIZE2 threads per block.


MATRIX_SIZE = 160
MATRIX_SIZE2 = 600
TILE_SIZE = 1
TILE_SIZE2 = 1


GRID_SIZE= 1 if MATRIX_SIZE // TILE_SIZE ==0 else MATRIX_SIZE // TILE_SIZE
GRID_SIZE2=1 if MATRIX_SIZE2 // TILE_SIZE2 ==0 else MATRIX_SIZE2 // TILE_SIZE2


# create two easily recognized arrays using numpy. The type must be float32
# because CUDA does not manage well with double precission things yet.
# MUST CHECK THIS

a_cpu = data.density*np.ones((MATRIX_SIZE, MATRIX_SIZE2)).astype(np.float64)
b_cpu = np.ones((MATRIX_SIZE, MATRIX_SIZE2)).astype(np.float64)*0

print data.density.shape
print a_cpu.shape
##### compute reference {Marscher:2013bk}on the CPU to verify GPU computation
#### c_cpu =np.dot( a_cpu, b_cpu)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
jet_limits_gpu=gpuarray.to_gpu(jet_limits*np.ones((MATRIX_SIZE2)).astype(np.float64))

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE2), np.float64)

# get the kernel code from the template
# by specifying the constant MATRIX_SIZE
# This is only a dictionary substitution of the string.

kernel_code = kernel_code_template % {
    'MATRIX_SIZE': MATRIX_SIZE,
    'MATRIX_SIZE2': MATRIX_SIZE2
    }

# Compile the kernel code using pycuda.compiler
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMulKernel")


# create two timers so we measure time
start = driver.Event()
end = driver.Event()

start.record() # start timing

# call the kernel on the card
matrixmul(
    # inputs
    jet_limits_gpu,a_gpu, b_gpu,
    # output
    c_gpu,
    # Grid definition -> number of blocks x number of blocks.
    grid = (GRID_SIZE,GRID_SIZE2),
    # block definition -> number of threads x number of threads
    block = (TILE_SIZE, TILE_SIZE2, 1),
    )

end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print "%f seconds" % (secs)

# print the results
print "-" * 80
print "Matrix A (GPU):"
print a_gpu.get()

print "-" * 80
print "Matrix B (GPU):"
print b_gpu.get()

print "-" * 80
print "Matrix C (GPU):"
cosa=c_gpu.get()
print cosa


