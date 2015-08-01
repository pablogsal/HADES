__author__ = 'pablogsal'


# This code is for pretty printing numpy arrays
import contextlib
import numpy as np
import os
import pycuda
from pycuda import driver, compiler, gpuarray, tools
import math
from tools import *
import cuda_toolbox
import logging
import tabulate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import seaborn.apionly as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


#Create logger for module
module_logger = logging.getLogger('HADES.cuda_driver')



# @memory_exit
def kernel_driver(data,input_par,obs_map,jet_limits):
    module_logger.info('Starting the CUDA kernel driver.')


    # Renaming things from input_par to easy reading
    # Notice that for the RMHD plane we are using the variable names:
    #
    # x= rho
    # y= z

    dim_x=input_par.x_grid_dim
    dim_y=input_par.y_grid_dim
    view_angle=input_par.viewing_angle
    theta=(90.e0-view_angle)*np.pi/180.e0 #Angle theta
    y_min=int(np.round(obs_map.y_min))
    y_max=int(np.round(obs_map.y_max))
    z_min=int(np.round(obs_map.z_min))
    z_max=int(np.round(obs_map.z_max))

    image_dim_y=int(np.round(obs_map.y_max)-np.round(obs_map.y_min))
    image_dim_z=int(np.round(obs_map.z_max)-np.round(obs_map.z_min))

    #Calculate max cells in a path throught the arrays

    max_cell_path=np.round(np.sqrt(image_dim_z**2+image_dim_y**2))

    module_logger.info('Max cells in the path are '+str(max_cell_path)+'.')


    # When importing this module we are initializing the device.
    # Now, we can call the device and send information using
    # the apropiate tools in the pycuda module.

    module_logger.info('Triying to initialize the device.')
    import pycuda.autoinit
    module_logger.info('Device correctly initialized.')


    # Get device attributes from cuda.toolbox

    device_attr= cuda_toolbox.get_device_attributes(0)


    # First, we start defining the Kernel. The kernel must accept
    # linearized arrays, so, when we construct the indexes, we
    # must take this into account.
    #
    #This Kernel must be written in C/C++.
    #
    # The Kernel is a string with the C code in python so beware of using
    # % and \n in the code because python will interpret these as scape
    # characters. Use %% and \\n instead

    #Reading the kernel C code from files

    cuda_dir=os.path.dirname(os.path.realpath(__file__))

    module_logger.info('Triying to read the Kernel source code.')

    try:

        cuda_geometry=open(cuda_dir+"/../Device_src/cuda_geometry.cc", "r").read()
        cuda_kernel=open(cuda_dir+"/../Device_src/kernel.cc", "r").read()
        cuda_tools=open(cuda_dir+"/../Device_src/cuda_tools.cc", "r").read()
        cuda_energy=open(cuda_dir+"/../Device_src/cuda_energy.cc", "r").read()

    except IOError:

        module_logger.error('Source code not found.')
        exit()

    # Unite the kernel code in kernel_code_template

    code_string=cuda_tools+cuda_energy+cuda_geometry+cuda_kernel

    kernel_code_template = code_string
    module_logger.info('Kernel code correctly loaded.')

    # Define the matrix size and the TILE size.
    # The TILE must divide the matrix size in order to run correctly.

    # The TILE size will be the dimx and dimy of threads per block.
    # So, the grid will be created dividing the MATRIX_SIZE per the TILE_SIZE
    # and then using TILE_SIZE* TILE_SIZE2 threads per block.


    MATRIX_SIZE = image_dim_z
    MATRIX_SIZE2 = image_dim_y

    #Let's find divisors of the matrix to construct the blocks in an optimisez way

    divisors_x=np.array(list(cuda_toolbox.divisorGenerator(image_dim_z)))
    divisors_y=np.array(list(cuda_toolbox.divisorGenerator(image_dim_y)))


    TILE_SIZE = divisors_x[divisors_x<32][-1]
    TILE_SIZE2 = divisors_y[divisors_y<32][-1]


    GRID_SIZE= 1 if MATRIX_SIZE // TILE_SIZE ==0 else MATRIX_SIZE // TILE_SIZE
    GRID_SIZE2=1 if MATRIX_SIZE2 // TILE_SIZE2 ==0 else MATRIX_SIZE2 // TILE_SIZE2

    #Print the GRID configuration

    module_logger.info('The kernel grid is setted as: '+'('+str(GRID_SIZE)+','+str(GRID_SIZE2)+') and ('+str(
        TILE_SIZE)+','+str(TILE_SIZE2)+')')

    headers=["Grid size x","Grid size y","Tile size x","Tile size y"]
    printdata=[GRID_SIZE,GRID_SIZE2,TILE_SIZE,TILE_SIZE2]
    stype('\n'+'CUDA grid properties:')
    print (tabulate.tabulate(zip(headers,printdata), headers=['Variable Name', 'Value'],
                             tablefmt='rst', stralign="left") +'\n')

    # Create two easily recognized arrays using numpy. The type must be float32
    # because CUDA does not manage well with double precission things yet.
    # MUST CHECK THIS

    # For some reason, CUDA only works well if you multiply your matrix with a numpy ones dtype matrix, so keep that
    # in mind.

    #a_cpu = obs_map.test*np.ones((MATRIX_SIZE, MATRIX_SIZE2)).astype(np.float64)+np.ones((MATRIX_SIZE,
    # MATRIX_SIZE2)).astype(np.float64)

    module_logger.info('Loading the arrays in CPU.')

    density = data.density*np.ones((dim_x, dim_y)).astype(np.float64)
    eps = data.eps*np.ones((dim_x, dim_y)).astype(np.float64)
    velx = data.velx*np.ones((dim_x, dim_y)).astype(np.float64)
    vely = data.vely*np.ones((dim_x, dim_y)).astype(np.float64)
    velz = data.velz*np.ones((dim_x, dim_y)).astype(np.float64)
    bx = data.bx*np.ones((dim_x, dim_y)).astype(np.float64)
    by = data.by*np.ones((dim_x, dim_y)).astype(np.float64)
    bz = data.bz*np.ones((dim_x, dim_y)).astype(np.float64)


    a_cpu = np.zeros((image_dim_y, image_dim_z)).astype(np.float64)
    b_cpu = np.zeros((image_dim_y, image_dim_z)).astype(np.float64)

    #b_cpu = np.random.random((MATRIX_SIZE, MATRIX_SIZE2)).astype(np.float64)



    module_logger.info('CPU arrays correctly loaded.')

    #Check if the system has enought memory for the calculation

    cuda_toolbox.cuda_mem_check(device_attr,max_cell_path*10,
                               (density,eps,velx,vely,velz,bx,by,bz,a_cpu,b_cpu,jet_limits))


    # transfer host (CPU) memory to device (GPU) memory

    module_logger.info('Loading the arrays in GPU.')


    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)


    density_gpu=gpuarray.to_gpu(density)
    eps_gpu=gpuarray.to_gpu(eps)
    velx_gpu=gpuarray.to_gpu(velx)
    vely_gpu=gpuarray.to_gpu(vely)
    velz_gpu=gpuarray.to_gpu(velz)
    bx_gpu=gpuarray.to_gpu(bx)
    by_gpu=gpuarray.to_gpu(by)
    bz_gpu=gpuarray.to_gpu(bz)
    jet_limits_gpu=gpuarray.to_gpu(jet_limits*np.ones((dim_y)).astype(np.float64))


    # create empty gpu array for the result
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE2), np.float64)

    module_logger.info('GPU arrays correctly loaded.')


    # get the kernel code from the template
    # by specifying the dimensional constant values.
    # This is only a dictionary substitution of the string to make the kernell call cleaner.

    module_logger.info('Substituting values in kernel code.')


    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE,
        'MATRIX_SIZE2': MATRIX_SIZE2,
        'ANGLE': theta,
        'DIM_X': dim_x,
        'DIM_Y': dim_y,
        'Y_MIN': y_min,
        'Y_MAX': y_max,
        'Z_MIN': z_min,
        'Z_MAX': z_max,
        'MAX_CELLS': int(max_cell_path+10) #The +10 is only for security pourposes
        }


    # Compile the kernel code using pycuda.compiler
    module_logger.info('Triying to compile the kernel.')

    mod = compiler.SourceModule(kernel_code)

    module_logger.info('Kernel correctly compiled.')


    # get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")


    # create two timers so we measure time
    start = driver.Event()
    end = driver.Event()

    start.record() # start timing

    module_logger.info('Starting CUDA Kernel.')

    # call the kernel on the card
    matrixmul(
        # inputs
        density_gpu,eps_gpu,velx_gpu,vely_gpu,velz_gpu,bx_gpu,by_gpu,bz_gpu,jet_limits_gpu,a_gpu, b_gpu,
        # output
        c_gpu,
        # Grid definition -> number of blocks x number of blocks.
        grid = (GRID_SIZE2,GRID_SIZE),
        # block definition -> number of threads x number of threads
        block = (TILE_SIZE2, TILE_SIZE, 1),
        )

    end.record() # end timing
    # calculate the run length
    end.synchronize()
    secs = start.time_till(end)*1e-3

    module_logger.ok( "Cuda kernel executed in %f seconds" % (secs))

    cuda_toolbox.memory_occupance()




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
    # del a_gpu,b_gpu,c_gpu,cosa
    print max_cell_path

    ax=plt.gca()
    im=ax.imshow(cosa,origin='lower',aspect=None)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="8%", pad=0)
    cb=plt.colorbar(im,orientation="horizontal",cax=cax)
    cb.ax.xaxis.set_ticks_position('top')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('foo.png', dpi=100)


