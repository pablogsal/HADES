__author__ = 'pablogsal'


import numpy as np
import os
import pycuda
from pycuda import driver, compiler, gpuarray, tools
from tools import *
from . import cuda_toolbox, cuda_dict
import logging



#Create logger for module
module_logger = logging.getLogger('HADES.cuda_driver')



# @memory_exit
def kernel_driver(data,input_par,obs_map,jet_limits,constants,bessel):
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

    max_cell_path=np.round(np.sqrt((dim_x*2)**2+(dim_y*2)**2))

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

    cuda_grid=cuda_toolbox.cuda_grid(image_dim_z,image_dim_y)

    cuda_grid.set_opt_tile()
    cuda_grid.print_grid()


    # Create two easily recognized arrays using numpy. The type must be float32
    # because CUDA does not manage well with double precission things yet.
    # MUST CHECK THIS

    # For some reason, CUDA only works well if you multiply your matrix with a numpy ones dtype matrix, so keep that
    # in mind.

    #a_cpu = obs_map.test*np.ones((cuda_grid.grid_x, cuda_grid.grid_y)).astype(np.float64)+np.ones((cuda_grid.grid_x,
    # cuda_grid.grid_y)).astype(np.float64)

    module_logger.info('Loading the arrays in CPU.')


    density = data.density*np.ones((dim_x, dim_y)).astype(np.float64)
    eps = data.eps*np.ones((dim_x, dim_y)).astype(np.float64)
    velx = data.velx*np.ones((dim_x, dim_y)).astype(np.float64)
    vely = data.vely*np.ones((dim_x, dim_y)).astype(np.float64)
    velz = data.velz*np.ones((dim_x, dim_y)).astype(np.float64)
    bx = data.bx*np.ones((dim_x, dim_y)).astype(np.float64)
    by = data.by*np.ones((dim_x, dim_y)).astype(np.float64)
    bz = data.bz*np.ones((dim_x, dim_y)).astype(np.float64)
    jet_limits_cpu=jet_limits*np.ones(dim_y).astype(np.int64)
    besselx=bessel.xvals*np.ones(len(bessel.xvals)).astype(np.float64)
    besself1=bessel.int_f1*np.ones(len(bessel.xvals)).astype(np.float64)
    besself2=bessel.int_f1*np.ones(len(bessel.xvals)).astype(np.float64)
    besselg1=bessel.int_g1*np.ones(len(bessel.xvals)).astype(np.float64)
    besselg2=bessel.int_g2*np.ones(len(bessel.xvals)).astype(np.float64)

    error_test=np.zeros(10)


    sia_cpu = np.zeros((cuda_grid.grid_x, cuda_grid.grid_y)).astype(np.float64)
    sib_cpu = np.zeros((cuda_grid.grid_x, cuda_grid.grid_y)).astype(np.float64)
    suab_cpu = np.zeros((cuda_grid.grid_x, cuda_grid.grid_y)).astype(np.float64)
    tau_cpu = np.zeros((cuda_grid.grid_x, cuda_grid.grid_y)).astype(np.float64)
    rmea_cpu = np.zeros((cuda_grid.grid_x, cuda_grid.grid_y)).astype(np.float64)
    botf_cpu = np.zeros((cuda_grid.grid_x, cuda_grid.grid_y)).astype(np.float64)


    module_logger.info('CPU arrays correctly loaded.')

    #Check if the system has enought memory for the calculation

    cuda_toolbox.cuda_mem_check(device_attr,max_cell_path*10,
                               (density,eps,velx,vely,velz,bx,by,bz,jet_limits,besselx,besself1,besself2,
                                besselg1,besselg2,sia_cpu,sib_cpu,suab_cpu,tau_cpu,rmea_cpu,botf_cpu))


    # transfer host (CPU) memory to device (GPU) memory

    module_logger.info('Loading the arrays in GPU.')

    #Output arrays

    sia_gpu = gpuarray.to_gpu(sia_cpu.astype(np.float64))
    sib_gpu = gpuarray.to_gpu(sib_cpu.astype(np.float64))
    suab_gpu = gpuarray.to_gpu(suab_cpu.astype(np.float64))
    tau_gpu = gpuarray.to_gpu(tau_cpu.astype(np.float64))
    rmea_gpu = gpuarray.to_gpu(rmea_cpu.astype(np.float64))
    botf_gpu = gpuarray.to_gpu(botf_cpu.astype(np.float64))

    #Input arrays
    density_gpu=gpuarray.to_gpu(density.astype(np.float32))
    eps_gpu=gpuarray.to_gpu(eps.astype(np.float32))
    velx_gpu=gpuarray.to_gpu(velx.astype(np.float32))
    vely_gpu=gpuarray.to_gpu(vely.astype(np.float32))
    velz_gpu=gpuarray.to_gpu(velz.astype(np.float32))
    bx_gpu=gpuarray.to_gpu(bx.astype(np.float32))
    by_gpu=gpuarray.to_gpu(by.astype(np.float32))
    bz_gpu=gpuarray.to_gpu(bz.astype(np.float32))
    jet_limits_gpu=gpuarray.to_gpu( jet_limits_cpu.astype(np.int32) )
    besselx_gpu=gpuarray.to_gpu(besselx.astype(np.float32))
    besself1_gpu=gpuarray.to_gpu(besself1.astype(np.float32))
    besself2_gpu=gpuarray.to_gpu(besself2.astype(np.float32))
    besselg1_gpu=gpuarray.to_gpu(besselg1.astype(np.float32))
    besselg2_gpu=gpuarray.to_gpu(besselg2.astype(np.float32))
    error_test_gpu=gpuarray.to_gpu(error_test.astype(np.int32))
    # create empty gpu array for the result


    module_logger.info('GPU arrays correctly loaded.')


    # get the kernel code from the template
    # by specifying the dimensional constant values.
    # This is only a dictionary substitution of the string to make the kernell call cleaner.

    module_logger.info('Substituting values in kernel code.')


    kernel_code = cuda_dict.generate_parameter_dict(kernel_code_template,cuda_grid,input_par,obs_map,constants)

    module_logger.info('Values correctly substituted.')

    # Compile the kernel code using pycuda.compiler


    module_logger.info('Triying to compile the kernel.')

    mod = compiler.SourceModule(kernel_code)

    module_logger.info('Kernel correctly compiled.')


    # get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")


    # create two timers so we measure time
    start = driver.Event()
    end = driver.Event()
    module_logger.info('Starting CUDA Kernel.')

    start.record() # start timing
    # call the kernel on the card
    matrixmul(
        # inputs
        density_gpu,eps_gpu,velx_gpu,vely_gpu,velz_gpu,bx_gpu,by_gpu,bz_gpu,jet_limits_gpu,
        besselx_gpu,besself1_gpu,besself2_gpu,besselg1_gpu,besselg2_gpu,error_test_gpu,
        # output
        sia_gpu,sib_gpu,suab_gpu,tau_gpu,rmea_gpu,botf_gpu,
        # Grid definition -> number of blocks x number of blocks.
        grid = (cuda_grid.block_y,cuda_grid.block_x),
        # block definition -> number of threads x number of threads
        block = (cuda_grid.tile_y, cuda_grid.tile_x, 1))


    end.record() # end timing
    # calculate the run length
    end.synchronize()
    secs = start.time_till(end)*1e-3

    module_logger.ok( "Cuda kernel executed in %f seconds" % (secs))

    cuda_toolbox.memory_occupance()


    #Rewrite the results to cpu arrays

    sia_cpu = cuda_toolbox.delete_nan(sia_gpu.get())
    sib_cpu = cuda_toolbox.delete_nan(sib_gpu.get())
    suab_cpu = cuda_toolbox.delete_nan(suab_gpu.get())
    tau_cpu = cuda_toolbox.delete_nan(tau_gpu.get())
    rmea_cpu = cuda_toolbox.delete_nan(rmea_gpu.get())
    botf_cpu = cuda_toolbox.delete_nan(botf_gpu.get())

    return sia_cpu,sib_cpu,suab_cpu,tau_cpu,rmea_cpu,botf_cpu

