__author__ = 'pablogsal'


# This code is for pretty printing numpy arrays
import contextlib
import numpy as np
import os
from pycuda import driver, compiler, gpuarray, tools

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)







def kernel_driver(data,input_par,obs_map,jet_limits):



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

    # When importing this module we are initializing the device.
    # Now, we can call the device and send information using
    # the apropiate tools in the pycuda module.

    import pycuda.autoinit


    # First, we start defining the Kernel. The kernel must accept
    # linearized arrays, so, when we construct the indexes, we
    # must take this into account.
    #
    #This Kernel must be written in C/C++.
    #
    # The Kernel is a string with the C code in python so beware of using
    # % and \n in the code because python will interpret these as scape
    # characters. Use %% and \\n instead
    cuda_dir=os.path.dirname(os.path.realpath(__file__))
    cuda_geometry=open(cuda_dir+"/Device_src/cuda_geometry.cc", "r").read()
    cuda_kernel=open(cuda_dir+"/Device_src/kernel.cc", "r").read()
    cuda_tools=open(cuda_dir+"/Device_src/cuda_tools.cc", "r").read()

    code_string=cuda_tools+cuda_geometry+cuda_kernel

    kernel_code_template = code_string

    # Define the matrix size and the TILE size.
    # The TILE must divide the matrix size in order to run correctly.

    # The TILE size will be the dimx and dimy of threads per block.
    # So, the grid will be created dividing the MATRIX_SIZE per the TILE_SIZE
    # and then using TILE_SIZE* TILE_SIZE2 threads per block.


    MATRIX_SIZE = image_dim_y
    MATRIX_SIZE2 = image_dim_z
    TILE_SIZE = 1
    TILE_SIZE2 = 1


    GRID_SIZE= 1 if MATRIX_SIZE // TILE_SIZE ==0 else MATRIX_SIZE // TILE_SIZE
    GRID_SIZE2=1 if MATRIX_SIZE2 // TILE_SIZE2 ==0 else MATRIX_SIZE2 // TILE_SIZE2


    # create two easily recognized arrays using numpy. The type must be float32
    # because CUDA does not manage well with double precission things yet.
    # MUST CHECK THIS

    # For some reason, CUDA only works well if you multiply your matrix with a numpy ones dtype matrix, so keep that
    # in mind.

    #a_cpu = obs_map.test*np.ones((MATRIX_SIZE, MATRIX_SIZE2)).astype(np.float64)+np.ones((MATRIX_SIZE,
    # MATRIX_SIZE2)).astype(np.float64)
    density = data.density*np.ones((dim_x, dim_y)).astype(np.float64)
    eps = data.eps*np.ones((dim_x, dim_y)).astype(np.float64)
    velx = data.velx*np.ones((dim_x, dim_y)).astype(np.float64)
    vely = data.vely*np.ones((dim_x, dim_y)).astype(np.float64)
    velz = data.velz*np.ones((dim_x, dim_y)).astype(np.float64)
    bx = data.bx*np.ones((dim_x, dim_y)).astype(np.float64)
    by = data.by*np.ones((dim_x, dim_y)).astype(np.float64)
    bz = data.bz*np.ones((dim_x, dim_y)).astype(np.float64)


    a_cpu = np.zeros((dim_x, dim_y)).astype(np.float64)
    b_cpu = np.zeros((dim_x, dim_y)).astype(np.float64)

    #b_cpu = np.random.random((MATRIX_SIZE, MATRIX_SIZE2)).astype(np.float64)


    # transfer host (CPU) memory to device (GPU) memory
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

    # get the kernel code from the template
    # by specifying the dimensional constant values.
    # This is only a dictionary substitution of the string to make the kernell call cleaner.

    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE,
        'MATRIX_SIZE2': MATRIX_SIZE2,
        'ANGLE': theta,
        'DIM_X': dim_x,
        'DIM_Y': dim_y,
        'Y_MIN': y_min,
        'Y_MAX': y_max,
        'Z_MIN': z_min,
        'Z_MAX': z_max
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
        density_gpu,eps_gpu,velx_gpu,vely_gpu,velz_gpu,bx_gpu,by_gpu,bz_gpu,jet_limits_gpu,a_gpu, b_gpu,
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


    del a_gpu,b_gpu,c_gpu,cosa