__author__ = 'pablogsal'

import pycuda
from pycuda import driver, compiler, gpuarray, tools
import sys
from tools import *
import tabulate
import logging
import math
import numpy as np


#Create logger for module
module_logger = logging.getLogger('HADES.cuda_toolbox')









class cuda_grid(object):

    def __init__(self,grid_dim_x,grid_dim_y,tile_dim_x=1,tile_dim_y=1):
        #Get class logger

        self.logger = logging.getLogger('HADES.cuda_toolbox.cuda_grid')
        self.logger.info('Creating an instance of cuda_grid')

        self.grid_x=grid_dim_x
        self.grid_y=grid_dim_y

        self.block_x=tile_dim_x
        self.block_y=tile_dim_y

        self.tile_x=tile_dim_x
        self.tile_y=tile_dim_y

        self.logger.info('Succesfully created instance of cuda_grid')


    def set_opt_tile(self):

        self.logger.info('Adjusting the maximal grid configuration.')


        #Let's find divisors of the matrix to construct the blocks in an optimisez way

        divisors_x=np.array(list(self.divisorGenerator(self.grid_x)))
        divisors_y=np.array(list(self.divisorGenerator(self.grid_y)))


        self.tile_x = divisors_x[divisors_x<32][-1]
        self.tile_y = divisors_y[divisors_y<32][-1]

        self.block_x = 1  if  self.grid_x //  self.tile_x ==0 else self.grid_x // self.tile_x
        self.block_y = 1  if  self.grid_y //  self.tile_y ==0 else self.grid_y // self.tile_y

        self.logger.info('Maximal configuration correctly adjusted.')

    def divisorGenerator(self,n):
        '''This function calculates the divisors of the number n'''
        self.logger.info('Requested the calculation of divisors of grid size '+str(n)+'.')
        large_divisors = []
        for i in xrange(1, int(math.sqrt(n) + 1)):
            if n % i is 0:
                yield i
                if i is not n / i:
                    large_divisors.insert(0, n / i)
        for divisor in large_divisors:
            yield divisor
        self.logger.info('Divisors of grid size '+str(n)+' correctly calculated.')




    def print_grid(self):

            #Print the GRID configuration

        self.logger.info('The kernel grid is setted as: '+'('+str(self.block_x)+','+str(self.block_y)+') and ('+str(
            self.tile_x)+','+str(self.tile_y)+')')

        headers=["Grid size x","Grid size y","Tile size x","Tile size y"]
        printdata=[self.block_x,self.block_y,self.tile_x,self.tile_y]
        stype('\n'+'CUDA grid properties:')
        stype (tabulate.tabulate(zip(headers,printdata), headers=['Variable Name', 'Value'],
                             tablefmt='rst', stralign="left") +'\n')











def get_device_attributes(device_number):
    '''Function to get the device properties
    as a dictionary when given the device number'''

    device=driver.Device(device_number)
    attrs=device.get_attributes()
    dict={}
    for (key,value) in attrs.iteritems():
                dict.update({str(key):value})
    return dict


def memory_occupance():
    '''Function to give the memory ocuppancy of the device'''
    module_logger.info('Requested the calculation of free memory on device.')
    free,total= driver.mem_get_info()
    module_logger.info("Global memory occupancy:%f %% free"%(free*100/total))
    module_logger.info('Free memory correctly calculated.')








def cuda_mem_check(device_dictionary,cache_size,arrays):
    '''Function to check if there will be enought memory in the GPU
       to perform the computation'''

    module_logger.info('Checking if the system has enought memory on device.')

    input_size=0

    for array in arrays:
        input_size = input_size + array.nbytes

    cache_size_bytes = cache_size *4

    free,total= driver.mem_get_info()

    max_mem_size=512*1000

    memory_limit=(total-input_size)/device_dictionary['MULTIPROCESSOR_COUNT']/device_dictionary[
        'MAX_THREADS_PER_MULTIPROCESSOR']

    limitator=min(max_mem_size,memory_limit)

    if cache_size_bytes >= limitator:

        module_logger.error("Cache memory per thread ("+bytes2human(cache_size_bytes)+") is greater than memory "
                            "limitation per thread ("+bytes2human(limitator)+")")
        exit()


    elif input_size >= total:

        module_logger.error("The arrays to transfer ("+bytes2human(input_size)+") is greater than global memory "
                            "limitations ("+bytes2human(total)+")")
        exit()


    else:

        headers=("Cache size per thread","Maximum memory size per thread")
        printdata=(bytes2human(cache_size_bytes),bytes2human(limitator))
        stype('\n'+'Memory limitation status on device:')
        stype (tabulate.tabulate(zip(headers,printdata), headers=['Variable Name', 'Value'],
                                             tablefmt='rst')+'\n')

        module_logger.ok('The system has enought memory to perform the calculation.')
        module_logger.info('Using '+bytes2human(cache_size_bytes)+' out of '+bytes2human(limitator)+'.')

        # module_logger.warning("Warning: The cuda kernel will use max capacity of graphics procesors, the screen could "
        #                "become unresponsible during the process.")

        stype('\n'+bcolors.WARNING +"Warning: The cuda kernel will use max capacity of graphics procesors,"
                                    +'\n the screen could become unresponsible during the process.'+ bcolors.ENDC+'\n')