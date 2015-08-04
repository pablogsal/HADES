__author__ = 'pablogsal'

import numpy as np
import logging


#Create logger for module
module_logger = logging.getLogger('HADES.cuda_dict')

def generate_parameter_dict(kernel_code_template,cuda_grid,input_par,obs_map,constants):

    module_logger.info('Requested the substitution of the parameters in the source code.')

    #Get some references of the input parameters for better reading

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





    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': cuda_grid.grid_x,
        'MATRIX_SIZE2': cuda_grid.grid_y,
        'ANGLE': theta,
        'DIM_X': dim_x,
        'DIM_Y': dim_y,
        'Y_MIN': y_min,
        'Y_MAX': y_max,
        'Z_MIN': z_min,
        'Z_MAX': z_max,
        'MAX_CELLS': int(max_cell_path+10), #The +10 is only for security pourposes
        'CLIGHT': constants.c,
        'E_MASS': constants.e_mass,
        'E_CHARGE': constants.e_charge,
        'C_EMISS': constants.c_emiss,
        'C_ABS': constants.c_absor,
        'GAM_SP': input_par.gam_sp,
        'FENONTH':input_par.non_thermal_fraction,
        'FDNONTH':input_par.non_thermal_density,
        'EXT_DEN': input_par.external_density,
        'OBS_FREQ':input_par.observation_freq,
        'RATE_ENE': input_par.rate_ene,
        'REDSHIFT': input_par.redshift,
        'DS': obs_map.cell_size,
        'C1': constants.c1

        }


    module_logger.info('Parameters correctly substituted in the source code.')

    return kernel_code