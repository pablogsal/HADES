__author__ = 'pablogsal'

import numpy as np


# This class has the information and contents of the observer map.
# When instantiating the class you must provide an instantiation of
# the input parameters class to create correctly the observer map.
#
# There are two methods that provide print_to_screen and print_to_file
# functionalities.


class image(object):

    def __init__(self,input_par):


        #Renaming things from input_par to easy reading

        dim_x=input_par.x_grid_dim
        dim_y=input_par.y_grid_dim
        view_angle=input_par.viewing_angle
        beam_radius=input_par.beam_radius
        beam_cells=input_par.beam_radius_cells
        lum_dis= input_par.luminosity_distance
        redshift=input_par.redshift
        self.out_file=input_par.result_file

        #Angle theta

        self.theta=(90.e0-view_angle)*np.pi/180.e0

        # Limits in the z-axis -> Easy as the actual rotation is around the z axis

        self.z_min=-dim_x+0.51e0
        self.z_max=dim_x-0.51e0

        # Limits in the y-axis; y_mi always < y_ma
        # These are the rotation of the points (dimx,0) and (-dimx,dimy)
        # around the z-axis. The transformation is from the lab-frame-basis
        # to the observer-frame-axis.

        self.y_min=-np.sin(self.theta)*dim_x

                #Start of evil hacking
                # We must take a decision based on the round of the y_min parameter
                # ASK JOSE LUIS ABOUT THIS BECAUSE IT IS A MESS

        if (np.abs(np.round(np.abs(self.y_min))+0.51e0-np.abs(self.y_min)) > 1.e-14):
            self.y_min==np.round(self.y_min)+0.51e0

                #End of evil hacking


        self.y_max=np.round(  np.sin(self.theta)*dim_x+np.cos(self.theta)*dim_y)-0.51e0


        #Total number of pixel in the observer's map

        self.total_pixels=((self.z_max-self.z_min)+1)*((self.y_max-self.y_min)+1)

        # Size of the cells in cm

        self.cell_lengh=beam_radius/beam_cells*3.0856e18

        # Angular distance (in cm)

        self.angular_distance=lum_dis/(1.e0+redshift)/(1.e0+redshift)

        #Size of cells (or pixels) in mas

        self.cell_area = self.cell_lengh/self.angular_distance*180.e0/np.pi*60.e0*60.e0*1.e3

        # Create image empty array data

        self.data = np.empty([np.round(self.y_max-self.y_min)+1, np.round(
                self.z_max-self.z_min)+1], dtype=np.float64)


    def print_to_file(self):

            outfile=open(self.out_file, 'a')
            outfile.write('\n'+"-----Size map properties--------"+'\n')

            outfile.write('Pixels in the map: '+str( np.round(self.y_max-self.y_min)+1)+' x '+ str( np.round(
                self.z_max-self.z_min)+1)+'  (horizontal x vertical)')
            outfile.write('Size of pixels in mas: '+str(self.cell_area))

            outfile.write ('\n'+"-----End Size map properties----")

    def print_to_screen(self):


            print('\n'+"-----Size map properties--------"+'\n')

            print('Pixels in the map: '+str( np.round(self.y_max-self.y_min)+1)+' x '+ str( np.round(
                self.z_max-self.z_min)+1)+'  (horizontal x vertical)')
            print('Size of pixels in mas: '+str(self.cell_area))

            print ('\n'+"-----End Size map properties----")








def tracer_limits(tracer,limit):
        #Redefinitions to easier access to dimensions and data

        dim_x=tracer.shape[0]
        dim_y=tracer.shape[1]

        # Get the values along the z axis (of the Jet, in the rho-z plane) when the jet starts.
        # These values come from the criterion of where the tracer is greater than some value.

        return np.array([np.where(x>.9)[0][-1] for x in tracer.transpose()])





