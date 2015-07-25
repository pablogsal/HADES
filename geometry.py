__author__ = 'pablogsal'

import numpy as np
from tools import stype



# This class has the information and contents of the observer map.
# When instantiating the class you must provide an instantiation of
# the input parameters class to create correctly the observer map.
#
# There are two methods that provide print_to_screen and print_to_file
# functionalities.


class image(object):

    def __init__(self,input_par):


        #Renaming things from input_par to easy reading

        # Notice that for the RMHD plane we are using the variable names:
        #
        # x= rho
        # y= z

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

        self.z_min=-dim_x
        self.z_max=dim_x

        # Limits in the y-axis; y_mi always < y_ma
        # These are the rotation of the points (dimx,0) and (-dimx,dimy)
        # around the z-axis. The transformation is from the lab-frame-basis
        # to the observer-frame-axis.

        self.y_min=-np.sin(self.theta)*dim_x

                #Start of evil hacking
                # We must take a decision based on the round of the y_min parameter
                # ASK JOSE LUIS ABOUT THIS BECAUSE IT IS A MESS

        # if (np.abs(np.round(np.abs(self.y_min))+0.51e0-np.abs(self.y_min)) > 1.e-14):
        #     self.y_min==np.round(self.y_min)+0.51e0

                #End of evil hacking


        self.y_max=np.round(  np.sin(self.theta)*dim_x+np.cos(self.theta)*dim_y)


        #Total number of pixel in the observer's map

        self.total_pixels=((self.z_max-self.z_min)+1)*((self.y_max-self.y_min)+1)

        # Size of the cells in cm

        self.cell_lengh=beam_radius/beam_cells*3.0856e18

        # Angular distance (in cm)

        self.angular_distance=lum_dis/(1.e0+redshift)/(1.e0+redshift)

        #Size of cells (or pixels) in mas

        self.cell_area = self.cell_lengh/self.angular_distance*180.e0/np.pi*60.e0*60.e0*1.e3

        # Create the array with indexes

        self.mesh = np.meshgrid( np.linspace(np.int(self.y_min), np.int(self.y_max), np.int(
            self.y_max)-np.int(self.y_min)),np.linspace(np.int(self.z_min), np.int(self.z_max),
            np.int(self.z_max)- np.int(self.z_min)))

        self.mesh= np.array(self.mesh).astype(int)


        # Create an empty array

        self.test = np.zeros([np.round(self.y_max-self.y_min), np.round(
                self.z_max-self.z_min)], dtype=np.float64)


    def print_to_file(self):

            outfile=open(self.out_file, 'a')
            outfile.write('\n'+"-----Size map properties--------"+'\n')

            outfile.write('Pixels in the map: '+str( np.round(self.y_max-self.y_min))+' x '+ str( np.round(
                self.z_max-self.z_min))+'  (horizontal x vertical)')
            outfile.write('Size of pixels in mas: '+str(self.cell_area))

            outfile.write ('\n'+"-----End Size map properties----")

    def print_to_screen(self):


            stype('\n'+"-----Size map properties--------"+'\n')

            print('Pixels in the map: '+str( np.round(self.y_max-self.y_min))+' x '+ str( np.round(
                self.z_max-self.z_min))+'  (horizontal x vertical)')
            print('Size of pixels in mas: '+str(self.cell_area))

            stype ('\n'+"-----End Size map properties----"+'\n')









def tracer_limits(tracer,limit):
        #Redefinitions to easier access to dimensions and data

        dim_x=tracer.shape[0]
        dim_y=tracer.shape[1]

        # Get the values along the z axis (of the Jet, in the rho-z plane) when the jet starts.
        # These values come from the criterion of where the tracer is greater than some value.

        return np.array([np.where(x>limit)[0][-1] for x in tracer.transpose()])



def x_limit(input_par,y_cell,z_cell):


    # Renaming things from input_par to easy reading
    # Notice that for the RMHD plane we are using the variable names:
    #
    # x= rho
    # y= z

    dim_x=input_par.x_grid_dim
    dim_y=input_par.y_grid_dim
    view_angle=input_par.viewing_angle
    theta=(90.e0-view_angle)*np.pi/180.e0 #Angle theta
    # As when we revolve the RMHD rho-z plane around the z axis we
    # are constructing a cylinder, the limits in the x coordinate
    # (perpendicular to the observer plane) change as we change the z
    # coordinate because we are slicing the cylinder at different heights.
    #
    # To take account of these we must calculate the limits in the RMHD slice
    # when doing the cylinder thing. This is easy when using the cylinder eq:
    #
    # X_max^2 = x^2 + z^2   (x,y,z is the LAB frame)
    #
    # So x_lim= Sqrt( x_lim^2-z^2)

    x_lim=np.sqrt(dim_x*dim_x-z_cell*z_cell)

    # Now, the limits for the y variable in the observer map change as the x_lim change. We must recalculate them
    # as we did in the image class.

    y_min=-np.sin(theta)*x_lim
    y_max= np.sin(theta)*x_lim+np.cos(theta)*dim_y


    if y_cell>y_min and y_cell < y_max:


        # Now we can calculate the x limits. These are a little tricky. There are two easy cases: theta=90,0. We
        # take account of these fist.

        if theta==np.pi/2:
            x_min=0+0.51
            x_max=dim_y-0.51

        if theta==0.0:
            x_min=-int(x_lim)+0.51
            x_max= int(x_lim)-0.51

        # Now for the serious business. The x lims are the point where the line y=y_cell cuts the RMHD rectangle.
        # These limits are a little tricky because it depend if the first crossing point is in the z-axis (parallel to) of the RMHD
        # rectangle of in the rho-axis (parallel to). So first we must calculate if the crossing point is in the z-axis (long one) or
        # in the rho axis (short) one. If the crossing is in the z axis (parallel to), we must use trigonometry to
        # calculate these
        # crossing points. The easiest way is to start with the line in the observer frame -> l=(t,y0), then using the
        # rotation matrix calculate this line in the LAB frame as l' = R.transpose * (t,y0) and now calculate the
        # crossing point of l' with the line (x_lim,t)->z axis or the line (t,dim_y) -> rho-axis. Then rotate the
        # crossing point to the observer frame and you are done.
        #
        # In this program we first calculate the crossing point with the z axis and then we check if we must calculate
        # the other one.



        if theta != 0.0 and theta != np.pi/2:

            # X_max calculation

            z_cross=  1/(np.cos(theta)) * ( y_cell + x_lim * np.sin(theta))# Cross with z axis in lab-frame

            if 0 < z_cross < dim_y: #This checks if the cross is long away the RMHD (x_lim,dim_y) coordinate.
                x_max = 1/(np.cos(theta)) * (x_lim + y_cell * np.sin(theta)) # Cross with z axis in obs-frame
            else:
                x_max= ( dim_y - y_cell * np.cos(theta) ) / np.sin(theta) # Cross with rho axis in OBS-frame








            # X_min calculation

            z_cross= y_cell /np.cos(theta) - x_lim * np.tan(theta) # Cross with z axis in lab-frame

            if 0 < z_cross < dim_y: #This checks if the cross is long away the RMHD (x_lim,dim_y) coordinate.
                x_min =  - x_lim /np.cos(theta) + y_cell * np.tan(theta)     # Cross with z axis in obs-frame
            else:
                x_min= -y_cell / np.tan(theta)# Cross with rho axis in OBS-frame







        #Hurray! Now we have to round the calculations.


                    #Start of evil hacking
                    # We must round the calculations.
                    # ASK JOSE LUIS ABOUT THIS BECAUSE IT IS A MESS


            if (np.abs(np.round(np.abs(x_min))+0.51e0-np.abs(x_min)) < 1.e-14):
                x_min==np.round(x_min)
            else:
                x_min==np.round(x_min)+0.51e0


                    #End of evil hacking.


            x_max=np.round(x_max)-0.51



            #Now we check if x_min < x_max and if this is the case, we equal the values so the integration column is 0

            if x_min > x_max:

                x_min= x_max

    else:

        x_min=1
        x_max=-1


    return (np.round(x_min),np.round(x_max))



# This function is to tranform from the x_cell and y_cell in the observer map to the
# x_cell (rho) and y_cell (z) in the RMHD code. It is only a cartesian to polar transformation
# taking into account the rotation between the frames and that the RMHD plane is in a cylinder.

def obs_to_rmhd(input_par,x,y,z):

        # Renaming things from input_par to easy reading
    # Notice that for the RMHD plane we are using the variable names:
    #
    # x= rho
    # y= z

    dim_x=input_par.x_grid_dim
    dim_y=input_par.y_grid_dim
    view_angle=input_par.viewing_angle
    theta=(90.e0-view_angle)*np.pi/180.e0 #Angle theta


    # The x,y,z in lab frame (a rotation of x,y,z)

    x_lab= np.cos(theta) * x - np.sin(theta) * y
    y_lab= np.cos(theta) * y + np.sin(theta) * x
    z_lab = z

    #The rho and z coordinates

    rho=np.sqrt(x_lab * x_lab + z_lab * z_lab )
    z= y_lab

    # Return the rounded values of the variables

    return ( np.int(rho) +1, np.int(z) +1 )

def tracerQ(jet_limits,point):

    return jet_limits[point[1]] > point[0]



