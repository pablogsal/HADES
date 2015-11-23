__author__ = 'pablogsal'

import os
import warnings
import h5py
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import collections as plt_collections

from tools import *


#Create logger for module
module_logger = logging.getLogger('HADES.outputer')


class output_data(object):

    def __init__(self,sia,sib,suab,tau,rmea,botf):
        #Get class logger

        self.logger = logging.getLogger('HADES.outputer.output_data')
        self.logger.info('Creating an instance of output_data')

        self.sia = sia
        self.sib = sib
        self.suab = suab
        self.tau = tau
        self.rmea = rmea
        self.botf = botf

        self.logger.info('Instance of output_data successfully created')


    def make_corrections(self,obs_map,parameters):

        self.logger.info('Requested the cosmological corrections and unit transformation of the output_data')

        #Renaming things to easy use

        rshift= parameters.redshift
        ds= obs_map.cell_size
        l_dist= parameters.luminosity_distance


        #Solid angle with cosmological corrections

        s_angle=ds*ds*(1.e0+rshift)**4/l_dist/l_dist

        #  Flux from specific intensity multiplying by the solid angle

        self.sia= self.sia *s_angle
        self.sib= self.sib *s_angle
        self.suab= self.suab *s_angle

        # C  Units to Janskys

        self.sia= self.sia *1.e23
        self.sib= self.sib *1.e23
        self.suab= self.suab *1.e23

        #  Total flux

        self.total_flux = self.sia+self.sib

        #  Q Stokes parameter

        self.stokes_Q = self.sia-self.sib

        #  U Stokes parameter

        self.stokes_U = self.suab

        #  Polarization angle

        self.pol_angle = 1/2.0 * np.arctan2(self.stokes_U,self.stokes_Q)

        self.logger.info('Cosmological corrections and unit transformation of the output_data done')


    def make_hdf5_file(self,parameters,obs_map):

        self.logger.info('Requested the creation of a HDF5 file')


        file = h5py.File(parameters.result_file, "w")

        Raw_data_group = file.create_group("Raw_data")

        Raw_data_group.create_dataset("sia", data=self.sia)
        Raw_data_group.create_dataset("sib", data=self.sib)
        Raw_data_group.create_dataset("suab", data=self.suab)
        Raw_data_group.create_dataset("tau", data=self.tau)
        Raw_data_group.create_dataset("rmea", data=self.rmea)
        Raw_data_group.create_dataset("botf", data=self.botf)

        Processed_data_group = file.create_group("Data")

        Processed_data_group.create_dataset("Total_flux", data=self.total_flux)
        Processed_data_group.create_dataset("Stokes_Q", data=self.stokes_Q)
        Processed_data_group.create_dataset("Stokes_U", data=self.stokes_U)
        Processed_data_group.create_dataset("Optical_depth", data=self.tau)
        Processed_data_group.create_dataset("Botf", data=self.botf)
        Processed_data_group.create_dataset("Rotation measure", data=self.rmea)
        Processed_data_group.create_dataset("Polarization angle", data=self.pol_angle)


        for key,entry in parameters.__dict__.items():

            if not isinstance(entry,logging.Logger):
                Processed_data_group.attrs[key] = entry

        Processed_data_group.attrs['obs_map_z_min'] = obs_map.z_min
        Processed_data_group.attrs['obs_map_z_max'] = obs_map.z_max
        Processed_data_group.attrs['obs_map_y_min'] = obs_map.y_min
        Processed_data_group.attrs['obs_map_y_max'] = obs_map.y_max
        Processed_data_group.attrs['obs_map_total_pixels'] = obs_map.total_pixels
        Processed_data_group.attrs['obs_map_cell_size'] = obs_map.cell_size
        Processed_data_group.attrs['obs_map_cell_area'] = obs_map.cell_area

        file.close()

        self.logger.info('HDF5 file created correctly.')


    def make_polarization_map(self):



        self.logger.info('Requested the creation of a polarization image.')

        output_folder=os.path.dirname(os.path.realpath(__file__))+'/'


        rescaled_flux=np.sqrt( np.power( self.stokes_Q,2) + np.power( self.stokes_U,2))


        colormap=sns.cubehelix_palette(start=2.3, light=1,gamma=0.8, as_cmap=True,
                                                                           reverse=True)
        self.logger.info('Constructing the background of the polarization map')
        # CONSTRUCTION OF THE BACKGROUND

        #Get the figure axes
        figure_axes=plt.gca()
        #Construct the flux map as an array image
        flux_map_background=figure_axes.imshow(rescaled_flux,origin='lower',aspect=None
                                               ,cmap=colormap)

        self.logger.info('Constructing the lines of the polarization map')

        #  CONSTRUCTION OF THE VECTOR PLOT

        #Get the polarization lines from the polarization angle

        lines =construct_lines_from_angle(self.pol_angle,self.total_flux)

        # Add the lines to the matplotlib collection lines with a good widths

        lc = plt_collections.LineCollection(lines, linewidths=.5,colors='white')

        # Add the lines to the current figure axes and scale

        figure_axes.add_collection(lc)
        figure_axes.autoscale()
        # figure_axes.margins(0)

        self.logger.info('Constructing the color bar of the polarization map')
        # #  CONSTRUCTION OF THE COLOR BAR
        #Construct the colorbar and place it on the top
        divider = make_axes_locatable(figure_axes)
        cax = divider.append_axes("top", size="4%", pad=0)
        image_colorbar=plt.colorbar(flux_map_background,orientation="horizontal",cax=cax)
        image_colorbar.ax.xaxis.set_ticks_position('top')

        #Get the figure, adjust it and save it in the folder

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(output_folder+'results/Polarization_map.png', dpi=100)

        self.logger.info('Polarization map correctly constructed.')

        plt.close()
        warnings.filterwarnings("always")


    def make_polarization_background(self):

        self.logger.info('Requested the creation of a background image.')

        output_folder=os.path.dirname(os.path.realpath(__file__))+'/'


        rescaled_flux=self.total_flux*1e7


        colormap=sns.cubehelix_palette(start=2.3, light=1,gamma=0.8, as_cmap=True,reverse=True)
        self.logger.info('Constructing the background.')
        # CONSTRUCTION OF THE BACKGROUND

        #Get the figure axes
        figure_axes=plt.gca()
        #Construct the flux map as an array image
        flux_map_background=figure_axes.imshow(rescaled_flux,origin='lower',aspect=None
                                               ,cmap=colormap)

        self.logger.info('Constructing the contours of the background image.')
        # CONSTRUCTION OF THE CONTOURS

        levels = np.linspace(np.max(rescaled_flux)*0.1,np.max(rescaled_flux)*0.6,5)

        contour_plot= plt.contour(rescaled_flux, levels,
                 origin='lower',
                 linewidths=1,
                 colors='w')

        # Construct labels for the contours
        warnings.filterwarnings("ignore")
        plt.clabel(contour_plot, fontsize=10)

        self.logger.info('Constructing the color bar of the background image.')


        # #  CONSTRUCTION OF THE COLOR BAR
        #Construct the colorbar and place it on the top
        divider = make_axes_locatable(figure_axes)
        cax = divider.append_axes("top", size="4%", pad=0)
        image_colorbar=plt.colorbar(flux_map_background,orientation="horizontal",cax=cax)
        image_colorbar.ax.xaxis.set_ticks_position('top')

        #Get the figure, adjust it and save it in the folder

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(output_folder+'results/Polarization_contours.png', dpi=100)

        self.logger.info('Background image correctly constructed.')
        warnings.filterwarnings("always")






