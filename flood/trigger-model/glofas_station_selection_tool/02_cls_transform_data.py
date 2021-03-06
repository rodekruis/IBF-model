##########################################################################################
# Overview
##########################################################################################
"""
Date: July 2020
Authors: Lisanne van Brussel (lisanne.van.brussel@vodw.ey.com)

This class will:
    - Determine the 10yr return period XX percentile discharge for each gridcell.
    - Create a numeric dataframe to identify extreme discharge days (for each 
      individual day/gridcell, e.g. if discharge is above it's own 10yr XX percentile).
    - Determines the surrounding neighbour cells of each cell.
    - Calculates the percentage of overlapping days between the cell and it's neighbours.
    - Saves 3 dataframes:
        * discharge data
        * numeric/dummy dataframe with identifiers of extreme discharge days
        * percentages of overlapping days (rows='selected'cell, cols=neighbours)

"""

##################################################################
# Set workdirectory
##################################################################

# Set workdirectory to folder of scripts
workdirectory_scripts = 'c:\\Users\\BOttow\\Documents\\IBF-system\\trigger-model-development\\flood\\trigger-model\\glofas_station_selection_tool\\'

##################################################################
# Importers
##################################################################

import os
import pandas as pd
import numpy as np

# Change workdirectory
os.chdir(workdirectory_scripts)

# Import class to read in data
cls_read_data = __import__('01_cls_read_data')
rd = cls_read_data.ReadData()

##################################################################
# Create class TransformDataConnectAreas()
##################################################################


class TransformDataConnectAreas:
    """
    ReadGloFAS discharge data per gridcell (0.1x0.1 degree).
    
    Parameters
    ----------  
    verbose
        Whether to print debug information about the current activities. 
        
    Attributes
    ----------
    perc_both_extremes : pd.DataFrame
        DataFrame containing the percentages overlap between a cell and its adjacent (neighbouring) cells.
        Each row is a gridcell ('lat_lon'), the (non-na) columns are the neighbours,
        and the values are the percentage overlap of extreme discharge days.
    
    data_numeric : pd.DataFrame
        Dataframe containing numeric data, 1 indicating extreme discharge day.
        Rows are dates, Columns are gridcells ('lat_lon').
    
    df_discharge : pd.DataFrame
        Dataframe containing the GloFAS discharge date for each day and gridcell.
        Rows are dates, Columns are gridcells ('lat_lon').
    
    
    Functions
    ----------
    _data_to_numeric_threshold
        Determine the 10yr return period XX percentile discharge for each gridcell
        Creates a pandas dataframe with dummies: 1= extreme discharge day (e.g.
        discharge value of that day is above it's own 10yr return period XX percentile).
        
        
    _find_strong_coherence_neighbours  
        Determines the surrounding neighbour cells of each cell.
        Calculates the percentage of overlapping days between the cell and it's neighbours.
        Creates a pandas dataframe with the percentages.       
        
    """ 
    
    def __init__(self,
                 verbose: bool = True):
        
        self.verbose = verbose
        
        # Get settings from config
        self.percentile = rd.cfg['percentile']
        self.neighbour_rounds = rd.cfg['neighbour_rounds']
        self.country = rd.cfg['country']
        self.save_final_data = rd.cfg['save_final_data']
        self.path_save_data = rd.cfg['path_save_data']
        
        # Get GloFAS discharge data
        self.df_discharge = rd.df_discharge.copy()
        
        # Transform data to numeric (based on percentiles of grid cell)
        self._data_to_numeric_threshold(data=self.df_discharge)
        
        # Dataframe with the percentage overlap in past 10 years 
        # (row = selected cell, columns = neighbours)
        self.perc_both_extremes = pd.DataFrame()
        
        # Get and store neighbours of grid cell of interest that have a strong relation
        self.dct_coherence_areas = {}
        self._find_strong_coherence_neighbours()
        
        # Save data
        if self.save_final_data:
            # Percentage overlap data
            path_name = self.path_save_data + 'df_' + self.country + '_percentages_10yr_'+ str(100*self.percentile) + 'percentile.csv'
            self.perc_both_extremes.to_csv(path_name)
            # Discharge data
            path_discharge =  self.path_save_data + 'df_' + self.country + '_discharge_10yr.csv'
            df_discharge_save = self.df_discharge.set_index('date').drop(columns=['month', 'year', 'day'], axis=1)
            df_discharge_save.to_csv(path_discharge)
            # Numeric/dummy data with extreme discharges
            path_numeric =  self.path_save_data + 'df_' + self.country + '_dummy_extreme_discharge_'+ str(100*self.percentile) + 'percentile.csv'
            self.data_numeric.to_csv(path_numeric)     
            print('Three dataframes of ', self.country, ' saved in ', self.path_save_data)
        
        
    def _data_to_numeric_threshold(self,
                                   data : pd.DataFrame() = None,
                                  vars_drop: list = ['year', 'month', 'day']
                                  ):
        """
        Drops unneccessary variables and identifies extreme discharge level for each grid cell, each day.
        Creates a numeric (dummy) dataframe where 1 is an extreme discharge level for that specific grid cell.
        Returns a dummy/numeric pandas dataframe.
        """ 
        
        # Selection of data (drop date columns) and set date as index
        data = data[[x for x in data if x not in vars_drop]].set_index('date')
    
        # Get xx return period xx percentiles per column (= per gridcell)
        self.df_percentile_thresholds = pd.DataFrame(data.quantile(q=self.percentile, 
                                                                   axis=0,
                                                                   interpolation='linear')).T.reset_index(drop=True)
    
        # Copy the data to data_numeric
        self.data_numeric = data.copy()
        
        # Create a numeric dataframe by identifying values for each grid cell that are larger than the corresponding percentile 'threshold' of the specific grid cell.
        for c in self.data_numeric.columns:
            self.data_numeric[c] = np.where(self.data_numeric[c].values > self.df_percentile_thresholds[c].values , 1, 0)
            
    
    def _find_strong_coherence_neighbours(self):
        """
        Determines the surrounding neighbour cells of each cell.
        Calculates the percentage of overlapping days between the cell and it's neighbours.
        Creates a pandas dataframe with the percentages.    
        """
        
        # Select lon/lat coordinates of grid cell of interest (=selected cell)
        for i,c in enumerate(self.data_numeric.columns):
            
            if self.verbose: print('\n location of area of interest ', c)
            
            # Select latitude and longitude 
            lat_interest = round(float(c.split('_')[0]),2)
            lon_interest = round(float(c.split('_')[1]),2)
            
            # Find neighbours, based on rounds of neighbours around
            lon_lat_steps = self.neighbour_rounds/10
            
            # Get all possible lon/lat values of neighbours
            # +0.0001 is needed to include the last value for coordinates in (-1,1)
            lat_range = np.arange(lat_interest -  lon_lat_steps, lat_interest + lon_lat_steps + 0.0001, 0.1)
            lon_range = np.arange(lon_interest -  lon_lat_steps, lon_interest + lon_lat_steps + 0.0001, 0.1)
            
            # Find all possible latitude/longitude combinations
            combis_coordinates = [str(round(lat,2))+'_'+str(round(lon,2)) for lat in lat_range for lon in lon_range]
            
            # Select the numeric data with extreme days of neighbour cells
            combis_coord_available = [x for x in combis_coordinates if x in self.data_numeric.columns]
            data_neighbours = self.data_numeric[combis_coord_available]
            
            if self.verbose: print('data_neighbours shape is ', data_neighbours.shape)        
                   
            # Percentage of both extreme values in cell of interest (selected cell) and its neighbours cell
            data_extremes = data_neighbours[data_neighbours[c]==1]
            perc_both_extremes = pd.DataFrame(100*data_extremes.groupby(c).sum()/data_extremes.shape[0])

    	    # Add new row with the overlapping percentages of the cell of interest (row) 
            # and its neighbours to the pandas dataframe
            if i == 0:
                # Change name index (to coordinates)
                self.perc_both_extremes = perc_both_extremes.rename({1: c}, axis='index')
            else :
                perc_both_extremes_ren = perc_both_extremes.rename({1: c}, axis='index')
                self.perc_both_extremes = pd.concat([self.perc_both_extremes, perc_both_extremes_ren])
        
        # Sort data based on index (=coordinates 'lat_lon' of cells)      
        self.perc_both_extremes = self.perc_both_extremes.sort_index(ascending=False)

  
        
        

        
        
        
        
        
        