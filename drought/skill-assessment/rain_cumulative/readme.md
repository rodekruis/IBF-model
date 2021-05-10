# Script for TAMSAT soil moisture data preparation 

## `rain_cumulative.R`

The script to align historical daily precipitation from CHIRPS.

It is a continuation of the DroughtIBF databricks notebook adapted from Aki's script. The notebook reads CHIRPS netcdf files (stored in the datalake) extracts the daily precipitation per admin2 and saves output as csv with columns `pcode`, `date`, `rain`. Note that it takes really long to execute the notebook (~8hrs for me).

The output is a csv file of monthly cumulative of rainfall per admin level 2 or livelihood zones from 1999 to 2020. Columns: `pcode`, `month`, `year`, `pcumul`.

### Usage:

Load input data as below and run the script: 

   `rain`: csv files of daily rainfall
   
All these data can be loaded in easily if synced through OneDrive from the IBF channel in Teams.


