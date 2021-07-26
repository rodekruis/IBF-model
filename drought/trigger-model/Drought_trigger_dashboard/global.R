rm(list=ls())
library(shiny)
library(janitor)
library(tidyverse)
library(lubridate)
library(plotly)
library(shinydashboard)
library(sf)
library(leaflet)
library(readr)
library(httr)
library(zoo)
library(sp)
#library(ncdf4)
#library(exactextractr)
library(lubridate)
#library(plyr)
#library(dplyr)
#library(lwgeom)




#---------------------- setting -------------------------------

crs1 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source('./r_resources/plot_functions.R')
source('./r_resources/misc_functions.R')
source('r_resources/Geo_settings.R')
source('load_data_zwe.R')



