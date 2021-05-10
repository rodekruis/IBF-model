rm(list=ls())
library(dplyr)
library(tidyverse)
library(raster)
library(sf)
library(velox)
library(stringr)
library(ncdf4)
library(exactextractr)
library(lubridate)
library(ggplot2)
library(ggthemes)
library(cowplot)
library(stringr)


################### Load inputs ##################


path = 'C:/Users/pphung/Rode Kruis'

zwe <- st_read(sprintf('%s/510 - Data preparedness and IBF - [PRJ] FbF - Zimbabwe - Danish Red Cross/3. Data - Hazard exposure, vulnerability/Admin/zwe_admbnda_adm2_zimstat_ocha_20180911/zwe_admbnda_adm2_zimstat_ocha_20180911.shp',path))
rain <- read.csv(sprintf("%s/510 - Data preparedness and IBF - [PRJ] FbF - Zimbabwe - Danish Red Cross/3. Data - Hazard exposure, vulnerability/zwe_rain/zwe_daily_rainfall.csv",path))


################### Arrange data ##################

rain$date <- as.Date(rain$date, format="%Y-%m-%d")
rain$year <- year(rain$date)
rain$month <- month(rain$date)

rain.df <- rain %>% 
  group_by(pcode, year, month) %>%
  summarise(pcumul=sum(rain))

rain.df_subset <- rain.df[rain.df$month %in% c(1:3,9:12),]
write.csv(rain.df_subset,
          sprintf("%s/510 - Data preparedness and IBF - [PRJ] FbF - Zimbabwe - Danish Red Cross/3. Data - Hazard exposure, vulnerability/zwe_rain/zwe_monthcumulative_rainfall.csv",path))
