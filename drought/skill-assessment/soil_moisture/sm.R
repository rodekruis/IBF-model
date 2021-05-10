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


path = 'C:/Users/pphung'

#zwe_lzh <- st_read(sprintf('%s/Desktop/zwe_spi/zwe_livelihoodzones/ZW_LHZ_2011/ZW_LHZ_2011.shp', path))
zwe <- st_read(sprintf('%s/Rode Kruis/510 - Data preparedness and IBF - [PRJ] FbF - Zimbabwe - Danish Red Cross/3. Data - Hazard exposure, vulnerability/Admin/zwe_admbnda_adm2_zimstat_ocha_20180911/zwe_admbnda_adm2_zimstat_ocha_20180911.shp',path))

sm <- nc_open(sprintf("%s/Desktop/zwe_rain/SM/data/sm_hist.nc",path), verbose = TRUE)



################### Arrange data ##################

years <- ncvar_get(sm,"year")
days_of_year <- ncvar_get(sm,"day")
lons <- ncvar_get(sm,"longitude")
lats <- ncvar_get(sm,"latitude")
days_month <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

sm.df <- NULL
sm_month <- stack()

# Loop per year in the NC file to take SM average per month per adm2

for (i in c(1:dim(years))) {
  for (j in c(1:12)) {
    days <- days_month[j]
    for (k in c(1,days)) {
      sm_1 <- ncvar_get(sm, varid="soil_moisture")[i,k,,]
      sm_r <- raster(sm_1,xmn=24.625, xmx=33.375, ymn=-22.875, ymx=-15.125) # create a raster with data (it is flipped)
      sm_r <- flip(sm_r,2) # correct raster
      projection(sm_r) <- '+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0'
      #sm_r_mean <- extract(sm_r, zwe, fun=mean, na.rm=T)
      sm_month <- stack(sm_month, sm_r)
    }
    sm_r_mean <- exact_extract(sm_month, zwe, 'mean')
    sm_r_mean <- as.data.frame(t(as.matrix(sm_r_mean)))
    names(sm_r_mean) <- as.character(zwe$ADM2_PCODE)            # name column as livelihood zone
    sm_r_mean$month <- j    # name row as month
    sm_r_mean$year <- sm$dim$year$vals[i]
    rbind(sm.df, sm_r_mean) -> sm.df   
    
  }
}
rownames(sm.df) <- NULL

# # save all data
# sm.df_pivot <- sm.df %>%
#   tidyr::pivot_longer(
#     cols = starts_with("ZW"),
#     names_to = "adm2_pcode",
#     values_to = "sm")
# write.csv(sm.df_pivot,"C:/zwe_sm.csv")

# save data of selected months
sm.df_subset <- sm.df[sm.df$month %in% c(1:3,9:12),]  # subset SONDJFM every year
sm.df_pivot <- sm.df_subset %>%
  tidyr::pivot_longer(
    cols = starts_with("ZW"),
    names_to = "adm2_pcode",
    values_to = "sm")
#write.csv(sm.df_pivot,"C:/zwe_sm_subset.csv")