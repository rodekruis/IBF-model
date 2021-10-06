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
library(xts)
#library(ncdf4)
#library(exactextractr)
library(lubridate)
#library(plyr)
#library(dplyr)
library(lwgeom)
library(raster)





#work_dir<-'C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/IBF-system/trigger-model-development/drought/Drought_trigger_dashboard_Ethiopia/'

#setwd(work_dir)

#setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source('./r_resources/plot_functions.R')
source('./r_resources/misc_functions.R')
source('r_resources/Geo_settings.R')
source('r_resources/predict_functions.R')


#---------------------- setting -------------------------------

crs1 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"

countries <- c("Ethiopia" = 1)

levels <- c("LEVEL 2" = 1, "LEVEL 3" = 2)

seasons <- c("Belg" = 1,"Kirmet"=2)

climate_indicator_variable1<- c("rain_season" = 'Average sesonal rain', "vci"='VCI',"spi_1"='1 Month SPI', "spi_2"='2 Month SPI',
                                "spi_3"='3 Month SPI', "spi_6"='6 Month SPI',"spi_12"='12 Month SPI',"TAMSAT_sm"='TAMSAT Soil Moisture')



###########3read admin boundaries 





eth<- sf::read_sf("./shapefiles/Ethiopia/eth-administrative-divisions-shapefiles/eth_admbnda_adm2_csa_bofed_20201008.shp")%>%
  dplyr::mutate(ADM1_PCODE=as.factor(ADM1_PCODE),ADM2_PCODE=as.factor(ADM2_PCODE))%>%
  dplyr::select(ADM1_PCODE,ADM2_PCODE,ADM0_EN)

eth_lhz <- sf::read_sf("./shapefiles/Ethiopia/ET_LHZ_2018/ET_LHZ_2018.shp")%>%
  dplyr::mutate(ADM1_PCODE=as.factor(LZCODE),ADM2_PCODE=as.factor(LZCODE),ADM0_EN=COUNTRY)%>%
  dplyr::select(ADM1_PCODE,ADM2_PCODE,ADM0_EN)


eth <- st_transform(eth, crs = "+proj=longlat +datum=WGS84")
eth_lhz<- st_transform(eth_lhz, crs = "+proj=longlat +datum=WGS84")

# create a data frame with admin boundaries 
admin_lhz <- sf::read_sf("./shapefiles/Ethiopia/Admin/admn2lhz.geojson")%>%dplyr::mutate(level=ifelse(level=='lhz',10,1))


admin_lhz <- st_transform(admin_lhz, crs = "+proj=longlat +datum=WGS84")
 

admin <- list(eth,eth_lhz)


########## correction of admin level 1 code for rainfall and vci


# 
# df1<-read.csv("./data/vci.csv")%>%filter(ADM0_EN %in% c('ET'))%>%
#   dplyr::mutate(date=ymd(date))%>%
#   dplyr::mutate(ADM2_PCODE=factor(pcode),ADM1_PCODE=factor(pcode))%>%
#   group_by(ADM1_PCODE,date)%>%dplyr::summarise(vci=mean(vci, na.rm=TRUE))%>%ungroup()
# 
# df2<-read.csv("./data/vci.csv")%>%filter(ADM0_EN %in% c('ETHIOPIA'))%>%
#   dplyr::mutate(date=ymd(date))%>%
#   dplyr::mutate(ADM2_PCODE=factor(pcode))%>%
#   dplyr::mutate(ADM1_PCODE= ADM2_PCODE)%>%
#   group_by(ADM1_PCODE,date)%>%dplyr::summarise(vci=mean(vci, na.rm=TRUE))%>%ungroup()
# 
# vci_df<-rbind(df1,df2)
# 
# all_days <- tibble(date = seq(min(vci_df$date),max(vci_df$date) , by="days"))
# 
# vci_df_filled <- as.data.frame(merge(all_days, tibble(ADM1_PCODE = unique(vci_df$ADM1_PCODE))))%>%
#   left_join(vci_df,by=c('ADM1_PCODE','date'))%>%
#   dplyr::group_by(ADM1_PCODE)%>%arrange(date)%>%
#   fill(vci, .direction = "downup")%>%dplyr::ungroup()
# 
# vci_df_m<-vci_df_filled%>%dplyr::mutate(Mon=month(date),Year=year(date))%>%
#   dplyr::mutate(season=ifelse(Mon %in%c(3,4,5),1,ifelse(Mon %in%c(7,8,9),2,3)),
#                 Mon=as.factor(as.numeric(Mon)),Year=as.factor(as.numeric(Year)))#%>%dplyr::group_by(ADM1_PCODE,Year,season)%>%dplyr::summarise(vci=mean(vci, na.rm=TRUE))
# 
# 
# 
# df1<-read.csv("./data/Daily_rainfall.csv")%>%
#   dplyr::mutate(date=ymd(date))%>%  filter(ADM0_EN %in% c('ET'))%>%
#   dplyr::mutate(ADM2_PCODE=factor(pcode),ADM1_PCODE=factor(pcode))%>%
#   group_by(ADM1_PCODE,date)%>%dplyr::summarise(rain=mean(rain))%>%ungroup()
# 
# df2<-read.csv("./data/Daily_rainfall.csv")%>%
#   dplyr::mutate(date=ymd(date))%>%  filter(ADM0_EN %in% c('ETHIOPIA'))%>%
#   dplyr::mutate(ADM2_PCODE=factor(pcode))%>%
#   dplyr::mutate(ADM1_PCODE= ADM2_PCODE)%>%
#   group_by(ADM1_PCODE,date)%>%dplyr::summarise(rain=mean(rain))%>%ungroup()
# 
# 
# 
# rain_df_daily<-rbind(df1,df2)%>%mutate(pcode=ADM1_PCODE,date=ymd(date),Year=as.factor(year(date)), Mon =format(date,"%m"),rain=as.numeric(rain))%>%
#   dplyr::mutate(season=ifelse(Mon %in%c('03','04','05'),1,ifelse(Mon %in%c('07','08','09'),2,3)))
# 
# 
# calculate_rain_stat <- function(df)
# {
#   
#   all_rainfall_monthly <-df%>%
#     dplyr::select(pcode,date,Mon,rain,Year,season)%>%group_by(Mon,Year)%>%dplyr::summarise(rain_mon=sum(rain),pcode=dplyr::first(pcode))%>%ungroup()
#   
#   
#   y <- zoo::zoo(df$rain, df$date)
#   m <- hydroTSM::daily2monthly(y, FUN=sum, na.rm=TRUE)
#   monthly_rain=data.frame(m)
#   
#   spi <- SPEI::spi(monthly_rain[,'m'], 1)
#   spi1_ <- as.data.frame(spi$fitted)
#   colnames(spi1_)<-'spi_1'
#   
#   spi <- SPEI::spi(monthly_rain[,'m'], 2)
#   spi2_ <- as.data.frame(spi$fitted)
#   colnames(spi2_)<-'spi_2'
#   
#   spi <- SPEI::spi(monthly_rain[,'m'], 3)
#   spi3_ <- as.data.frame(spi$fitted)
#   colnames(spi3_)<-'spi_3'
#   
#   spi <- SPEI::spi(monthly_rain[,'m'], 6)
#   spi6_ <- as.data.frame(spi$fitted)
#   colnames(spi6_)<-'spi_6'
#   
#   spi <- SPEI::spi(monthly_rain[,'m'], 12)
#   spi12_ <- as.data.frame(spi$fitted)
#   colnames(spi12_)<-'spi_12'
#   
#   spi_df<-bind_cols(spi1_,spi2_,spi3_,spi6_,spi12_)
#   
#   df_spi<-spi_df%>%dplyr::mutate(Mon=format(time(m), "%m"),Year=format(time(m), "%Y"))%>%
#     dplyr::mutate(season=ifelse(Mon %in%c('03','04','05'),1,ifelse(Mon %in%c('07','08','09'),2,3)),
#                   Mon=as.factor(as.numeric(Mon)),Year=as.factor(as.numeric(Year)))#%>%dplyr::filter(season==Season_Obs_Rain)%>%dplyr::group_by(Year)%>%dplyr::summarise(spi_value=min(spi_value))
#   
#   
#   all_rainfall_sesonal <-all_rainfall_monthly%>%dplyr::mutate(season=ifelse(Mon %in%c('03','04','05'),1,ifelse(Mon %in%c('07','08','09'),2,3)))%>%
#     dplyr::select(season,rain_mon,Year,pcode)%>%group_by(season,Year)%>%dplyr::summarise(rain_season=sum(rain_mon,na.rm=TRUE),ADM1_PCODE=dplyr::first(pcode))%>%ungroup()
#   
#   all_rainfall_sesonal_ave <-all_rainfall_sesonal%>%group_by(season)%>%dplyr::summarise(rain_season_ave=mean(rain_season,na.rm=TRUE))%>%ungroup()
#   
#   all_rainfall_stat<-all_rainfall_sesonal%>%left_join(all_rainfall_sesonal_ave,by='season')
#   
#   all_rainfall_stat2<-df_spi%>%left_join(all_rainfall_stat,by=c('Year','season'))
#   
#   
#   
# }
# 
# rainfall_sesonal_df<-rain_df_daily%>%group_by(ADM1_PCODE)%>%group_map(~ calculate_rain_stat(.x))
# 
# rainfall_sesonal_df<-bind_rows(rainfall_sesonal_df)
# 
# #rain_df_daily<-rain_df_daily%>%dplyr::mutate(Mon=as.factor(as.numeric(Mon)))%>%left_join(rainfall_sesonal_df,by = c("ADM1_PCODE" , "Year", "Mon","season"))
# 
# 
# 
# 
# SM_ADMIN<-read.csv("./data/eth_TAMSAT_soil_moisture_adm2.csv")%>%gather('ADM1_PCODE','SM',-X,-date)%>%dplyr::select(-X)
# 
# SM_LHZ<-read.csv("./data/eth_TAMSAT_soil_moisture_lhz.csv")%>%gather('ADM1_PCODE','SM',-X,-date)%>%dplyr::select(-X)
# 
# 
# SM_df<-rbind(SM_ADMIN,SM_LHZ)%>%dplyr::mutate(Mon=month(ymd(date)),ADM1_PCODE=as.factor(ADM1_PCODE),Year=year(ymd(date)))%>%
#   dplyr::mutate(season=ifelse(Mon %in%c(3,4,5),1,ifelse(Mon %in%c(7,8,9),2,3)),
#                 Mon=as.factor(as.numeric(Mon)),Year=as.factor(as.numeric(Year)))%>%dplyr::group_by(ADM1_PCODE,Year,season)%>%
#   dplyr::summarise(TAMSAT_sm=mean(SM, na.rm=TRUE))#%>%dplyr::mutate('SPI'=as.factor('TAMSAT_sm'),'spi_value'=sm,'rain_season_ave'=0)%>%dplyr::select(-sm) #,-X,-date)
# 
# write.csv(SM_df,"./data/eth_TAMSAT_soil_moisture_sesonal.csv")
# 
# sesonal_df<-rainfall_sesonal_df%>%left_join(vci_df_m%>%dplyr::select(-date),by = c("ADM1_PCODE" , "Year", "Mon","season"))   
# 
# sesonal_df<-sesonal_df%>%group_by(ADM1_PCODE,Year,season)%>%
#   dplyr::summarise(rain_season_ave=mean(rain_season_ave,na.rm=TRUE),
#                    vci=mean(vci,na.rm=TRUE),
#                    spi_1=min(spi_1,na.rm=TRUE),
#                    spi_2=min(spi_2,na.rm=TRUE),
#                    spi_3=min(spi_3,na.rm=TRUE),
#                    spi_6=min(spi_6,na.rm=TRUE),
#                    spi_12=min(spi_12,na.rm=TRUE),
#                    #ADM1_PCODE=dplyr::first(ADM1_PCODE),
#                    rain_season =mean(rain_season,na.rm=TRUE ))%>%ungroup()%>%left_join(SM_df,by = c("ADM1_PCODE" , "Year","season"))%>%drop_na()%>%
#   gather("SPI", "spi_value",-Year,-season,-rain_season_ave,-ADM1_PCODE)%>%dplyr::mutate(ADM2_PCODE=ADM1_PCODE)
# 
# 
# 
# 
# 
# write.csv(sesonal_df,"./data/eth_sesona_indicators.csv")
# 




rainfall_sesonal_df<-read.csv("./data/eth_sesona_indicators.csv")%>%dplyr::mutate(Year=as.factor(as.numeric(Year)))%>%dplyr::select(-X)

# rainfall_sesonal_df<-rainfall_sesonal_df%>%group_by(ADM1_PCODE,Year,season)%>%
#   dplyr::summarise(rain_season_ave=mean(rain_season_ave),
#                    spi_1=min(spi_1),
#                    spi_2=min(spi_2),
#                    spi_3=min(spi_3),
#                    spi_6=min(spi_6),
#                    spi_12=min(spi_12),
#                    #ADM1_PCODE=dplyr::first(ADM1_PCODE),
#                    rain_season =mean(rain_season ))%>%ungroup()%>% gather("SPI", "spi_value",-Year,-season,-rain_season_ave,-ADM1_PCODE)






#Dipole Mode Index (DMI)





df_impact_raw <- list()

eth1<- sf::read_sf("./shapefiles/Ethiopia/eth-administrative-divisions-shapefiles/eth_admbnda_adm2_csa_bofed_20201008.shp")%>%
  dplyr::select(ADM2_PCODE,ADM0_EN)

eth1 <- st_transform(eth1, crs = "+proj=longlat +datum=WGS84")

eth_lhz1 <- sf::read_sf("./shapefiles/Ethiopia/ET_LHZ_2018/ET_LHZ_2018.shp") 

eth_lhz1<- st_transform(eth_lhz1, crs = "+proj=longlat +datum=WGS84")


eth_clean_impact_data_updated <- read_delim("./data/impact_eth.csv",",",escape_double = FALSE,trim_ws = TRUE)%>%
  dplyr::mutate(ADM1_PCODE=ADM2_PCODE,ADM2_PCODE=ADM2_PCODE,date=ymd(date_),Year=year(date))%>%
  dplyr::select(ADM1_PCODE,ADM2_PCODE,Year,date,PopulationAffected)

df_impact_raw[[1]] <- eth_clean_impact_data_updated %>%dplyr::mutate(pcode=ADM2_PCODE)%>%
  dplyr::group_by(pcode,Year)%>%dplyr::summarise(PopulationAffected=max(PopulationAffected))%>%dplyr::ungroup()%>%
  dplyr::mutate(date=ymd(paste0(Year,'-01-01')))




MERGED<-st_join(eth1,eth_lhz1)
MERGED$area_sqkm <- st_area(MERGED)/1000000

st_geometry(MERGED)<-NULL 
MERGED<-MERGED %>%group_by(LZCODE)%>% top_n(1, area_sqkm)%>%dplyr::ungroup()%>%dplyr::select(ADM2_PCODE,LZCODE)


df_impact_raw[[10]] <- eth_clean_impact_data_updated%>%left_join(MERGED,by='ADM2_PCODE')%>%drop_na()%>% dplyr::mutate(pcode=LZCODE)%>%
  dplyr::group_by(pcode,Year)%>%
  dplyr::summarise(PopulationAffected=max(PopulationAffected))%>%dplyr::ungroup()%>%
  dplyr::mutate(date=ymd(paste0(Year,'-01-01')))

df_impact_final<-rbind.data.frame(df_impact_raw[[1]],df_impact_raw[[10]])%>%
  dplyr::rename('ADM2_PCODE'='pcode')%>%
  dplyr::mutate(Year=as.factor(Year),ADM2_PCODE=as.factor(ADM2_PCODE),drought_events=TRUE)%>%
  dplyr::select(Year,ADM2_PCODE,drought_events)
  
impact_hazard_sesonal_df<-rainfall_sesonal_df%>%left_join(df_impact_final,by=c('ADM2_PCODE','Year'))


summarize_events <- function(df) {
  df %>% group_by(pcode) %>%  dplyr::summarise(n_events = dplyr::n()) %>% arrange(n_events) %>% ungroup()
}


#summarize_events(df_impact_raw[[1]]) %>% dplyr::select(pcode, n_events)

admin[[3]] <- admin_lhz%>%dplyr::mutate(ADM2_PCODE=as.factor(ADM2_PCODE))#rbind(admin[[1]]%>%dplyr::select(ADM1_PCODE),admin[[2]]%>%dplyr::select(ADM1_PCODE))
 


admin[[1]] <- admin[[1]] %>%
  left_join(summarize_events(df_impact_raw[[1]]) %>%
              dplyr::select(pcode, n_events),   by = c("ADM2_PCODE" = "pcode")) %>%   dplyr::filter(!is.na(n_events))


admin[[10]] <- admin[[2]] %>%
  left_join(summarize_events(df_impact_raw[[10]]) %>%
              dplyr::select(pcode, n_events),   by = c("ADM2_PCODE" = "pcode")) %>%   dplyr::filter(!is.na(n_events))



df_indicators <- list()


#df_indicators[[1]] <- All_df_filled %>% dplyr::filter( ADM0_EN %in% c("KENYA","Kenya"))



#df_indicators[[10]] <-All_df_filled %>% dplyr::filter( ADM0_EN %in% c("KE"))


label <- list()
label[[1]] <- "ADM2_PCODE"
label[[3]] <- "ADM2_PCODE"
label[[10]] <- "ADM2_PCODE"


layerId <- list()
layerId[[1]] <- "ADM2_PCODE"
layerId[[3]] <- "ADM2_PCODE"
layerId[[10]] <- "ADM2_PCODE"


