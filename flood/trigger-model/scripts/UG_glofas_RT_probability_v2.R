rm(list=ls())
require(ncdf4)
library(lubridate)
library(dplyr)
library(xts) 
library(zoo)
library(extRemes)
library(janitor)
library(tidyverse)

library(ggplot2)
# ------------------------ import DATA  -----------------------------------


all_glofas_dfs<-read.csv('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/GLOFAS/Uganda_Glofas_df_v1.csv',sep=',')



all_glofas_dfs1<-all_glofas_dfs%>%
  dplyr::mutate(Date=as.Date(Date),
                date2=strptime(Date,"%Y-%m-%d"),
                threshold=act_rt5,                         #DEFINE THRESHOLD IN THE FOLLOWING LINE achere act_rt5=5 year return period act_rt3=3 year return period etc)
                Year=year(Date))%>%
  filter(Date > as.Date("2000-01-01"))%>%
  dplyr::select(Date,date2,st,step,act_rt10,threshold,Year)

##################################
day_range<-50 #between flood events

prob_thr=50

all_days <- tibble(Date = seq(min(all_glofas_dfs1$Date),max(all_glofas_dfs1$Date) , by="days"))

df_filled <- as.data.frame(merge(all_days, tibble(st = unique(all_glofas_dfs1$st))))%>%
  left_join(all_glofas_dfs1,by=c('st','Date'))%>%
  dplyr::group_by(st)%>%arrange(Date)%>%
  fill(threshold, .direction = "downup")%>%fill(step, .direction = "downup")%>%
  dplyr::ungroup()

######### seven day lead time 

df_7 <- df_filled %>%filter(step==7) %>%dplyr::group_by(st)%>%arrange(Date)%>%
  dplyr::mutate(either_exceeds_threshold=threshold>prob_thr,
                either_next_exceeds_threshold = lead(either_exceeds_threshold),
                either_prev_exceeds_threshold = dplyr::lag(either_exceeds_threshold),
                either_peak_start = either_exceeds_threshold & !either_prev_exceeds_threshold,
                either_peak_end = either_exceeds_threshold & !either_next_exceeds_threshold,
                either_peak_start_range = lead(either_peak_start, 0),  # There is an option here to also count days before the trigger but currently we don't use it
                either_peak_end_range = dplyr::lag(either_peak_end, day_range),  # Count events day_range amount of days after the trigger as the same peak
                either_peak_end_range = replace_na(either_peak_end_range, FALSE),
                either_in_peak_range = cumsum(either_peak_start_range) > cumsum(either_peak_end_range), # Combine peaks within the same range into a single peak
                #flood_in_which_peak = cumsum(either_peak_start_range) * flood * either_in_peak_range,
                # Check when protocol is triggered and whether floods are forecasted correctly
                protocol_triggered = either_in_peak_range & !dplyr::lag(either_in_peak_range))%>%
  dplyr::ungroup()%>%filter(protocol_triggered=='TRUE')%>% dplyr::select(st,step,date2,threshold,protocol_triggered,Year)


names(df_7)<-c("Glofas Station","Lead Time","Date_trigger","Probability","Protocol_triggered","Year")



df_7<-df_7 %>% dplyr::mutate(behind=lead(Date_trigger),deltat=abs(difftime(behind,Date_trigger,units="days")))%>%
  filter(deltat>90)%>%
  dplyr::select(-Protocol_triggered,-deltat,-behind)

######################visualize table

df_7%>%dplyr::select(-Year)%>%
  as_hux()%>%
  set_text_color(1, everywhere, "blue")%>%
  theme_article() %>% 
  set_caption("5 year RTP Trigger Table for ")


