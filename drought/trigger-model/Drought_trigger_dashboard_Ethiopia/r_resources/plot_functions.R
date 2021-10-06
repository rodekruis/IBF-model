
plot_drought_indicators <- function(RAIN_PCODE,
                                    impact_df,
                                    Drought_indicator_variable,
                                    climate_indicator_variable){
  vci_rain<-RAIN_PCODE %>% dplyr::mutate(date2=ymd(date))
  max_rain = max(RAIN_PCODE$climate_indicator_variable, na.rm=T)
  min_rain = min(RAIN_PCODE$climate_indicator_variable, na.rm=T)
  max_glofas = max(RAIN_PCODE$climate_indicator_variable, na.rm=T)
  min_glofas = min(RAIN_PCODE$climate_indicator_variable, na.rm=T)
  

  
  p1 <- plot_ly(RAIN_PCODE) %>%
    add_lines(x=~date, y=as.formula(paste0('~', climate_indicator_variable))) %>%
    #add_segments(x=~min(date2), xend=~max(date2), y = vci_threshold, yend=spi_threshold, line=list(color="black")) %>%
    layout(yaxis=list(title="climate_indicator_variable"), showlegend=FALSE)
  

  

  p20<-p1
  for(date in unique(as.character(impact_df$date))) {
  date<-ymd(date)

    p20 <- p20 %>%
      add_segments(x=date ,xend=date, y=min_rain, yend=max_glofas,name = 'Impact event', line=list(color="rgb(255, 153, 0)",width = 4))
  }
  p3 <- subplot(p20, p1, nrows=2)
  return(p3)

}


plot_matrix_spi <- function(spi_index,spi_threshold,RAIN_PCODE){
  
  y <- zoo::zoo(RAIN_PCODE$rain, RAIN_PCODE$date)
  m <- hydroTSM::daily2monthly(y, FUN=sum, na.rm=TRUE)
  monthly_rain=data.frame(m)
  spi <- SPEI::spi(monthly_rain[,'m'], spi_index)
  spi3_ <- as.data.frame(spi$fitted)
  colnames(spi3_)<-'spi'
  spi3_<-spi3_%>%dplyr::mutate(month=format(time(m), "%m"),year=format(time(m), "%Y"))
  spi3_$spi[spi3_$spi>spi_threshold]<-NA
  
  

  
  
  
  p <- ggplot(spi3_, aes(x = year, y = month)) + 
    geom_tile(aes(fill=spi),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "red", high = "white", space = "Lab")
  
  p2<-ggplotly(p)%>%layout(legend = list(orientation = "h"))
  return(p2)
}








plot_matrix_spi_vci <- function(spi_index=1,
                                spi_threshold=-1,
                                RAIN_PCODE,
                                vci_threshold=30,
                                vci_PCODE,
                                Drought_indicator_variable2){
  y <- zoo::zoo(RAIN_PCODE$rain, RAIN_PCODE$date)
  m <- hydroTSM::daily2monthly(y, FUN=sum, na.rm=TRUE)
  
  # all_rainfall_monthly <-RAIN_PCODE%>%
  #   dplyr::mutate(date=ymd(date),Year=year(date),Mon =format(date,"%m"),rain=as.numeric(rain),
  #                 season=ifelse(Mon %in%c('10','11','12'),'OND',ifelse(Mon %in%c('03','04','05','06'),'MAMJ','JF-JAS')))%>%
  #   dplyr::select(Mon,rain,Year,season)%>%group_by(Mon,Year)%>%dplyr::summarise(rain_mon=sum(rain))%>%ungroup()
  # 
  # all_rainfall_sesonal <-all_rainfall_monthly%>%group_by(season,Year)%>%dplyr::summarise(rain_season=sum(rain))%>%ungroup()
  # 
  # all_rainfall_sesonal_ave <-all_rainfall_sesonal%>%group_by(season)%>%dplyr::summarise(rain_season_ave=mean(rain_season))%>%ungroup()
  # 
  # all_rainfall_stat<-all_rainfall_sesonal%>%left_join(all_rainfall_sesonal_ave,by=c('season'))
  # 
  # 
  
  monthly_rain=data.frame(m)
  spi <- SPEI::spi(monthly_rain[,'m'], spi_index)
  spi3_ <- as.data.frame(spi$fitted)
  colnames(spi3_)<-'spi'
  spi3_<-spi3_%>%dplyr::mutate(month=format(time(m), "%m"),year=format(time(m), "%Y"))
  spi3_$spi[spi3_$spi>spi_threshold]<-NA
  y <- zoo::zoo(vci_PCODE$vci, vci_PCODE$date)
  
  vci <- hydroTSM::daily2monthly(y, FUN=mean, na.rm=TRUE)
  monthly_vci=data.frame(vci)
  a<-format(time(vci), "%m")
  b<-format(time(vci), "%Y")
  monthly_vci<-monthly_vci %>% dplyr::mutate(month=a,year=b)
  monthly_vci$vci[monthly_vci$vci>vci_threshold]<-NA
  
  df<-monthly_vci%>%left_join(spi3_,by=c('year','month'))
  p <-ggplot(df, aes(x = ~year, y = ~month)) + 
    geom_tile(aes(fill=as.formula(paste0('~', Drought_indicator_variable2))),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "red", high = "white", space = "Lab")
  
  p3<-ggplotly(p)%>%layout(legend = list(orientation = "h"))
  return(p3)
  
}

plot_matrix_vci <- function(vci_threshold,vci_PCODE){
  
  
  y <- zoo::zoo(vci_PCODE$vci, vci_PCODE$date)
  
  vci_m <- hydroTSM::daily2monthly(y, FUN=mean, na.rm=TRUE)
  monthly_vci=data.frame(vci_m)
  a<-format(time(vci_m), "%m")
  b<-format(time(vci_m), "%Y")
  monthly_vci<-monthly_vci %>% dplyr::mutate(month=a,year=b)
  
  monthly_vci$vci_m[monthly_vci$vci_m>vci_threshold]<-NA
  
  
  p <- ggplot(monthly_vci, aes(x = year, y = month)) + 
    geom_tile(aes(fill=vci_m),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "red", high = "white", space = "Lab")
  p2<-ggplotly(p)
  return(p2)
}


plot_enso1 <- function(ENSO1){
  Nina<-ENSO1
  Nino<-ENSO1
  Nina$ENSO[Nina$ENSO>-1.0]<-NA
  
  Nino$ENSO[Nino$ENSO<1.0]<-NA
  p1 <- ggplot(Nina, aes(x = Year, y = MON)) + 
    geom_tile(aes(fill=ENSO),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "blue", high = "white", space = "Lab")+
    ggtitle("negative SST (<-1.0)")
  
  p2 <- ggplot(Nino, aes(x = Year, y = MON)) + 
    geom_tile(aes(fill=ENSO),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "red", high = "white", space = "Lab")+
    ggtitle("Posetive SST (>1.0)")
  
  
  
  
  p3<-ggplotly(p1)
  return(p3)
  
}

plot_enso2 <- function(ENSO1){
  Nina<-ENSO1
  Nino<-ENSO1
  Nina$ENSO[Nina$ENSO>-1.0]<-NA
  
  Nino$ENSO[Nino$ENSO<1.0]<-NA
  p1 <- ggplot(Nina, aes(x = Year, y = MON)) + 
    geom_tile(aes(fill=ENSO),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "blue", high = "white", space = "Lab")+
    ggtitle("negative SST (<-1.0)")
  
  p2 <- ggplot(Nino, aes(x = Year, y = MON)) + 
    geom_tile(aes(fill=ENSO),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "red", high = "white", space = "Lab")+
    ggtitle("Posetive SST (>1.0)")
  
  
  
  
  p3<-ggplotly(p2)
  return(p3)
  
}

plot_IOD1 <- function(IOD){
  Nina<-IOD
  Nino<-IOD
  Nina$IOD[Nina$IOD>-0.4]<-NA
  
  Nino$IOD[Nino$IOD<0.4]<-NA
  p1 <- ggplot(Nina, aes(x = year, y = MON)) + 
    geom_tile(aes(fill=IOD),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "blue", high = "white", space = "Lab")+
    ggtitle("negative IOD (<-0.4)")
  
  p2 <- ggplot(Nino, aes(x = year, y = MON)) + 
    geom_tile(aes(fill=IOD),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "red", high = "white", space = "Lab")+ 
    ggtitle("Posetive IOD (>0.4)")
  
  
  p3<-ggplotly(P1)
  return(p3)
  
}

plot_IOD2 <- function(IOD){
  Nina<-IOD
  Nino<-IOD
  Nina$IOD[Nina$IOD>-0.4]<-NA
  
  Nino$IOD[Nino$IOD<0.4]<-NA
  p1 <- ggplot(Nina, aes(x = year, y = MON)) + 
    geom_tile(aes(fill=IOD),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "blue", high = "white", space = "Lab")+
    ggtitle("negative IOD (<-0.4)")
  
  p2 <- ggplot(Nino, aes(x = year, y = MON)) + 
    geom_tile(aes(fill=IOD),colour='grey50')+ 
    #scale_color_brewer(palette = "YlOrRd")+
    scale_fill_gradient(low = "red", high = "white", space = "Lab")+ 
    ggtitle("Posetive IOD (>0.4)")
  
  
  p3<-ggplotly(P2)
  return(p3)
  
}


plot_ipc <- function(ipc_df){
  ipc_df$CS[ipc_df$CS<3]<-NA

  p <- ggplot(ipc_df, aes(x = Year, y = month)) + 
    geom_tile(aes(fill=CS),colour='grey50')+ 
    scale_fill_gradient(low = "white", high = "red", space = "Lab")+
    ggtitle("IPC CLASS")
  

  
  p3<-ggplotly(p)
  return(p3)
  
}
