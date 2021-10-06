
predict_with_rain <- function(RAIN_PCODE2,
                              Season_Obs_Rain,
                              SM_threshold,
                              vci_threshold,
                              spi_threshold,
                              rain_threshold,
                              Impact_df,
                              climate_indicator_variable) {
  
  
  
  
  if (climate_indicator_variable=='rain_season'){
    df<-RAIN_PCODE2%>%dplyr::filter(SPI==climate_indicator_variable,season==Season_Obs_Rain)%>%dplyr::distinct(Year,.keep_all = TRUE)%>%
      left_join(Impact_df %>% dplyr::select(Year) %>% mutate(drought_events = TRUE,Year=as.factor(as.numeric(Year))), by = "Year") %>%
      dplyr::mutate(drought_events = replace_na(drought_events, FALSE),
                    threshold_rain=0.01*rain_season_ave*(100-rain_threshold),
                    exceeds_threshold = spi_value < threshold_rain,
                    #drought_correct = (drought_events == exceeds_threshold),
                    TP = (exceeds_threshold==TRUE) & (drought_events==TRUE),
                    FN = (exceeds_threshold==FALSE) & (drought_events==TRUE),
                    FP = (exceeds_threshold==TRUE) & (drought_events==FALSE),
                    TN = (exceeds_threshold==FALSE) & (drought_events==FALSE),
                    #drought_false_pos = (exceeds_threshold & (drought_events==FALSE)),#FP
                    #drought_correct1 = drought_events & exceeds_threshold
      ) %>%dplyr::summarise(
        TP=sum(TP),
        FN=sum(FN),
        FP=sum(FP),
        TN=sum(TN),
        triggered_in_vain = FP,
        triggered_correct = TP,
        #detection_ratio = round(drought_correct / droughts, 2),
        #false_alarm_ratio = round(triggered_in_vain/protocol_triggered, 2),
        POD=round(TP/(TP+FN),2),
        FAR=round(FP/(TP+FP),2)
      ) }
  else if (climate_indicator_variable=='vci') {
    df<-RAIN_PCODE2 %>%dplyr::filter(SPI==climate_indicator_variable,season==Season_Obs_Rain)%>%dplyr::distinct(Year,.keep_all = TRUE)%>%
      left_join(Impact_df %>% dplyr::select(Year) %>% mutate(drought_events = TRUE,Year=as.factor(as.numeric(Year))), by = "Year") %>%
      dplyr::mutate(drought_events = replace_na(drought_events, FALSE),
                    exceeds_threshold = spi_value < vci_threshold,
                    drought_correct = drought_events & exceeds_threshold,
                    TP = (exceeds_threshold==TRUE) & (drought_events==TRUE),
                    FN = (exceeds_threshold==FALSE) & (drought_events==TRUE),
                    FP = (exceeds_threshold==TRUE) & (drought_events==FALSE),
                    TN = (exceeds_threshold==FALSE) & (drought_events==FALSE)
                    
      ) %>%dplyr::summarise(
        TP=sum(TP,na.rm = TRUE),
        FN=sum(FN,na.rm = TRUE),
        FP=sum(FP,na.rm = TRUE),
        TN=sum(TN,na.rm = TRUE),
        triggered_in_vain = FP,
        triggered_correct = TP,
        #detection_ratio = round(drought_correct / droughts, 2),
        #false_alarm_ratio = round(triggered_in_vain/protocol_triggered, 2),
        POD=round(TP/(TP+FN),2),
        FAR=round(FP/(TP+FP),2)
      )
  }
  else if (climate_indicator_variable=='TAMSAT_sm') {
    df<-RAIN_PCODE2 %>%dplyr::filter(SPI==climate_indicator_variable,season==Season_Obs_Rain)%>%dplyr::distinct(Year,.keep_all = TRUE)%>%
      left_join(Impact_df %>% dplyr::select(Year) %>% mutate(drought_events = TRUE,Year=as.factor(as.numeric(Year))), by = "Year") %>%
      dplyr::mutate(drought_events = replace_na(drought_events, FALSE),
                    exceeds_threshold = spi_value < SM_threshold,
                    drought_correct = drought_events & exceeds_threshold,
                    TP = (exceeds_threshold==TRUE) & (drought_events==TRUE),
                    FN = (exceeds_threshold==FALSE) & (drought_events==TRUE),
                    FP = (exceeds_threshold==TRUE) & (drought_events==FALSE),
                    TN = (exceeds_threshold==FALSE) & (drought_events==FALSE)
                    
      ) %>%dplyr::summarise(
        TP=sum(TP,na.rm = TRUE),
        FN=sum(FN,na.rm = TRUE),
        FP=sum(FP,na.rm = TRUE),
        TN=sum(TN,na.rm = TRUE),
        triggered_in_vain = FP,
        triggered_correct = TP,
        #detection_ratio = round(drought_correct / droughts, 2),
        #false_alarm_ratio = round(triggered_in_vain/protocol_triggered, 2),
        POD=round(TP/(TP+FN),2),
        FAR=round(FP/(TP+FP),2)
      )
  }
  else{
    
    
    
    df<-RAIN_PCODE2 %>%dplyr::filter(SPI==climate_indicator_variable,season==Season_Obs_Rain)%>%dplyr::distinct(Year,.keep_all = TRUE)%>%
      left_join(Impact_df %>% dplyr::select(Year) %>% mutate(drought_events = TRUE,Year=as.factor(as.numeric(Year))), by = "Year") %>%
      dplyr::mutate(drought_events = replace_na(drought_events, FALSE),
                    exceeds_threshold = spi_value < spi_threshold,
                    drought_correct = drought_events & exceeds_threshold,
                    TP = (exceeds_threshold==TRUE) & (drought_events==TRUE),
                    FN = (exceeds_threshold==FALSE) & (drought_events==TRUE),
                    FP = (exceeds_threshold==TRUE) & (drought_events==FALSE),
                    TN = (exceeds_threshold==FALSE) & (drought_events==FALSE)
                    
      ) %>%dplyr::summarise(
        TP=sum(TP),
        FN=sum(FN),
        FP=sum(FP),
        TN=sum(TN),
        triggered_in_vain = FP,
        triggered_correct = TP,
        #detection_ratio = round(drought_correct / droughts, 2),
        #false_alarm_ratio = round(triggered_in_vain/protocol_triggered, 2),
        POD=round(TP/(TP+FN),2),
        FAR=round(FP/(TP+FP),2)
      )
  }
  return(df)
}

####################################

predict_with_rain_skill <- function(impact_hazard_sesonal,
                              SM_threshold,
                              Season_Obs_Rain,
                              vci_threshold,
                              spi_threshold,
                              rain_threshold,
                              climate_indicator_variable,
                              admin3,
                              Level) {
  
  
  
  
  if (climate_indicator_variable=='rain_season'){
    
    df<-impact_hazard_sesonal%>%
      dplyr::filter(SPI==climate_indicator_variable,season==Season_Obs_Rain)%>%
      group_by(ADM2_PCODE)%>%dplyr::distinct(Year,.keep_all = TRUE)%>%
      dplyr::mutate(drought_events = replace_na(drought_events, FALSE),
                    threshold_rain=0.01*rain_season_ave*(100-rain_threshold),
                    exceeds_threshold = spi_value < threshold_rain,
                    drought_correct = (drought_events == exceeds_threshold),
                    TP = exceeds_threshold & (drought_events==TRUE),
                    FN = (exceeds_threshold==FALSE) & (drought_events==TRUE),
                    FP = (exceeds_threshold==TRUE) & (drought_events==FALSE),
                    TN = (exceeds_threshold==FALSE) & (drought_events==FALSE),
                    drought_false_pos = (exceeds_threshold & (drought_events==FALSE)),#FP
                    drought_correct1 = drought_events & exceeds_threshold
      ) %>%dplyr::summarise(
        TP=sum(TP),
        FN=sum(FN),
        FP=sum(FP),
        TN=sum(TN),
        triggered_in_vain = FP,
        triggered_correct = TP,
        #detection_ratio = round(drought_correct / droughts, 2),
        #false_alarm_ratio = round(triggered_in_vain/protocol_triggered, 2),
        POD=round(TP/(TP+FN),2),
        FAR=round(FP/(TP+FP),2),
        CSI=round(TP/(TP+FN+FP),2)
      )%>%dplyr::ungroup()%>%dplyr::select(ADM2_PCODE,POD,FAR,CSI)
    
    df<-admin3%>%dplyr::filter(level==Level)%>%left_join(df,by = 'ADM2_PCODE' )
  }
  
  else if (climate_indicator_variable=='vci') {
    df<-impact_hazard_sesonal%>%
      dplyr::filter(SPI==climate_indicator_variable,season==Season_Obs_Rain)%>%
      group_by(ADM2_PCODE)%>%dplyr::distinct(Year,.keep_all = TRUE)%>%
      dplyr::mutate(drought_events = replace_na(drought_events, FALSE),
                    exceeds_threshold = spi_value < vci_threshold,
                    drought_correct = drought_events & exceeds_threshold,
                    TP = exceeds_threshold & (drought_events==TRUE),
                    FN = (exceeds_threshold==FALSE) & (drought_events==TRUE),
                    FP = (exceeds_threshold==TRUE) & (drought_events==FALSE),
                    TN = (exceeds_threshold==FALSE) & (drought_events==FALSE)
                    
      ) %>%dplyr::summarise(
        TP=sum(TP,na.rm = TRUE),
        FN=sum(FN,na.rm = TRUE),
        FP=sum(FP,na.rm = TRUE),
        TN=sum(TN,na.rm = TRUE),
        triggered_in_vain = FP,
        triggered_correct = TP,
        #detection_ratio = round(drought_correct / droughts, 2),
        #false_alarm_ratio = round(triggered_in_vain/protocol_triggered, 2),
        POD=round(TP/(TP+FN),2),
        FAR=round(FP/(TP+FP),2),
        CSI=round(TP/(TP+FN+FP),2)
      )%>%dplyr::ungroup()%>%dplyr::select(ADM2_PCODE,POD,FAR,CSI)
    
    df<-admin3%>%dplyr::filter(level==Level)%>%left_join(df,by = 'ADM2_PCODE' )
  }
  else if (climate_indicator_variable=='TAMSAT_sm') {
    df<-impact_hazard_sesonal%>%
      dplyr::filter(SPI==climate_indicator_variable,season==Season_Obs_Rain)%>%
      group_by(ADM2_PCODE)%>%dplyr::distinct(Year,.keep_all = TRUE)%>%
      dplyr::mutate(drought_events = replace_na(drought_events, FALSE),
                    exceeds_threshold = spi_value < SM_threshold,
                    drought_correct = drought_events & exceeds_threshold,
                    TP = exceeds_threshold & (drought_events==TRUE),
                    FN = (exceeds_threshold==FALSE) & (drought_events==TRUE),
                    FP = (exceeds_threshold==TRUE) & (drought_events==FALSE),
                    TN = (exceeds_threshold==FALSE) & (drought_events==FALSE)
                    
      ) %>%dplyr::summarise(
        TP=sum(TP,na.rm = TRUE),
        FN=sum(FN,na.rm = TRUE),
        FP=sum(FP,na.rm = TRUE),
        TN=sum(TN,na.rm = TRUE),
        triggered_in_vain = FP,
        triggered_correct = TP,
        #detection_ratio = round(drought_correct / droughts, 2),
        #false_alarm_ratio = round(triggered_in_vain/protocol_triggered, 2),
        POD=round(TP/(TP+FN),2),
        FAR=round(FP/(TP+FP),2),
        CSI=round(TP/(TP+FN+FP),2)
      )%>%dplyr::ungroup()%>%dplyr::select(ADM2_PCODE,POD,FAR,CSI)
    
    df<-admin3%>%dplyr::filter(level==Level)%>%left_join(df,by = 'ADM2_PCODE' )
  }

  else {
    df<-impact_hazard_sesonal%>%
      dplyr::filter(SPI==climate_indicator_variable,season==Season_Obs_Rain)%>%
      group_by(ADM2_PCODE)%>%
      dplyr::mutate(drought_events = replace_na(drought_events, FALSE),
                    exceeds_threshold = spi_value < SM_threshold,
                    drought_correct = drought_events & exceeds_threshold,
                    TP = exceeds_threshold & (drought_events==TRUE),
                    FN = (exceeds_threshold==FALSE) & (drought_events==TRUE),
                    FP = (exceeds_threshold==TRUE) & (drought_events==FALSE),
                    TN = (exceeds_threshold==FALSE) & (drought_events==FALSE)
                    
      ) %>%dplyr::summarise(
        TP=sum(TP,na.rm = TRUE),
        FN=sum(FN,na.rm = TRUE),
        FP=sum(FP,na.rm = TRUE),
        TN=sum(TN,na.rm = TRUE),
        triggered_in_vain = FP,
        triggered_correct = TP,
        #detection_ratio = round(drought_correct / droughts, 2),
        #false_alarm_ratio = round(triggered_in_vain/protocol_triggered, 2),
        POD=round(TP/(TP+FN),2),
        FAR=round(FP/(TP+FP),2),
        CSI=round(TP/(TP+FN+FP),2)
      )%>%dplyr::ungroup()%>%dplyr::select(ADM2_PCODE,POD,FAR,CSI)
    df<-admin3%>%dplyr::filter(level==Level)%>%left_join(df,by = 'ADM2_PCODE' )
  }
  return(df)
}
