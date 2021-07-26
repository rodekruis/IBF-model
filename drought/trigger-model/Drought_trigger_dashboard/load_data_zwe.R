
countries <- c("Zimbabwe" = 1)
levels <- c("LEVEL 2" = 2, "LEVEL 3" = 3)



###########3read admin boundaries 
zwe <- sf::read_sf("./shapefiles/zwe/zwe_admbnda_adm2_zimstat_ocha_20180911.shp") %>%
  dplyr::select(ADM1_PCODE, ADM2_PCODE, ADM0_EN)

zwe_lhz <-sf::read_sf("./shapefiles/zwe/ZW_LHZ_2011_fixed.shp") %>%
  dplyr::mutate(ADM1_PCODE=LZCODE, ADM2_PCODE=LZCODE, ADM0_EN=COUNTRY) %>%
  dplyr::select(ADM1_PCODE, ADM2_PCODE, ADM0_EN)

# create a data frame with admin boundaries 

Admin_all <- rbind(zwe,zwe_lhz)
admin <- list(zwe)



Admin_all1 <- Admin_all
st_geometry(Admin_all1) <- NULL 

vci_df <- read.csv("./data/zwe_vci.csv") %>% 
  dplyr::mutate(date = ymd(sprintf("%d%02d01", Year, Month))) %>% 
  dplyr::mutate(ADM0_EN = "zwe") %>%
  dplyr::mutate(vci = VCI_0) %>% 
  dplyr::select(ADM2_PCODE, date, vci) %>%
  left_join(Admin_all1,by='ADM2_PCODE')#%>%filter(ADM0_EN %in% c('KENYA','ETHIOPIA'))

#====================
rain_df_daily<-read.csv("./data/Daily_rainfall.csv")%>%
  dplyr::mutate(date=ymd(date))%>%
  dplyr::mutate(ADM2_PCODE=factor(pcode))%>%
  dplyr::select(ADM2_PCODE,date,rain)%>%
  left_join(Admin_all1,by='ADM2_PCODE')#%>%  filter(ADM0_EN %in% c('KENYA','ETHIOPIA'))
#group_by(ADM1_PCODE,date)%>%  dplyr::summarise(rain=mean(rain),ADM0_EN=first(ADM0_EN))%>%ungroup()%>%filter(ADM0_EN %in% c('Mozambique','Namibia','Lesotho'))
#====================


ipc_zwe <- read.csv("./data/zwe_ipc.csv") %>% mutate(date = ymd(as.Date(Date)))

ipc_data <- rbind(ipc_zwe) 


all_days <- tibble(date = seq(min(ipc_data$date),max(ipc_data$date) , by="months"))

ipc_filled <- as.data.frame(merge(all_days, tibble(ADM2_PCODE = unique(ipc_data$ADM2_PCODE))))%>%
  dplyr::mutate(ADM2_PCODE=factor(ADM2_PCODE),
                month=format(date, '%b'),Year=year(date))%>%
  left_join(ipc_data,by=c('ADM2_PCODE','date'))%>%
  dplyr::group_by(ADM2_PCODE)%>%arrange(date)%>%
  fill(CS, .direction = "downup")%>%
  dplyr::ungroup()%>%left_join(Admin_all1,by='ADM2_PCODE')%>%dplyr::mutate(ipc_class=CS)


ENSO1 <- read.csv("./data/ENSO.csv") %>% gather("MON",'ENSO',-Year)%>% 
  arrange(Year) %>%
  dplyr::mutate(date=seq(as.Date("1950/01/01"), by = "month", length.out = 852))%>%
  filter(date>= as.Date("1980/01/01"))

#Dipole Mode Index (DMI)

IOD <- read.csv("./data/IOD_DMI_standard.csv") %>%
  gather("MON",'IOD',-Year) %>% arrange(Year)%>%
  dplyr::mutate(date=seq(as.Date("1870/01/01"), by = "month", length.out = 1812),IOD=as.numeric(IOD))%>%
  filter(date >= as.Date("1980/01/01"))


SST_var<-ENSO1%>%
  dplyr::select(date,ENSO)%>%
  left_join(IOD%>% dplyr::select(date,IOD),by='date')

all_days <- tibble(date = seq(min(vci_df$date),max(vci_df$date) , by="months"))

df_filled <- merge(all_days, tibble(ADM2_PCODE = unique(vci_df$ADM2_PCODE)))

df_filled <- as.data.frame(df_filled) %>% dplyr::mutate(ADM2_PCODE=factor(ADM2_PCODE))

df_filled2 <- df_filled %>%
  left_join(vci_df%>% dplyr::mutate(ADM2_PCODE=factor(ADM2_PCODE)), by = c("ADM2_PCODE", "date"))



All_df_filled <- df_filled2 %>% dplyr::select(ADM2_PCODE,date,vci) %>%
  dplyr::select(ADM2_PCODE,date,vci) %>% left_join(Admin_all1,by='ADM2_PCODE') %>%
  left_join(SST_var,by='date') %>%
  dplyr::group_by(ADM2_PCODE) %>% arrange(date) %>% 
  fill(vci, .direction = "downup") %>%
  dplyr::ungroup()


Emdat_impact_sff <- read.csv("./data/impact_zwe.csv") %>% 
  dplyr::mutate(total_affected = TtlAffc, no_affected = NAffctd)

zwe_impact <- Emdat_impact_sff



for (n in range(1,length(admin))){
  admin[[n]] <- st_transform(admin[[n]], crs = "+proj=longlat +datum=WGS84")
}


# Clean impact and keep relevant columns
df_impact_raw <- list()

df_impact_raw[[1]] <- zwe_impact %>%
  clean_names() %>% dplyr::mutate(date = dmy(substr(date_event, 1,10)), pcode = adm1_pcode) %>% 
  dplyr::select(pcode, date, total_affected, no_affected) %>%
  unique() %>%  arrange(pcode, date)


df_impact_raw[[10]] <- df_impact_raw[[1]] # admin lvl's 1 and 2


# Used to join against
all_days <- tibble(date = seq(min(c(df_impact_raw[[1]]$date, df_impact_raw[[2]]$date), na.rm=T) - 60,
                              max(c(df_impact_raw[[1]]$date, df_impact_raw[[2]]$date), na.rm=T) + 60, by="months"))



summarize_events <- function(df) {
  df %>% group_by(pcode) %>%  dplyr::summarise(n_events = dplyr::n()) %>% arrange(n_events) %>% ungroup()
}


admin[[1]] <- admin[[1]] %>%
  left_join(summarize_events(df_impact_raw[[1]]) %>%
              dplyr::select(pcode, n_events),   by = c("ADM1_PCODE" = "pcode")) %>%   dplyr::filter(!is.na(n_events))

#change wih livelyhodzone
admin[[10]] <- admin[[1]]

df_indicators <- list()

df_indicators[[1]] <- All_df_filled %>% dplyr::filter( ADM0_EN %in% c("Zimbabwe"))
df_indicators[[10]] <- df_indicators[[1]]

label <- list()
label[[1]] <- "ADM2_PCODE"
label[[10]] <- "ADM2_PCODE"

layerId <- list()
layerId[[1]] <- "ADM2_PCODE"

layerId[[10]] <- "ADM2_PCODE"
