library(ncdf4)
library(dplyr)
library(xts)
library(extRemes)


get_ID <- function(file, stations) {
  re <- regexec(".*_([0-9]+)\\.([0-9]+)_([-]*[0-9]+)\\.([0-9]+)_",file)
  l <- attributes(re[[1]])$match.length
  lon <- paste(substr(file,re[[1]][2],re[[1]][2]+l[2] - 1),
               substr(file,re[[1]][3],re[[1]][3]+l[3] - 1), sep = ".") %>% 
    as.numeric()
  lat <- paste(substr(file,re[[1]][4],re[[1]][4]+l[4] - 1),
               substr(file,re[[1]][5],re[[1]][5]+l[5] - 1), sep = ".") %>% 
    as.numeric()
  id <- stations$ID[which(stations$lat == lat & stations$lon == lon)]
  return(id)
}

extract_lead_time <- function(data, lead_time) {
  dis <- ncvar_get(data, "dis24")
  #step <- ncvar_get(data, "step")
  time <- ncvar_get(data, "time")
  number <- ncvar_get(data, "number")
  d <- dim(dis)
  dis5 <- as.data.frame(dis[lead_time, 1:d[2], 1:d[3]])
  names(dis5) <- paste0("ens", number)
  dis5$mean <-  rowMeans(dis5)
  dis5$date <- as.Date("1999-01-01") + time
  return(dis5)
}

calc_return <- function(dis5) {
  discharge <- apply.yearly(xts(dis5$mean, order.by=dis5$date), max)
  RT <- return.level(fevd(discharge, data.frame(discharge), units = "cms"), return.period = c(2, 5, 10, 20, 25), do.ci = FALSE)
  return(as.vector(RT))
}
stations <- read.csv("c:/Users/BOttow/Rode Kruis/510 - Data preparedness and IBF - [CTRY] Uganda/IBF Dashboard data/rp_glofas_station_uga_3_1.csv")
dir <- "c:/Users/BOttow/Rode Kruis/510 - Data preparedness and IBF - [CTRY] Uganda/GIS Data/GloFAS/3.1"
files <- list.files(dir, pattern = ".nc")

id <- get_ID(files[1], stations)
dis5 <- extract_lead_time(nc_open(paste(dir, files[1], sep = "/")), 5)
rt <- calc_return(dis5)
df <- data.frame(matrix(rt,nrow=1))
names(df) <- c("rl2", "rl5", "rl10", "rl20", "rl25")
df$id <- id

for (i in 2:length(files)) {
  id <- get_ID(files[i], stations)
  dis5 <- extract_lead_time(nc_open(paste(dir, files[i], sep = "/")), 5)
  rt <- calc_return(dis5)
  df[i,1:5] <- data.frame(matrix(rt,nrow=1))
  df$id[i] <- id
}

write.csv(df, "c:/Users/BOttow/Rode Kruis/510 - Data preparedness and IBF - [CTRY] Uganda/IBF Dashboard data/glofas_3.1_rt_values.csv", row.names = F)

RT <- return.level(fevd(xts(dis5$mean, order.by=dis5$date), data.frame(xts(dis5$mean, order.by=dis5$date)), units = "cms"), return.period = c(2 * 365, 5 * 365, 10 * 365, 20 * 365, 25 * 365), do.ci = FALSE)
