# Databricks notebook source
# MAGIC %sh
# MAGIC gdal-config --version

# COMMAND ----------

# MAGIC %fs ls dbfs:/mnt/DroughtPipelineV1/shape-files/zwe/zwe_admbnda_adm1_zimstat_ocha_20180911/

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
# MAGIC sudo apt-get update
# MAGIC sudo apt-get install libudunits2-dev libgdal-dev libgeos-dev libproj-dev -y

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("sf")
# MAGIC install.packages("R.utils")
# MAGIC install.packages("exactextractr")

# COMMAND ----------

# MAGIC %r
# MAGIC library(raster)
# MAGIC library(sf)
# MAGIC library(tidyverse)
# MAGIC library(R.utils)
# MAGIC library(exactextractr)
# MAGIC library(stringr)

# COMMAND ----------

# MAGIC %r
# MAGIC path='/dbfs/mnt/DroughtPipelineV1'
# MAGIC 
# MAGIC zwe <- st_read("/dbfs//mnt/DroughtPipelineV1/shape-files/zwe/zwe_admbnda_adm2_zimstat_ocha_20180911/zwe_admbnda_adm2_zimstat_ocha_20180911.shp")     
# MAGIC head(zwe)

# COMMAND ----------

# MAGIC %r
# MAGIC CHIRPS_dir <- '/dbfs//mnt/DroughtPipelineV1/bio-physcial-data/Raw/CHIRPS/'
# MAGIC CHIRPS_files <- list.files(path = CHIRPS_dir, pattern ="*.tif", all.files = FALSE, full.names = FALSE, recursive = TRUE, include.dirs = FALSE)
# MAGIC CHIRPS_files <- CHIRPS_files[!str_detect(CHIRPS_files,pattern=".gz")]  #remove quality flag files 
# MAGIC 
# MAGIC CHIRPS_files

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("rgdal")

# COMMAND ----------

# MAGIC %r
# MAGIC ##stack files per year
# MAGIC #CHIRPS_stack <- do.call("stack",lapply(CHIRPS_files,FUN=function(CHIRPS_files){raster(paste0(CHIRPS_dir,CHIRPS_files))}))
# MAGIC #CHIRPS_stack[DMP_stack < 0] <- NA

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC rainfall_dfs <- list()
# MAGIC 
# MAGIC for (raster_file_path in CHIRPS_files) {
# MAGIC   print(paste0("Calculating raster ", raster_file_path, " time ", Sys.time()))
# MAGIC   file_date <- str_replace_all(gsub(".tif", "", gsub(".*v2.0.", "", raster_file_path)), "\\.", "-")
# MAGIC   raster_file <- raster(paste0(CHIRPS_dir,raster_file_path))
# MAGIC   raster_file[raster_file < 0] <- NA
# MAGIC   mean_rain <- exact_extract(raster_file, zwe, 'mean')
# MAGIC   
# MAGIC   rainfall_df <- tibble(
# MAGIC     pcode = zwe$ADM2_PCODE,
# MAGIC     date = file_date,
# MAGIC     rain = mean_rain
# MAGIC   )
# MAGIC   
# MAGIC   #write.csv(rainfall_df, paste0("raw_data/Ethiopia/rainfall/", file_date, ".csv"), row.names=F)
# MAGIC   
# MAGIC   rainfall_dfs[[file_date]] <- rainfall_df
# MAGIC }

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC # Combining files
# MAGIC all_rainfall <- bind_rows(rainfall_dfs)
# MAGIC head(all_rainfall)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC # Combining files
# MAGIC all_rainfall <- bind_rows(rainfall_dfs)
# MAGIC write.csv(all_rainfall, "/dbfs//mnt/DroughtPipelineV1/bio-physcial-data/Processed/zwe_daily_rainfall.csv", row.names=F)