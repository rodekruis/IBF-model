library(sf)
library(raster)

# UGA
admin <- st_read("c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-boundaries/UGA_adm2.json")
pop <- raster("c:/Users/BOttow/OneDrive - Rode Kruis/Documenten/Temp_workdir/hrsl_uga_pop_resized_100.tif")
res <- extract(pop, admin, fun = sum, na.rm = T)
df <- data.frame(placeCode = admin$ADM2_PCODE, indicator = "population", value = round(res))
write.csv(df, "c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-area-data/population_UGA.csv",
          row.names = F)

# ZMB
admin <- st_read("c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-boundaries/ZMB_adm2.json")
pop <- raster("c:/Users/BOttow/OneDrive - Rode Kruis/Documenten/Temp_workdir/hrsl_zmb_pop_resized_100.tif")
res <- extract(pop, admin, fun = sum, na.rm = T)
admin1 <- st_read("c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-boundaries/ZMB_adm1.json")
res1 <- extract(pop, admin1, fun = sum, na.rm = T)
admin3 <- st_read("c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-boundaries/ZMB_adm3.json")
res3 <- extract(pop, admin3, fun = sum, na.rm = T)
df <- data.frame(placeCode = c(admin1$ADM1_PCODE, admin$ADM2_PCODE, admin3$ADM3_PCODE),
                 indicator = "population", 
                 value = c(round(res1), round(res), round(res3)))
write.csv(df, "c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-area-data/population_ZMB.csv",
          row.names = F)

# EGY
admin <- st_read("c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-boundaries/EGY_adm1.json")
pop <- raster("c:/Users/BOttow/OneDrive - Rode Kruis/Documenten/Temp_workdir/hrsl_egy_pop_resized_100.tif")
res <- extract(pop, admin, fun = sum, na.rm = T)
df <- data.frame(placeCode = admin$ADM1_PCODE, indicator = "population", value = round(res))
write.csv(df, "c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-area-data/population_EGY.csv",
          row.names = F)

# KEN
admin <- st_read("c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-boundaries/KEN_adm1.json")
pop <- raster("c:/Users/BOttow/OneDrive - Rode Kruis/Documenten/Temp_workdir/hrsl_ken_pop_resized_100.tif")
res <- extract(pop, admin, fun = sum, na.rm = T)
df <- data.frame(placeCode = admin$ADM1_PCODE, indicator = "population", value = round(res))
write.csv(df, "c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-area-data/population_KEN.csv",
          row.names = F)

# ETH
admin <- st_read("c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-boundaries/ETH_adm2.json")
pop <- raster("c:/Users/BOttow/OneDrive - Rode Kruis/Documenten/Temp_workdir/worldpop_eth_resized_100.tif")
res <- extract(pop, admin, fun = sum, na.rm = T)
df <- data.frame(placeCode = admin$ADM2_PCODE, indicator = "population", value = round(res))
write.csv(df, "c:/Users/BOttow/Documents/IBF-system/services/API-service/src/scripts/git-lfs/admin-area-data/population_ETH.csv",
          row.names = F)
