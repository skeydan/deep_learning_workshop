library(dplyr)
library(readr)
library(ggplot2)
library(keras)
library(zoo)


colnames <- c("day", "avg_temp", "min_temp_5cm", "max_temp", "min_temp", "precip", "precip_ind",
              "snow_total", "pressure", "humidity", "vapor_pressure", "pressure_red", "sunshine", "cloud",
              "avg_wind", "max_wind", "quality", "characterization", "snow_line")
coltypes <- "Dddddddddddddddddcd"
garmisch <- read_delim("csv/garmisch_1936_2018.csv",
                       delim = ";",
                       col_names = colnames,
                       col_types = coltypes,
                       skip = 1,
                       locale = locale(date_format = "%d.%m.%Y"))


View(garmisch)

garmisch_ts1 <- zoo(garmisch %>% select(c(avg_temp, precip, snow_total)),
                   order.by = garmisch %>% select(day) %>% pull())

str(garmisch_ts1)

autoplot(garmisch_ts1) #+ facet_free() 
