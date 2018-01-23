library(dplyr)
library(readr)
library(ggplot2)
library(keras)
library(zoo)
library(rprojroot)

setwd(file.path(find_root(criterion = is_rstudio_project), "timeseries"))
data_dir <- "csv"
fname <- file.path(data_dir, "schneehoehe_kaltenbrunn.csv")

colnames <- c("day", "snow_total")
coltypes <- "Dn"
kaltenbrunn <- read_delim(fname,
                       delim = ";",
                       col_names = colnames,
                       col_types = coltypes,
                       skip = 1,
                       locale = locale(date_format = "%d.%m.%Y", decimal_mark = ","))


View(kaltenbrunn)

kaltenbrunn_ts <- zoo(kaltenbrunn$snow_total, order.by = kaltenbrunn$day)

str(kaltenbrunn_ts)

autoplot(kaltenbrunn_ts) 
