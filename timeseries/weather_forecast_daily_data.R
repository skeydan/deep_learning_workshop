library(dplyr)
library(readr)
library(ggplot2)
library(keras)
library(zoo)
library(rprojroot)
library(lubridate)

setwd(file.path(find_root(criterion = is_rstudio_project), "timeseries"))
source("timeseries_generator.R")


#################################### tbd ###############################################
# handle missing values!!!
########################################################################################

data_dir <- "csv"


# from: http://sklima.de/datenbank.php
fname <- file.path(data_dir, "garmisch_1936_2018.csv")

colnames <- c("day", "avg_temp", "min_temp_5cm", "max_temp", "min_temp", "precip_mm", "precip_ind",
              "snow_total_cm", "pressure_hPa", "humidity_%", "vapor_pressure_hPa", "pressure_NN_hPa", "sunshine_hours", "cloud_8th",
              "avg_wind_ms", "max_wind_ms", "quality", "characterization", "snow_line_m")
coltypes <- "Dddddddddddddddddcd"
garmisch <- read_delim(fname,
                       delim = ";",
                       col_names = colnames,
                       col_types = coltypes,
                       skip = 1,
                       locale = locale(date_format = "%d.%m.%Y"))


#View(garmisch)

garmisch_ts <- zoo(garmisch[c("avg_temp", "precip_mm", "snow_total_cm")],
                   order.by = garmisch$day)

str(garmisch_ts)

#autoplot(garmisch_ts) #+ facet_free() 


# Subset the data (zoom in) -----------------------------------------------

garmisch_2000 <- garmisch %>% filter(day >= ymd("2000-01-01"))
nrow(garmisch_2000)

garmisch_ts_2000 <- zoo(garmisch_2000[c("avg_temp", "precip_mm", "snow_total_cm")],
                   order.by = garmisch_2000$day)
autoplot(garmisch_ts_2000)  

df <- garmisch_2000


# Some more visualization -------------------------------------------------

ggplot(df, aes(x = avg_temp)) + geom_histogram()
ggplot(df, aes(x = pressure_hPa)) + geom_histogram()
ggplot(df, aes(x = snow_line_m)) + geom_histogram()

ggplot(df, aes(x = pressure_hPa, y = precip_mm)) + geom_point()

ggplot(df, aes(x = precip_ind)) + geom_histogram()

ggplot(df, aes(x = avg_temp)) + geom_histogram() + facet_wrap(~ precip_ind)


# Data preprocessing ------------------------------------------------------

data <- data.matrix(df[ , -c(1, 17, 18)])

# scale variables
train_data <- data[1:5000,]
mean <- apply(train_data, 2, mean, na.rm = TRUE)
mean
std <- apply(train_data, 2, sd, na.rm = TRUE)
std
data <- scale(data, center = mean, scale = std)

# data generators
lookback <- 10 # 10 days back
step <- 1 # use every day
delay <- 1 # forecast next day
batch_size <- 32
train_start <- 1
valid_start <- 5001
test_start <- 5801

train_gen <- generator(
  data,
  target_position = 1,
  lookback = lookback,
  delay = delay,
  min_index = train_start,
  max_index = valid_start - 1,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)
val_gen = generator(
  data,
  target_position = 1,
  lookback = lookback,
  delay = delay,
  min_index = valid_start,
  max_index = test_start - 1,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  target_position = 1,
  lookback = lookback,
  delay = delay,
  min_index = test_start,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

train_steps <- (valid_start - 1 - train_start - lookback) / batch_size
val_steps <- (test_start - 1 - valid_start - lookback) / batch_size
test_steps <- (nrow(data) - test_start - lookback) / batch_size


# Baselines ---------------------------------------------------------------

# a common-sense baseline
baseline_mae <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[ ,dim(samples)[[2]],2] # just take the current temperature and predict this for 24 hours later 
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

common_sense <- baseline_mae() #0.39
cat("Common sense MAE: ", common_sense * std[2], " degrees C")


# a simple densely connected network as baseline
model_name <- "MLP"
n_epochs <- 10
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = train_steps,
  epochs = n_epochs,
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history) + geom_hline(yintercept = common_sense, color = "blue") + ggtitle(paste0(model_name, ", ", n_epochs, " epochs"))
model %>% save_model_hdf5(paste0(model_name, "_", n_epochs, "_epochs.hdf5"))
test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.92

