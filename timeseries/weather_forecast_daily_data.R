library(dplyr)
library(readr)
library(ggplot2)
library(keras)
library(zoo)
library(rprojroot)
library(lubridate)
library(stringr)
library(tidyr)
library(purrr)

setwd(file.path(find_root(criterion = is_rstudio_project), "timeseries"))
source("timeseries_generator.R")

data_dir <- "csv"


# from: http://sklima.de/datenbank.php
fname <- file.path(data_dir, "garmisch_1936_2018.csv")

colnames <- c("day", "avg_temp", "min_temp_5cm", "max_temp", "min_temp", "precip_mm", "precip_ind",
              "snow_total_cm", "pressure_hPa", "humidity_percent", "vapor_pressure_hPa", "pressure_NN_hPa", "sunshine_hours", "cloud_8th",
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

autoplot(garmisch_ts) + facet_free() 


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

# perhaps meteorologists could explain...
ggplot(df, aes(x = pressure_hPa, y = precip_mm)) + geom_point()

# no idea what this column might be
ggplot(df, aes(x = precip_ind)) + geom_histogram()
ggplot(df, aes(x = avg_temp)) + geom_histogram() + facet_wrap(~ precip_ind)



# Missing values ----------------------------------------------------------

summary(df)
df %>% summarise_all(funs(num_nas = sum(is.na(.)))) %>%
  gather(key = "variable", value = "num_NA") %>% 
  mutate(variable = str_sub(variable, end = -9)) %>%
  arrange(desc(num_NA))

# fill with last known value
df <- df %>% fill(1:19)
summary(df)


# Data preprocessing ------------------------------------------------------

# let's try to predict precip_mm, keeping as predictors
# avg_temp, max_temp, min_temp, pressure_hPa, humidity_percent, vapor_pressure_hPa,
# pressure_NN_hPa, avg_wind_ms, max_wind_ms
data <- df %>%  select(
  avg_temp, max_temp, min_temp, precip_mm, pressure_hPa, humidity_percent, vapor_pressure_hPa,
  pressure_NN_hPa, avg_wind_ms, max_wind_ms) %>%
  data.matrix()
dim(data)

target_position <- 4

# scale variables
train_data <- data[1:5000,]
mean <- apply(train_data, 2, mean, na.rm = TRUE)
mean
std <- apply(train_data, 2, sd, na.rm = TRUE)
std
data <- scale(data, center = mean, scale = std)

# we will need this later
unscale <- function(vec, mean, sd) {
  vec * sd + mean
}

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
  target_position = target_position,
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
  target_position = target_position,
  lookback = lookback,
  delay = delay,
  min_index = valid_start,
  max_index = test_start - 1,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  target_position = target_position,
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
c(train_steps, val_steps, test_steps)

# Baselines ---------------------------------------------------------------

# a common-sense baseline
baseline_mae <- function(target_position) {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[ ,dim(samples)[[2]],target_position] # just take the current temperature and predict this for 24 hours later 
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  mean(batch_maes)
}

common_sense <- baseline_mae(target_position) 
common_sense # 0.69
cat("Common sense MAE: ", common_sense * std[target_position], " units") # 4.8


# GRU with dropout ----------------------------------------------------

model_name <- "GRU_dropout_garmisch"
n_epochs <- 20
model_file <- paste0(model_name, "_", n_epochs, "_epochs.hdf5")


# dropout: fraction to drop of input
# recurrent_dropout: fraction to drop of recurrent connections (same for every timestep)
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% summary()

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

if (!file.exists(model_file)) {
  
  history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = train_steps,
    epochs = n_epochs,
    validation_data = val_gen,
    validation_steps = val_steps
  )
  
  p <- plot(history) + geom_hline(yintercept = common_sense, color = "cyan") + ggtitle(paste0(model_name, ", ", n_epochs, " epochs"))
  ggsave(str_replace(model_file, "hdf5", "png"), p)
  plot(p)
  model %>% save_model_hdf5(model_file)
  
} else {
  load_model_hdf5(model_file)
}

test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.28
cat(model_name, ": MAE: ", test_loss * std[2], " units")



# Look at predictions -----------------------------------------------------

# one continuous batch of test data
batch_size <- nrow(data) - test_start - lookback - delay
test_gen <- generator(
  data,
  target_position = target_position,
  lookback = lookback,
  delay = delay,
  min_index = test_start,
  max_index = NULL,
  step = step,
  batch_size = batch_size,
  shuffle = FALSE
)

c(samples, targets) %<-% test_gen()
dim(samples)
dim(targets)

batch_preds <- model %>% predict_on_batch(samples)
batch_preds[1:10]

compare_df <- data.frame(actual = c(samples[ , 10, target_position], NA)) %>%
  bind_cols(pred = c(NA, batch_preds))
compare_df[1:10, ]

compare_df <- compare_df %>% map_df(function (vec) unscale(vec, mean[target_position], std[target_position]))
compare_df[1:10, ]
compare_df <- compare_df %>% mutate(ind = row_number())
compare_df %>% gather(key = "key", value="value", -ind) %>% ggplot(aes(x = ind, y = value, color = key)) + geom_line()

compare_ts <- zoo(compare_df[1:2], order.by = compare_df$ind)
compare_ts %>% autoplot()

                         