# based on: https://www.manning.com/books/deep-learning-with-r

library(keras)
library(rprojroot)
library(tibble)
library(readr)
library(ggplot2)
library(lubridate)
library(dplyr)


setwd(file.path(find_root(criterion = is_rstudio_project), "timeseries"))
source("timeseries_generator.R")


# Data preprocessing ------------------------------------------------------

data_dir <- "csv"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
df <- read_csv(fname)

glimpse(df)
df <- df %>% mutate(`Date Time` = dmy_hms(`Date Time`))

# clear long-term periodicity
ggplot(df, aes(x = `Date Time`, y = `T (degC)`)) + geom_line()
# but much more chaotic short-term
df %>% filter(`Date Time` < ymd("2009-01-11")) %>% ggplot(aes(x = `Date Time`, y = `T (degC)`)) + geom_line()

data <- data.matrix(df[,-1])

# scale variables
# we use the training set to determine mean and std and then scale the whole dataset accordingly
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

# data generators
# we have 10-minutely data
lookback <- 1440 # 10 days back
step <- 6 # just consider every hour (downsampling)
delay <- 144 # forecast one day later
batch_size <- 128

train_start <- 1
valid_start <- 200000
test_start <- 300001

# the variable we want to predict is in position 2
target_position <- 2

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

# how many times to draw from iterators in 1 epoch
train_steps <- (valid_start - 1 - train_start - lookback) / batch_size
val_steps <- (test_start - 1 - valid_start - lookback) / batch_size
test_steps <- (nrow(data) - test_start - lookback) / batch_size



# Baselines ---------------------------------------------------------------

# a common-sense baseline
baseline_mae <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    # just take the last known temperature and predict this for 24 hours later 
    # dim(samples)[[2]] picks the last known point in time
    preds <- samples[ ,dim(samples)[[2]], target_position] 
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

common_sense <- baseline_mae() #0.28
cat("Common sense MAE: ", common_sense * std[target_position], " degrees C") # 2.5


# a simple densely connected network as baseline
model_name <- "MLP"
n_epochs <- 5
model_file <- paste0(model_name, "_", n_epochs, "_epochs.hdf5")

model <- keras_model_sequential() %>% 
  # as we are not using an RNN, we can't keep the timesteps and have to flatten
  # into <batch_size> * (240 * 14) = <batch_size> * 3360
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  # 1-dimensional output for regression
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
  model <- load_model_hdf5(model_file)
}

test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.92

cat(model_name, ": MAE: ", test_loss * std[target_position], " degrees C")


# Going recurrent: GRU ----------------------------------------------------

model_name <- "GRU"
n_epochs <- 10
model_file <- paste0(model_name, "_", n_epochs, "_epochs.hdf5")

model <- keras_model_sequential() %>% 
  # dimensions: samples, timesteps, features
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
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
  save_model_weights_hdf5(model_file)
}


test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.33
cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")


# Adding dropout ----------------------------------------------------

model_name <- "GRU_dropout"
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
  save_model_hdf5(model_file)
}

test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.28
cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")



# Stacking recurrent layers (GRU) ----------------------------------------

model_name <- "GRU_2layers"
n_epochs <- 10
model_file <- paste0(model_name, "_", n_epochs, "_epochs.hdf5")

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            # we need this if we want to add another layer
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
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
  save_model_hdf5(model_file)
}

test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.28
cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")


# LSTM --------------------------------------------------------------------

model_name <- "LSTM_dropout"
n_epochs <- 20
model_file <- paste0(model_name, "_", n_epochs, "_epochs.hdf5")


model <- keras_model_sequential() %>% 
  layer_lstm(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
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
  save_model_hdf5(model_file)
}

test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.54

cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")


# Stacked LSTM ----------------------------------------

model_name <- "LSTM_2layers"
n_epochs <- 20
model_file <- paste0(model_name, "_", n_epochs, "_epochs.hdf5")


model <- keras_model_sequential() %>% 
  layer_lstm(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_lstm(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
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
  save_model_hdf5(model_file)
}

test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.92
cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")


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

compare_ts <- zoo(compare_df)
compare_ts %>% autoplot()


