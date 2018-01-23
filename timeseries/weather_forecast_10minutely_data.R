# based on: https://www.manning.com/books/deep-learning-with-r

library(keras)
library(rprojroot)
library(tibble)
library(readr)
library(ggplot2)


setwd(file.path(find_root(criterion = is_rstudio_project), "timeseries"))
source("timeseries_generator.R")


# Data preprocessing ------------------------------------------------------


data_dir <- "csv"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
df <- read_csv(fname)

glimpse(df)

# clear long-term periodicity
ggplot(df, aes(x = 1:nrow(df), y = `T (degC)`)) + geom_line()
# but much more chaotic short-term
ggplot(df[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()

data <- data.matrix(df[,-1])

# scale variables
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

# data generators
lookback <- 1440 # 10 days back
step <- 6 # just consider every hour (downsampling)
delay <- 144 # forecast one day later
batch_size <- 128

train_start <- 1
valid_start <- 200000
test_start <- 300001

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
    preds <- samples[,dim(samples)[[2]],2] # just take the current temperature and predict this for 24 hours later 
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

common_sense <- baseline_mae() #0.28
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

cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")


# Going recurrent: GRU ----------------------------------------------------

model_name <- "GRU"
n_epochs <- 20
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
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
test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.33

cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")


# Adding dropout ----------------------------------------------------

model_name <- "GRU_dropout"
n_epochs <- 40

# dropout: fraction to drop of input
# recurrent_dropout: fraction to drop of recurrent connections (same for every timestep)
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
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
test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.28

cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")



# Stacking recurrent layers (GRU) ----------------------------------------

model_name <- "GRU_2layers"
n_epochs <- 20

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
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
test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.28

cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")


# LSTM --------------------------------------------------------------------

model_name <- "LSTM_dropout"
n_epochs <- 40

model <- keras_model_sequential() %>% 
  layer_lstm(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
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
test_loss <- model %>% evaluate_generator(test_gen, steps = test_steps) # 0.54

cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")


# Stacked LSTM ----------------------------------------

model_name <- "LSTM_2layers"
n_epochs <- 20

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

cat(model_name, ": MAE: ", test_loss * std[2], " degrees C")

