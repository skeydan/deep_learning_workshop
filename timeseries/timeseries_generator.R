
# target_position - Where in the matrix is the target
# lookback – How many timesteps back the input data should go
# delay – How many timesteps in the future the target should be
# min_index and max_index – Indices in the data array that delimit which timesteps to draw from. This is useful for keeping a segment of the data for validation and another for testing.
# shuffle – Whether to shuffle the samples or draw them in chronological order.
# batch_size – The number of samples per batch.
# step – The period, in timesteps, at which you sample data. You’ll set it 6 in order to draw one data point every hour.

generator <- function(data, target_position = 2, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows),     # sample dimension 
                                lookback / step,  # timesteps
                                dim(data)[[-1]])) # number of variables (features)
    targets <- array(0, dim = c(length(rows)))    # target is one-dimensional
    
    for (j in 1:length(rows)) {
      # create predictors, starting from earliest lookback time to sampled end time, spaced by step size
      # here length.out allows for downsampling so we just get hourly predictors
      indices <- seq(rows[[j]] - lookback, rows[[j]], 
                     length.out = dim(samples)[[2]]) 
      samples[j,,] <- data[indices,]
      # create targets
      targets[[j]] <- data[rows[[j]] + delay, target_position] 
    }            
    
    list(samples, targets)
  }
}
