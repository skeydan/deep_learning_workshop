library(keras)

source("build_generator.R")

model_exists <- TRUE
model_name <- "cnn_single_output.hdf5"

yhat_size <- if (length(image_classes) == 2) 1+ 4 else (length(image_classes) + 4)
num_epochs <- 5

train_gen <-  generator()

if (!model_exists) {
  model <- keras_model_sequential()
  
  model %>%
    layer_conv_2d(
      filter = 32, kernel_size = c(3,3), padding = "same", 
      input_shape = c(target_height, target_width, 3)
    ) %>%
    layer_activation("relu") %>%
    layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(0.25) %>%
    
    layer_conv_2d(filter = 32, kernel_size = c(5,5), padding = "same") %>%
    layer_activation("relu") %>%
    layer_conv_2d(filter = 32, kernel_size = c(5,5)) %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(0.25) %>%
    
    layer_flatten() %>%
    layer_dense(512) %>%
    layer_activation("relu") %>%
    layer_dropout(0.5) %>%
    
    layer_dense(yhat_size) 
  
  opt <- optimizer_adam(lr = 0.001)
  
  model %>% compile(
    loss = "mean_squared_error",
    optimizer = opt
  )

  history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = generator_num_items()/batch_size,
    epochs = num_epochs
  )
  
  plot(history)  
  
  model %>% save_model_hdf5(model_name)
  
} else {
  model <- load_model_hdf5(model_name)
  model %>% summary()
}

test_images <- train_gen()
test_images[[2]]
model %>% predict_on_batch(test_images[[1]])

model %>% evaluate_generator(train_gen, steps = generator_num_items()/batch_size)
