library(keras)
library(rprojroot)

setwd(file.path(find_root(criterion = is_rstudio_project), "localization"))

source("build_generator.R")
source("plotting_utils.R")
source("metrics.R")

model_exists <- FALSE
model_name <- "resnet_single_output.hdf5"

yhat_size <- if (length(image_classes) == 2) 1 + 4 else (length(image_classes) + 4)
num_epochs <- 100

c(train_gen, train_total) %<-% generator(type = "train", two_outputs = TRUE)
attr(train_gen, "name") <- "train"

c(valid_gen, valid_total) %<-% generator(type = "validation", two_outputs = TRUE)
attr(valid_gen, "name") <- "valid"

c(test_gen, test_total) %<-% generator(type = "test", two_outputs = TRUE)
attr(test_gen, "name") <- "test"

cat(
  paste0(
    "Data sizes overall (train/validation/test): ",
    train_total,
    "/",
    valid_total,
    "/",
    test_total
  )
)


if (!model_exists) {
  conv_base <-
    application_resnet50(include_top = FALSE,
                         input_shape = c(224, 224, 3))
  model <-
    keras_model_sequential() %>% conv_base %>%
    layer_flatten() %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = yhat_size)
  
  model %>% summary()
  
  freeze_weights(conv_base)
  model %>% summary()
  
  opt <- optimizer_adam(lr = 0.001)
  
  model %>% compile(loss = c("binary_crossentropy", "mean_squared_error"),
                    loss_weights = c(0.75, 0.25),
                    optimizer = opt)
  
  history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = train_total / batch_size,
    epochs = num_epochs,
    validation_data = valid_gen,
    validation_steps = valid_total / batch_size,
    callbacks = list(
      callback_early_stopping(patience = 25),
      callback_reduce_lr_on_plateau(patience = 5)
      # ,
      # callback_tensorboard(log_dir = "/tmp/tensorboard",
      #                      histogram_freq = 5,
      #                      write_grads = TRUE,
      #                      write_images = TRUE
      # )
    )
  )
  
  
  plot(history)
  
  model %>% save_model_hdf5(model_name)
  
} else {
  model <- load_model_hdf5(model_name)
  model %>% summary()
}


for (g in c(train_gen, valid_gen, test_gen)) {
  
  print(paste0("Evaluating on batch: ", attr(g, "name")))
  
  samples <- g()
  xs <- samples[[1]]
  ys <- samples[[2]]
  yhats <- model %>% predict_on_batch(xs)
  for (i in 1:(nrow(ys[[2]]))) {
    plot_with_boxes(xs[i, , ,], ys[[2]][i,], yhats[[2]][i,], paste0(attr(g, "name"), ": ", i))
  }
  
  # evaluate class predictions
  true_classes <- ys[[1]]
  class_preds <- yhats[[1]]
  class_preds <- ifelse(class_preds > 0.5, 1, 0)
  print(table(true_classes, class_preds))
  
  model %>% test_on_batch(xs, ys) %>% print()
}
