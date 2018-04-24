library(keras)
library(rprojroot)

setwd(file.path(find_root(criterion = is_rstudio_project), "localization"))

source("image_generator.R")
source("plotting_utils.R")
source("metrics.R")

model_exists <- TRUE
model_name <- "cnn_two_outputs"

yhat_size <-
  if (length(image_classes) == 2) {
    1 + 4
  } else {
    (length(image_classes) + 4)
  }

num_epochs <- 50

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
    test_total,
    "\n"
  )
)

if (!model_exists) {
  input <- layer_input(shape = c(target_height, target_width, 3))
  net <- input %>%
    layer_separable_conv_2d(
      filter = 32,
      kernel_size = c(3, 3),
      padding = "same",
      input_shape = c(target_height, target_width, 3)
    ) %>%
    layer_activation("relu") %>%
    
    layer_separable_conv_2d(filter = 32, kernel_size = c(5, 5)) %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(0.4) %>%
    
    layer_separable_conv_2d(filter = 64,
                            kernel_size = c(9, 9),
                            padding = "same") %>%
    layer_activation("relu") %>%
    
    layer_separable_conv_2d(filter = 64, kernel_size = c(13, 13)) %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(0.4) %>%
    
    layer_flatten() %>%
    layer_dense(512) %>%
    layer_activation("relu") %>%
    layer_dropout(0.4)
  
  class_prediction <-
    net %>% layer_dense(1, activation = "sigmoid", name = "class_prediction")
  box_prediction <-
    net %>% layer_dense(4, name = "box_prediction")
  model <-
    keras_model(input, list(class_prediction, box_prediction))
  
  opt <- optimizer_adam(lr = 0.001)
  
  model %>% compile(
    loss = c("binary_crossentropy", "mean_squared_error"),
    # experiment with this!
    loss_weights = c(0.75, 0.25),
    optimizer = opt
  )
  
  history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = train_total / batch_size,
    epochs = num_epochs,
    validation_data = valid_gen,
    validation_steps = valid_total / batch_size,
    callbacks = list(
      callback_early_stopping(patience = 20),
      callback_reduce_lr_on_plateau(patience = 10)
    )
  )
  
  
  plot(history)
  
  model %>% save_model_hdf5(paste0(model_name, ".hdf5"))
  
} else {
  model <- load_model_hdf5(paste0(model_name, ".hdf5"))

}

model %>% summary()

for (g in c(train_gen, valid_gen, test_gen)) {
  cat(paste0(
    "\n========== Evaluating on batch: ",
    attr(g, "name"),
    " ==========\n\n"
  ))
  
  samples <- g()
  xs <- samples[[1]]
  dim(xs)
  ys <- samples[[2]]
  y_classes <- ys[[1]]
  y_boxes <- ys[[2]]
  yhats <- model %>% predict_on_batch(xs)
  yhat_classes <- yhats[[1]][ ,1]
  yhat_boxes <- yhats[[2]]
  
  for (i in 1:(nrow(yhat_boxes))) {
    iou <-
      iou_two_outputs(y_boxes[i,] %>% k_constant(shape = c(1,4)),
                        yhat_boxes[i,] %>% k_constant(shape = c(1,4)))
    iou <- k_get_session()$run(iou)
    plot_with_boxes(xs[i, , , ],
                    y_boxes[i, ],
                    yhat_boxes[i, ],
                    paste0(attr(g, "name"), ": ", i, "   (IOU: ", round(iou, 2), ")"))
  }
  
  # evaluate class predictions (on one batch)
  cat("Class probabilities:\n")
  true_classes <- y_classes
  class_preds <- yhat_classes
  print(class_preds)
  class_preds <-
    ifelse(class_preds > 0.5, 1, 0)
  cat("\nCross tab:\n")
  print(table(true_classes, class_preds))
  
  # evaluate overall (on one batch)
  cat("\nMetrics:\n")
  model %>% test_on_batch(xs, ys) %>% print()

}


# evaluate on complete test set
#cat("Evaluation on whole test set:\n")
#model %>% evaluate_generator(test_gen, steps = test_total)
