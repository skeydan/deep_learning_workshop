library(keras)
library(tensorflow)
library(rprojroot)

setwd(file.path(find_root(criterion = is_rstudio_project), "localization"))

source("image_generator.R")
source("plotting_utils.R")
source("metrics.R")

model_exists <- TRUE
model_name <- "resnet_single_output"

yhat_size <-
  if (length(image_classes) == 2) {
    1 + 4
  } else {
    (length(image_classes) + 4)
  }

num_epochs <- 50

c(train_gen, train_total) %<-% generator(type = "train")
attr(train_gen, "name") <- "train"

c(valid_gen, valid_total) %<-% generator(type = "validation")
attr(valid_gen, "name") <- "valid"

c(test_gen, test_total) %<-% generator(type = "test")
attr(test_gen, "name") <- "test"

cat(
  paste0(
    "Data sizes overall (train/validation/test): ",
    train_total,
    "/",
    valid_total,
    "/",
    test_total,
    '\n'
  )
)

with_custom_object_scope(
  c(
    metric_binary_crossentropy_1elem =  binary_crossentropy_1elem,
    metric_iou = iou_single_output
  ),
  {
    if (!model_exists) {
      
      conv_base <-
        application_resnet50(include_top = FALSE,
                             input_shape = c(224, 224, 3))
      model <- keras_model_sequential() %>% conv_base %>%
        layer_flatten() %>%
        layer_dropout(0.2) %>% 
        layer_dense(units = 512, activation = "relu") %>%
        layer_dropout(0.2) %>%
        layer_dense(units = yhat_size)
      
      model %>% summary()
      
      freeze_weights(conv_base)
      model %>% summary()
      
      opt <- optimizer_adam(lr = 0.001)
      
      model %>% compile(loss = "mean_squared_error",
                        optimizer = opt,
                        metrics = c("metric_binary_crossentropy_1elem", "metric_iou"))
      
      
      history <- model %>% fit_generator(
        train_gen,
        steps_per_epoch = train_total / batch_size,
        epochs = num_epochs,
        validation_data = valid_gen,
        validation_steps = valid_total / batch_size,
        callbacks = list(
          callback_early_stopping(patience = 20),
          callback_reduce_lr_on_plateau(patience = 10)))#,
      #    callback_model_checkpoint(
      #      filepath = paste0(model_name,
      #                        "-{epoch:02d}-{val_loss:.2f}.hdf5"),
      #      period = 5))),
      #    callback_tensorboard(log_dir = "/tmp/tensorboard",
      #                     histogram_freq = 5,
      #                     write_grads = TRUE,
      #                     write_images = TRUE)
      
      plot(history)
      
      model %>% save_model_hdf5(paste0(model_name, ".hdf5"))
      
    } else {
      model <- load_model_hdf5(paste0(model_name, ".hdf5"))
      
    }
    
  }
)

model

for (g in c(train_gen, valid_gen, test_gen)) {
  cat(paste0(
    "\n========== Evaluating on single batch from set: ",
    attr(g, "name"),
    " ==========\n\n"
  ))
  
  samples <- g()
  xs <- samples[[1]]
  dim(xs)
  ys <- samples[[2]]
  dim(ys)
  
  yhats <- model %>% predict_on_batch(xs)
  
  for (i in 1:(nrow(ys))) {
    iou <-
      iou_single_output(ys[i,] %>% k_constant(shape = c(1,5)),
                        yhats[i,] %>% k_constant(shape = c(1,5)))
    iou <- k_get_session()$run(iou)
    plot_with_boxes(xs[i, , , ], ys[i, ], yhats[i, ], paste0(attr(g, "name"), ": ", i, "   (IOU: ", round(iou, 2), ")"))
  }
  
  # evaluate class predictions (on one batch)
  cat("Class probabilities:\n")
  true_classes <- ys[, 1]
  class_preds <- yhats[, 1]
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
