library(keras)
library(rprojroot)

setwd(file.path(find_root(criterion = is_rstudio_project), "localization"))

source("image_generator.R")
source("plotting_utils.R")
source("metrics.R")

model_exists <- FALSE
model_name <- "cnn_single_output"

yhat_size <- if (length(image_classes) == 2) 1 + 4 else (length(image_classes) + 4)
num_epochs <- 100

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

with_custom_object_scope(c(metric_binary_crossentropy_1elem =  binary_crossentropy_1elem,
                           metric_iou = iou_single_output),
                         {
                           if (!model_exists) {
                             model <- keras_model_sequential()
                             
                             model %>%
                               layer_separable_conv_2d(
                                 filter = 32,
                                 kernel_size = c(3, 3),
                                 padding = "same",
                                 input_shape = c(target_height, target_width, 3)
                               ) %>%
                               layer_activation("relu") %>%
                               
                               layer_separable_conv_2d(filter = 32, kernel_size = c(7, 7)) %>%
                               layer_activation("relu") %>%
                               layer_max_pooling_2d(pool_size = c(2, 2)) %>%
                               layer_dropout(0.25) %>%
                               
                               layer_separable_conv_2d(filter = 64,
                                                       kernel_size = c(9, 9),
                                                       padding = "same") %>%
                               layer_activation("relu") %>%
                               
                               layer_separable_conv_2d(filter = 64, kernel_size = c(11, 11)) %>%
                               layer_activation("relu") %>%
                               layer_max_pooling_2d(pool_size = c(2, 2)) %>%
                               layer_dropout(0.25) %>%
                               
                               layer_flatten() %>%
                               layer_dense(512) %>%
                               layer_activation("relu") %>%
                               layer_dropout(0.25) %>%
                               
                               layer_dense(yhat_size)
                             
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
                                 callback_reduce_lr_on_plateau(patience = 10),
                                 callback_model_checkpoint(filepath = paste0(
                                   model_name,
                                   "-{epoch:02d}-{val_loss:.2f}.hdf5"),
                                   period = 5)))
                                 # ,
                                 # callback_tensorboard(log_dir = "/tmp/tensorboard",
                                 #                     histogram_freq = 5,
                                 #                     write_grads = TRUE,
                                 #                     write_images = TRUE
                                 #                     )
                             
                             plot(history)
                             
                             model %>% save_model_hdf5(paste0(model_name, ".hdf5"))
                          
                           } else {
                             model <- load_model_hdf5(paste0(model_name, ".hdf5"))
                             model %>% summary()
                           }
                           
                           for (g in c(train_gen, valid_gen, test_gen)) {
                             
                             cat(paste0("\n========== Evaluating on batch: ",
                                          attr(g, "name"),
                                          " ==========\n\n"))
                             
                             samples <- g()
                             xs <- samples[[1]]
                             ys <- samples[[2]]
                             yhats <- model %>% predict_on_batch(xs)
                             
                             for (i in 1:(nrow(ys))) {
                               iou <- iou_single_output(K$expand_dims(ys[i, ], axis = -2L), K$expand_dims(yhats[i, ], axis = -2L))
                               iou <- k_get_session()$run(iou)
                               plot_with_boxes(xs[i, , ,], ys[i,], yhats[i,],paste0(attr(g, "name"), ": ", i, "   (IOU: ", round(iou, 2), ")"))
                             }
                             
                             # evaluate class predictions (on one batch)
                             cat("Class probabilities:\n")
                             true_classes <- ys[, 1]
                             class_preds <- yhats[, 1]
                             print(class_preds)
                             class_preds <- ifelse(class_preds > 0.5, 1, 0)
                             cat("\nCross tab:\n")
                             print(table(true_classes, class_preds))
                             
                             # evaluate overall (on one batch)
                             cat("\nMetrics:\n")
                             model %>% test_on_batch(xs, ys) %>% print()
                             
                             # evaluate on complete test set
                             #cat("Evaluation on whole test set:\n")
                             #model %>% evaluate_generator(test_gen, steps = test_total) 
                           }
                           
                         })

