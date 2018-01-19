library(keras)
library(rprojroot)

setwd(file.path(find_root(criterion = is_rstudio_project), "localization"))

source("build_generator.R")
source("plotting_utils.R")
source("metrics.R")

model_exists <- FALSE
model_name <- "resnet_single_output.hdf5"

yhat_size <- if (length(image_classes) == 2) 1 + 4 else (length(image_classes) + 4)
num_epochs <- 1

c(train_gen, train_total) %<-% generator(type = "train")
c(valid_gen, valid_total) %<-% generator(type = "validation")
c(test_gen, test_total) %<-% generator(type = "test")

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

with_custom_object_scope(c(metric_binary_crossentropy_1elem =  binary_crossentropy_1elem),
                         {
                           if (!model_exists) {
                             conv_base <-
                               application_resnet50(include_top = FALSE,
                                                     input_shape = c(224, 224, 3))
                             model <- keras_model_sequential() %>% conv_base %>%
                               layer_flatten() %>%
                               layer_dense(units = 32, activation = "relu") %>%
                               layer_dense(units = yhat_size)
                             
                             model %>% summary()
                             
                             freeze_weights(conv_base)
                             model %>% summary()
                             
                             opt <- optimizer_adam(lr = 0.001)
                             
                             model %>% compile(loss = "mean_squared_error",
                                               optimizer = opt,
                                               metrics = "metric_binary_crossentropy_1elem")
                             
                             history <- model %>% fit_generator(
                               train_gen,
                               steps_per_epoch = train_total / batch_size,
                               epochs = num_epochs,
                               validation_data = valid_gen,
                               validation_steps = valid_total / batch_size,
                               callbacks = list(
                                 callback_early_stopping(patience = 3),
                                 callback_reduce_lr_on_plateau(patience = 2)
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
                         })

test_images <- test_gen()
xs <- test_images[[1]]
ys <- test_images[[2]]
yhats <- model %>% predict_on_batch(xs)
for (i in 1:(nrow(ys))) {
  plot_with_boxes(xs[i, , ,], ys[i,], yhats[i,])
}

# evaluate class predictions
true_classes <- ys[, 1]
class_preds <- yhats[, 1]
class_preds
class_preds <- ifelse(class_preds > 0.5, 1, 0)
class_preds
table(true_classes, class_preds)

# tbd
# define metric intersection over union

model %>% evaluate_generator(test_gen, steps = test_total)
