library(stringr)
library(dplyr)
library(keras)
library(purrr)

data_dir <- "data/train"
annotation_dir <- "annotation/train"

image_classes <- c("n02509815", "n02510455")

target_width <- 224
target_height <- 224

batch_size <- 10
class_ids <- image_classes

debug <- FALSE

generator_num_items <- function() {
  all_annotations <- sapply(class_ids, function(c) list.files(file.path(annotation_dir, c), full.names = TRUE))
  length(all_annotations %>% flatten())
}
  
generator <- function() {
  
  all_annotations <- sapply(class_ids, function(c) list.files(file.path(annotation_dir, c), full.names = TRUE))
  all_annotations <- flatten(all_annotations) %>% unlist()
  num_annotations <- length(all_annotations)
  
  function() {
    
    sampled_indices <- sample(1:num_annotations, size = batch_size)
    sampled_annotations <- all_annotations[sampled_indices]
    sampled_classes <- map(sampled_annotations, compose(basename, dirname)) %>% 
      unlist()
    sampled_class_levels <- sampled_classes %>% as.factor() %>% as.numeric() - 1
    sampled_images <-  file.path(data_dir, sampled_classes, str_replace(basename(sampled_annotations), "xml", "JPEG"))
    
    
    if (debug) cat("\nsampled indices: \n", sampled_indices)
    if (debug) cat("\nannotation files: \n", sampled_annotations)
    if (debug) cat("\nclass membership: \n", sampled_classes)
    if (debug) cat("\nclass codes: \n", sampled_class_levels)
    if (debug) cat("\nimage files: \n", sampled_images)
    
    y <- matrix(0, nrow = batch_size, ncol = 5)
    x <- array(0, dim = c(batch_size, target_height, target_width, 3))
    
    for (i in seq_along(sampled_annotations)) {
      text <- readChar(sampled_annotations[i], n = file.size(sampled_annotations[i]))
      width <- str_match(text, ".*<width>(.*?)</width>.*")[1,2] %>% as.numeric()
      height <- str_match(text, ".*<height>(.*?)</height>.*")[1,2] %>% as.numeric()
      xmin <- str_match(text, ".*<xmin>(.*?)</xmin>.*")[1,2] %>% as.numeric()
      xmax <- str_match(text, ".*<xmax>(.*?)</xmax>.*")[1,2] %>% as.numeric()
      ymin <- str_match(text, ".*<ymin>(.*?)</ymin>.*")[1,2] %>% as.numeric()
      ymax <- str_match(text, ".*<ymax>(.*?)</ymax>.*")[1,2] %>% as.numeric()
      
      img <- image_load(sampled_images[i])
      img_array <- image_to_array(img)
      dims <- dim(img_array)
      
      # resize all images and bounding boxes to target_width* target_height
      img_array <- image_load(sampled_images[i]) %>% image_to_array()
      img_array <- img_array %>% image_array_resize(height = target_height, width = target_width)
      dim(img_array)
      xmin <- xmin * target_width/width
      xmax <- xmax * target_width/width
      ymin <- ymin * target_height/height
      ymax <- ymax * target_height/height
      
      y[i, ] <- c(sampled_class_levels[i], xmin, xmax, ymin, ymax)
      x[i, , , ] <- img_array
    }
    list(x,y)
  }
}

#g <- generator()
#c(x,y) %<-% g()
#dim(x)
#dim(y)
