library(stringr)
library(dplyr)
library(keras)
library(purrr)

data_dir <- "data/train"
annotation_dir <- "annotation/train"

class_ids <- c("n02509815", "n02510455")

target_width <- 224
target_height <- 224


generator <- function(batch_size = 10, shuffle = TRUE, classes = class_ids, debug = TRUE) {
  
  annotation_paths <- map(class_ids, function(c) list.files(file.path(annotation_dir, c), full.names = TRUE))
  annotation_paths <- flatten(annotation_paths) %>% unlist()
  num_annotations <- length(annotation_paths)
  
  start_index <- 1
  
  function() {
    
    if (shuffle) {
      annotation_indices <- sample(1:num_annotations, size = batch_size)
    } else {
      annotation_indices <- start_index:(start_index + batch_size -1)
      start_index <<- start_index + 1
    }
    
    if (debug) cat("annotation indices: ", annotation_indices)
    
    annotations <- annotation_paths[annotation_indices]
    
    
    y <- matrix(0, nrow = batchsize, ncol = 5)
    
    x <- array(0, dim = c(batch_size, target_height, target_width, 3))
    
    for (a in seq_along(annotations)) {
      text <- readChar(paths[a], n = file.size(paths[a]))
      width <- str_match(text, ".*<width>(.*?)</width>.*")[1,2] %>% as.numeric()
      height <- str_match(text, ".*<height>(.*?)</height>.*")[1,2] %>% as.numeric()
      xmin <- str_match(text, ".*<xmin>(.*?)</xmin>.*")[1,2] %>% as.numeric()
      xmax <- str_match(text, ".*<xmax>(.*?)</xmax>.*")[1,2] %>% as.numeric()
      ymin <- str_match(text, ".*<ymin>(.*?)</ymin>.*")[1,2] %>% as.numeric()
      ymax <- str_match(text, ".*<ymax>(.*?)</ymax>.*")[1,2] %>% as.numeric()
      
      img_path <- file.path(data_dir, class_ids[c], str_replace(basename(paths[p]), "xml", "JPEG"))
      img <- image_load(img_path)
      img_array <- image_to_array(img)
      dims <- dim(img_array)
      
      # resize all images and bounding boxes to target_width* target_height
      
      img_array <- img_array %>% image_array_resize(height = target_height, width = target_width)
      dim(img_array)
      xmin <- xmin * target_width/width
      xmax <- xmax * target_width/width
      ymin <- ymin * target_height/height
      ymax <- ymax * target_height/height
      
      y_str[p, ] <- c(class_ids[c], paths[p],xmin, xmax, ymin, ymax)
      x[p, , , ] <- img_array
    }
    
  }
}

