library(stringr)
library(dplyr)
library(keras)
library(purrr)

train_dir <- "data/train"
validation_dir <- "data/validation"
test_dir <- "data/test"
train_dir_annot <- "annotation/train"
validation_dir_annot <- "annotation/validation"
test_dir_annot <- "annotation/test"

image_classes <- c("n02509815", "n02510455")

target_width <- 224
target_height <- 224


###

batch_size <- 10
class_ids <- image_classes

generator <- function(type = "train", two_outputs = FALSE, debug = FALSE, inception = FALSE) {
  
  c(annotation_dir, image_dir) %<-% if(type == "train") {
    list(train_dir_annot, train_dir)
  } else if (type == "validation") {
    list(validation_dir_annot, validation_dir)
  } else if (type == "test") {
    list(test_dir_annot, test_dir)
  }
  
  all_annotations <- sapply(class_ids, function(c) list.files(file.path(annotation_dir, c), full.names = TRUE))
  all_annotations <- flatten(all_annotations) %>% unlist()
  num_annotations <- length(all_annotations)
  
  gen <- function() {
    
    sampled_indices <- sample(1:num_annotations, size = batch_size)
    sampled_annotations <- all_annotations[sampled_indices]
    sampled_classes <- map(sampled_annotations, compose(basename, dirname)) %>% 
      unlist()
    sampled_class_levels <- sampled_classes %>% as.factor() %>% as.numeric() - 1
    sampled_images <-  file.path(image_dir, sampled_classes, str_replace(basename(sampled_annotations), "xml", "JPEG"))
    
    
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
 
      # resize all images and bounding boxes to target_width* target_height
      img_array <- image_load(sampled_images[i]) %>% image_to_array()
      img_array <- img_array %>% image_array_resize(height = target_height, width = target_width)
      dim(img_array)
      xmin <- xmin * target_width/width
      xmax <- xmax * target_width/width
      ymin <- ymin * target_height/height
      ymax <- ymax * target_height/height
      
      # if(inception == TRUE) {
      #     img_array <- img_array / 127.5
      #     img_array <- img_array - 1
      # }
      
      y[i, ] <- c(sampled_class_levels[i], xmin, xmax, ymin, ymax)
      x[i, , , ] <- img_array
    }
    if (!two_outputs) {
      list(x,y)
    } else {
      list(x, list(y[ ,1], y[ ,2:4]))
    }
   
  }
  
  list(gen, num_annotations)
}

# c(g, size_total) %<-% generator(debug = TRUE)
# c(x, y) %<-% g()
# dim(x)
# dim(y)
# size_total
#   
# c(g, size_total) %<-% generator(type = "validation", debug = TRUE)
# c(x, y) %<-% g()
# dim(x)
# dim(y)
# size_total
# 
# c(g, size_total) %<-% generator(type = "test", debug = TRUE)
# c(x, y) %<-% g()
# dim(x)
# dim(y)
# size_total

