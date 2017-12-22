library(stringr)
library(dplyr)
library(keras)
library("EBImage")

target_width <- 224
target_height <- 224

class_ids <- c("n02509815", "n02510455")

y_str <- matrix(rep(NA, (556 + 521) * 6), nrow = 556 + 521)

x <- array(rep(NA, (556+ 521) * target_height * target_width *3), dim = c(556 + 521, target_height, target_width, 3))
  

for (c in seq_along(class_ids)) { 
  paths <- list.files(file.path("Annotation", class_ids[c]))
  print(paste0("Found ", length(paths), " annotation files for class ", c))
  
for (p in seq_along(paths)) {  
    
    print(paste0("Processing annotation ", p, ": ", paths[p]))
    suppressWarnings(text <- readLines(file.path("Annotation", class_ids[c], paths[p])))
    text <- Reduce(paste, text)
    width <- str_match(text, ".*<width>(.*?)</width>.*")[1,2] %>% as.numeric()
    height <- str_match(text, ".*<height>(.*?)</height>.*")[1,2] %>% as.numeric()
    xmin <- str_match(text, ".*<xmin>(.*?)</xmin>.*")[1,2] %>% as.numeric()
    xmax <- str_match(text, ".*<xmax>(.*?)</xmax>.*")[1,2] %>% as.numeric()
    ymin <- str_match(text, ".*<ymin>(.*?)</ymin>.*")[1,2] %>% as.numeric()
    ymax <- str_match(text, ".*<ymax>(.*?)</ymax>.*")[1,2] %>% as.numeric()

    tryCatch({
      img <- image_load(file.path("../image_classification/data/train",
                                  class_ids[c],
                                  paste0(str_sub(paths[p], end = -5), ".JPEG")))
      img_array <- image_to_array(img)
      #height, width, channels
      dims <- dim(img_array)
      dims
      
      tryCatch(stopifnot(height == dims[1], width == dims[2]), 
               error = function(e) {
                 print(paste0("Image ", paths[p], " has different dimensions than annotation!"))
               })
      
      # resize all images and bounding boxes to target_width* target_height
   
      img_array <- img_array %>% resize(h = target_height, w = target_width)
      dim(img_array)
      xmin <- xmin * target_width/width
      xmax <- xmax * target_width/width
      ymin <- ymin * target_height/height
      ymax <- ymax * target_height/height
      
      y_str[p, ] <- c(class_ids[c], paths[p],xmin, xmax, ymin, ymax)
      x[p, , , ] <- img_array
      
    }, error = function(e) {
      print(paste0("Image ", str_sub(paths[p], end = -4), "JPEG not found, skipping"))}
    )
    
  
    
  }
  
}

y_str <- na.omit(y_str)
dim(y_str)

dim(x)
