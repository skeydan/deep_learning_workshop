library(dplyr)
library(stringr)
library(keras)

data_dir <- "data/train"
annotation_dir <- "annotation/train"
missing_dir <- "annotation_image_missing"

class_ids <- c("n02509815", "n02510455")

for (c in seq_along(class_ids)) { 
  paths <- list.files(file.path(annotation_dir, class_ids[c]), full.names = TRUE)
  print(paste0("Found ", length(paths), " annotation files for class ", c))
  
  for (p in seq_along(paths)) {  
    
    #print(paste0("Processing annotation ", p, ": ", paths[p]))
    text <- readChar(paths[p], n = file.size(paths[p]))
    width <- str_match(text, ".*<width>(.*?)</width>.*")[1,2] %>% as.numeric()
    height <- str_match(text, ".*<height>(.*?)</height>.*")[1,2] %>% as.numeric()
    
    tryCatch({
      img_path <- file.path(data_dir, class_ids[c], str_replace(basename(paths[p]), "xml", "JPEG"))
      img <- image_load(img_path)
      img_array <- image_to_array(img)
      
      # check sizes match 
      dims <- dim(img_array)
      stopifnot(height == dims[1], width == dims[2])
      
    }, error = function(e) {
      print(paste0("Image ", img_path, " not found, moving annotation to missing"))
      file.rename(paths[p], file.path(missing_dir, class_ids[c], basename((paths[p]))))}
    )
      
     
    }
  }


