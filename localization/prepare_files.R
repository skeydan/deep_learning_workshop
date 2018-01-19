library(dplyr)
library(stringr)
library(keras)

setwd(file.path(find_root(criterion = is_rstudio_project), "localization"))

class_ids <- c("n02509815", "n02510455")

# originally, all data are in train directory
start_data_dir <- "data/train"
start_annotation_dir <- "annotation/train"
missing_dir <- "annotation_image_missing"

train_dir <- "data/train"
validation_dir <- "data/validation"
test_dir <- "data/test"
train_dir_annot <- "annotation/train"
validation_dir_annot <- "annotation/validation"
test_dir_annot <- "annotation/test"

create_dirs <- c(validation_dir, test_dir, validation_dir_annot, test_dir_annot)
all_dirs <- c(train_dir, train_dir_annot, validation_dir, validation_dir_annot, test_dir, test_dir_annot)


# create test and validation directories
for (c in seq_along(class_ids))  {
  for (d in create_dirs) {
    subdir <- file.path(d, class_ids[c])
    if (!dir.exists(subdir)) {
      cat(paste0(subdir, " does not exist - creating it"))
      #dir.create(subdir, recursive = TRUE)
    }
  }
}

# check image sizes match sizes in annotation file
# also move all annotations that don't have images to annotation_image_missing directory
for (c in seq_along(class_ids)) { 
  paths <- list.files(file.path(start_annotation_dir, class_ids[c]), full.names = TRUE)
  print(paste0("Found ", length(paths), " annotation files for class ", c))
  
  for (p in seq_along(paths)) {  
    
    #print(paste0("Processing annotation ", p, ": ", paths[p]))
    text <- readChar(paths[p], n = file.size(paths[p]))
    width <- str_match(text, ".*<width>(.*?)</width>.*")[1,2] %>% as.numeric()
    height <- str_match(text, ".*<height>(.*?)</height>.*")[1,2] %>% as.numeric()
    
    tryCatch({
      img_path <- file.path(start_data_dir, class_ids[c], str_replace(basename(paths[p]), "xml", "JPEG"))
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

# move proportion of images to validation and test directories
proportion_test_validation <- 0.1

for (c in seq_along(class_ids)) {
  
  annotation_paths <- list.files(file.path(start_annotation_dir, class_ids[c]), full.names = TRUE)
  annotation_files_to_move <- paths[as.logical(rbinom(length(annotation_paths), 1, proportion_test_validation))]
  
  annotation_files_validation <- annotation_files_to_move[as.logical(rbinom(length(annotation_files_to_move), 1, 0.5))]
  image_files_validation <- str_replace(annotation_files_validation, "annotation", "data") %>%
    str_replace("xml", "JPEG")
  
  annotation_files_test <- setdiff(annotation_files_to_move, annotation_files_validation)  
  image_files_test <- str_replace(annotation_files_test, "annotation", "data") %>%
    str_replace("xml", "JPEG")
  
  for (f in seq_along(annotation_files_validation)) {
    file.rename(annotation_files_validation[f], file.path(validation_dir_annot,  class_ids[c], basename((annotation_files_validation[f]))))
    file.rename(image_files_validation[f], file.path(validation_dir,  class_ids[c], basename((image_files_validation[f]))))
  }
  
  for (f in seq_along(annotation_files_test)) {
    file.rename(annotation_files_test[f], file.path(test_dir_annot,  class_ids[c], basename((annotation_files_test[f]))))
    file.rename(image_files_test[f], file.path(test_dir,  class_ids[c], basename((image_files_test[f]))))
  }
  
}

# how many files do we have now in each directory

cat("Checking number of files: ")
for (c in seq_along(class_ids)) {
  cat(paste0("\n\nClass: ", class_ids[c], "\n\n"))
  for(d in seq_along(all_dirs)) {
    cat(paste0(file.path(all_dirs[d], class_ids[c]), ": ", length(list.files(file.path(all_dirs[d], class_ids[c]))), "\n"))
  }
}
