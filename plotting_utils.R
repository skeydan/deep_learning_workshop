library(ggplot2)
library(jpeg)
library(grid)
library(keras)
 
target_width <- 224
target_height <- 224       
        
test_img_path <- "data/train/n02510455/n02510455_1000.JPEG" 

img <- image_load(test_img_path, target_size = c(target_height, target_width))
img_array <- image_to_array(img)/255
img_array

img_raster <- as.raster(img_array)
img_raster

df <- data.frame()

ggplot() + annotation_raster(img_raster, -Inf, Inf, -Inf, Inf)
