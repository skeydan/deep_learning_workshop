library(keras)
library(reticulate)

test_img_path <- "data/train/n02510455/n02510455_1000.JPEG" 

img <- image_load(test_img_path)
img
img_array <- image_to_array(img)
dim(img_array)                    

img_array <- img_array/255
plot(as.raster(img_array))

plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp=1, col = terrain.colors(12))
}

test_channel <- matrix(c(0,0,0,1,1,
                         0,0,0,1,1,
                         1,1,1,1,1,
                         0,0,0,0,0,
                         1,1,0,0,0),
                       nrow = 5)

image(test_channel, axes = FALSE, asp=1)

plot_channel(test_channel)

