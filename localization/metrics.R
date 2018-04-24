library(keras)
library(tensorflow)

# binary crossentropy on just first element of result vector
binary_crossentropy_1elem <- function(ys, yhats) {
  k_mean(k_binary_crossentropy(ys[ ,1], yhats[ ,1]))
}

# intersection over union
iou_single_output <- function(ys_, yhats_) {
  # remove class code if it's part of ys/yhats (single-output models)
  # important: use tf$slice instead of indexing because with indexing, results differ when calling from user code resp. keras
  ys <- tf$slice(ys_, begin = c(0L, 1L), size = c(-1L,4L))
  yhats <- tf$slice(yhats_, begin = c(0L, 1L), size = c(-1L,4L))
  # the order is [xmin, xmax, ymin, ymax]
  # in this coordinate system, (0,0) is on the top left
  intersection_xmin <- k_maximum(ys[ ,1], yhats[ ,1])
  intersection_xmax <- k_minimum(ys[ ,2], yhats[ ,2])
  intersection_ymin <- k_maximum(ys[ ,3], yhats[ ,3])
  intersection_ymax <- k_minimum(ys[ ,4], yhats[ ,4])
  area_intersection <- (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
  area_y <- (ys[ ,2] - ys[ ,1]) * (ys[ ,4] - ys[ ,3])
  area_yhat <- (yhats[ ,2] - yhats[ ,1]) * (yhats[ ,4] - yhats[ ,3])
  area_union <- area_y + area_yhat - area_intersection
  iou <- area_intersection/area_union
  k_mean(iou)
}

# intersection over union
iou_two_outputs <- function(ys, yhats) {
  # the order is [xmin, xmax, ymin, ymax]
  # in this coordinate system, (0,0) is on the top left
  intersection_xmin <- k_maximum(ys[ ,1], yhats[ ,1])
  intersection_xmax <- k_minimum(ys[ ,2], yhats[ ,2])
  intersection_ymin <- k_maximum(ys[ ,3], yhats[ ,3])
  intersection_ymax <- k_minimum(ys[ ,4], yhats[ ,4])
  area_intersection <- (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
  area_y <- (ys[ ,2] - ys[ ,1]) * (ys[ ,4] - ys[ ,3])
  area_yhat <- (yhats[ ,2] - yhats[ ,1]) * (yhats[ ,4] - yhats[ ,3])
  area_union <- area_y + area_yhat - area_intersection
  iou <- area_intersection/area_union
  k_mean(iou)
}




# Tests -------------------------------------------------------------------

sess <- k_get_session()

# test binary_crossentropy
ys <- k_constant(c(0,99,99,99,99), shape = c(1,5))
yhats <- k_constant(c(0.5,0,0,0,0), shape = c(1,5))
# should be same as:
k_binary_crossentropy(k_constant(0), k_constant(0.5)) %>% sess$run()
# also equal to:
- (0 * log(0.5) + (1-0) * log(1 - 0.5))
sess$run(binary_crossentropy_1elem(ys, yhats))

ys <- k_constant(c(0,99,99,99,99,0,99,99,99,99), shape = c(2,5))
yhats <- k_constant(c(1,99,99,99,99,0,99,99,99,99), shape = c(2,5))
ys
sess$run(binary_crossentropy_1elem(ys, yhats))

# K$maximum
ys <- k_constant(c(1,2,3,4), shape = c(2,2))
yhats <- k_constant(c(1.1,1.9,1.7,1.8), shape = c(2,2))
sess$run(k_maximum(ys, yhats))
sess$run(k_maximum(ys[ ,1], yhats[ ,1]))

# test iou
# xmin, xmax, ymin, ymax
ys <- k_constant(c(0,1,3,1,3), shape = c(1,5))
yhats <- k_constant(c(1,2.5,6,2,4), shape = c(1,5))
sess$run(iou_single_output(ys, yhats))

ys <- k_constant(c(0,1,3,1,3,0,1,4,1,4,0,2,1,2), shape = c(3,5))
yhats <- k_constant(c(1,2.5,6,2,4,1,2.5,6,2,4,1,2.5,6,2,4), shape = c(3,5))
sess$run(iou_single_output(ys, yhats))

