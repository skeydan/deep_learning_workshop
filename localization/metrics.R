library(keras)
K <- backend()

# binary crossentropy on just first element of result vector
binary_crossentropy_1elem <- function(ys, yhats) {
  K$mean(K$binary_crossentropy(ys[ ,1], yhats[ ,1]))
}

# intersection over union
iou_single_output <- function(ys, yhats) {
  # remove class code if it's part of ys/yhats (single-output models)
  ys <- ys[ ,2:5] 
  yhats <- yhats[ ,2:5] 
  # reminder: the [xmin, xmax, ymin, ymax] coordinates refer to a coordinate system where (0,0) is on the top left
  intersection_xmin <- K$maximum(ys[ ,1], yhats[ ,1])
  intersection_xmax <- K$minimum(ys[ ,2], yhats[ ,2])
  intersection_ymin <- K$maximum(ys[ ,3], yhats[ ,3])
  intersection_ymax <- K$minimum(ys[ ,4], yhats[ ,4])
  area_intersection <- (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
  area_y <- (ys[ ,2] - ys[ ,1]) * (ys[ ,4] - ys[ ,3])
  area_yhat <- (yhats[ ,2] - yhats[ ,1]) * (yhats[ ,4] - yhats[ ,3])
  area_union <- area_y + area_yhat - area_intersection
  iou <- area_intersection/area_union
  K$mean(iou)
  
}



# Tests -------------------------------------------------------------------

sess <- k_get_session()

# test binary_crossentropy
ys <- K$constant(c(0,99,99,99,99), shape = c(1,5))
yhats <- K$constant(c(1,0,0,0,0), shape = c(1,5))
sess$run(binary_crossentropy_1elem(ys, yhats))

ys <- K$constant(c(0,99,99,99,99,0,99,99,99,99), shape = c(2,5))
yhats <- K$constant(c(1,99,99,99,99,0,99,99,99,99), shape = c(2,5))
ys
sess$run(binary_crossentropy_1elem(ys, yhats))


# K$maximum
ys <- K$constant(c(1,2,3,4), shape = c(2,2))
yhats <- K$constant(c(1.1,1.9,1.7,1.8), shape = c(2,2))
sess$run(K$maximum(ys, yhats))
sess$run(K$maximum(ys[ ,1], yhats[ ,1]))

# test iou
# xmin, xmax, ymin, ymax
ys <- K$constant(c(0,1,3,1,3), shape = c(1,5))
yhats <- K$constant(c(1,2.5,6,2,4), shape = c(1,5))
sess$run(iou_single_output(ys, yhats))

ys <- K$constant(c(0,1,3,1,3,0,1,4,1,4,0,2,1,2), shape = c(3,5))
yhats <- K$constant(c(1,2.5,6,2,4,1,2.5,6,2,4,1,2.5,6,2,4), shape = c(3,5))
sess$run(iou_single_output(ys, yhats))

