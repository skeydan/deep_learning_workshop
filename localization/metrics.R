K <- backend()

# binary crossentropy on just first element of result vector
binary_crossentropy_1elem <- function(ys, yhats) {
  K$binary_crossentropy(ys[ , 1], yhats[ , 1])
}

# test
ys <- K$constant(c(0,0,0,0,0), shape = c(1,5))
yhats <- K$constant(c(1,0,0,0,0), shape = c(1,5)) 
binary_crossentropy_1elem(ys, yhats)
