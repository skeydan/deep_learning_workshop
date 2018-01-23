library(ggplot2)
library(jpeg)
library(grid)
library(keras)
library(rprojroot)

setwd(file.path(find_root(criterion = is_rstudio_project), "localization"))

source("build_generator.R")
 
target_width <- 224
target_height <- 224     

plot_with_boxes <- function(img, y, yhat = NULL, title = NULL) {
  
  img <- img/255.001
  img <- as.raster(img)
  
  true_box <- if (length(y) == 5) y[2:5] else y
  # transpose to usual cartesian coordinate system
  true_box <- c(true_box[1:2], target_height - true_box[3], target_height - true_box[4])
  # construct dataframe for geom_path
  true_box <- data.frame(xs = c(true_box[1], true_box[1], true_box[2],
                                true_box[2], true_box[1]),
                       ys = c(true_box[4], true_box[3], true_box[3],
                              true_box[4], true_box[4]))
  
  g <- ggplot(true_box, aes(x = xs, y = ys)) +
    annotation_raster(img, 0, target_width, 0, target_height) +
    geom_path(color = "yellow", size = 1) +
    theme(aspect.ratio = 1, axis.ticks = element_blank(), axis.text.x = element_blank(),
          axis.text.y = element_blank()) +
    coord_cartesian(xlim = c(0, target_width), ylim = c(0, target_height))  +
    scale_x_continuous(expand=c(0,0)) +
    scale_y_continuous(expand=c(0,0)) +
    labs(x=NULL, y=NULL)
  
  if (!is.null(yhat)) {
    estimated_box <- if(length(yhat) == 5) yhat[2:5] else yhat
    estimated_box <- c(estimated_box[1:2], target_height - estimated_box[3], target_height - estimated_box[4])
    estimated_box <- data.frame(xs = c(estimated_box[1], estimated_box[1], estimated_box[2],
                                       estimated_box[2], estimated_box[1]),
                                ys = c(estimated_box[4], estimated_box[3], estimated_box[3],
                                       estimated_box[4], estimated_box[4]))
    g <- g + geom_path(data = estimated_box,  aes(x = xs, y = ys), color = "cyan", size = 1)
  }
  
  if (!is.null(title)) {
    g <- g + ggtitle(title)
  }
  print(g)
}


# example usage -----------------------------------------------------------

c(g, size_total) %<-% generator()
c(x, y) %<-% g() 
img <- x[1, , , ]
plot_with_boxes(img, y[1, ])

random_annotations <- c(rbinom(1, 1, 0.5), replicate(4, runif(1,0,224)))
plot_with_boxes(img, y[1, ], random_annotations)



# playground --------------------------------------------------------------

c(g, size_total) %<-% generator()
c(x, y) %<-% g() 
img <- x[1, , , ]
img <- img/255.00001 
img <- as.raster(img)


box <- y[1, 2:5]
box_cartesian <- c(box[1:2], target_height - box[3], target_height - box[4])
box_df <- data.frame(xs = c(box_cartesian[1], box_cartesian[1], box_cartesian[2],
                            box_cartesian[2], box_cartesian[1]),
                     ys = c(box_cartesian[4], box_cartesian[3], box_cartesian[3],
                            box_cartesian[4], box_cartesian[4]))
box_df

ggplot(box_df, aes(x = xs, y = ys)) +
  annotation_raster(img, 0, target_width, 0, target_height) +
  geom_path(color = "yellow", size = 1) + 
  theme(aspect.ratio = 1, axis.ticks = element_blank(), axis.text.x = element_blank(),
        axis.text.y = element_blank()) +
  coord_cartesian(xlim = c(0, target_width), ylim = c(0, target_height))  +
  scale_x_continuous(expand=c(0,0)) +
  scale_y_continuous(expand=c(0,0)) +
  labs(x=NULL, y=NULL)
