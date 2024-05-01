options(scipen=999)
library(ggplot2)
data(iris)
head(iris)


ggplot(iris, aes(x=Sepal.Length, y=Petal.Length, col=Species))+geom_point()+geom_smooth()

data(mtcars)
library (tidyverse)
glimpse (mtcars)

ggplot(mtcars, aes(x = gear)) +geom_bar()+coord_flip()

ggplot(mtcars, aes(mpg, hp)) + geom_density_2d_filled(show.legend = FALSE) + coord_cartesian(expand = FALSE) + labs(x = "mpg")


library(lattice)
data(iris)
head(iris)

xyplot(Sepal.Length ~ Petal.Length, data = iris,
       group = Species, auto.key = TRUE)
xyplot(Sepal.Length ~ Petal.Length | Species, group = Species,data = iris,
       type = c("p", "smooth"), scales = "free")
cloud(Sepal.Length ~ Sepal.Length*Petal.Width, data = iris,
      group = Species, auto.key = TRUE)

ToothGrowth$dose <- as.factor(ToothGrowth$dose)
bwplot(len ~ dose, data = ToothGrowth,
       xlab = "Dose",
       ylab = "Length")
bwplot(len ~ dose, data = ToothGrowth,
       xlab = "Dose",
       ylab = "Length",
       panel = panel.violin)

histogram(~ len, data = ToothGrowth, breaks = 20)
