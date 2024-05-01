#variable value assignment and arithmetic
x <- 3
print(x + 1)
5 -> y
print(x*y)
sqrt(y-1)
cos(x+1)

# Objects in R: Vectors, Lists, Matrices, Data Frames 
apple <- c('red','green',"yellow") 
print(apple) 
print(class(apple))

k <- list(1:3, TRUE, "Hello", list(1:2, 5)) 
k[[3]] 

matrix(1:12, nrow=3, ncol=4) 

emp.data <- data.frame( 
  emp_id = c (1:5), 
  emp_name = 
    c("Rick","Dan","Michelle","Ryan","Gary"),salary = 
    c(623.3,515.2,611.0,729.0,843.25), 
  start_date = as.Date(c("2012-01-01", "2013-09-23", "2014-11-15", "2014-05-11", "2015-03-27")), 
  stringsAsFactors = FALSE 
) 
print(summary(emp.data))

#Control Flow Statements: If else and For Loop
x <- c(2,5,3,9,8,11,6) 
count <- 0 
for (val in x) {
  
  if(val %% 2 == 0) count = count+1 
} 
print(count)

#Visualization: Barplot and histogram
A <- c(17, 32, 8, 53, 1) 
barplot(A, xlab = "X-axis", ylab = "Y-axis", main ="Bar-Chart") 

v <- c(19, 23, 11, 5, 16, 21, 32, 14, 19, 27, 39) 
hist(v, xlab = "No.of Articles ",col = "green", border = "black") 
