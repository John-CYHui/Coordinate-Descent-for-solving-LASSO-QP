library(MASS)
rm(list=ls())
dev.off(dev.list()["RStudioGD"])

source('coordinate_descent.R')

#######################################################################

# # Use coordinate descent for real LASSO regression problem
raw_data = read.csv('diabetes.csv')
X = data.matrix(raw_data[,1:10])
Y = data.matrix(raw_data[,11])

# Scale the data
X = scale(X)
Y = scale(Y)

# Covert LASSO to QP programming problem
    # Append X when column of 1
      n= dim(X)[1]
      
      X_tilta = cbind(replicate(n, 1), X)
      #X_tilta = X
      A = t(X_tilta) %*% X_tilta
      B = -2 * t(X_tilta) %*% Y
      c = norm(Y, '2')^2
      
      n = dim(X_tilta)[1]
      p = dim(X_tilta)[2]

    # initialize regression parameters for coordinate descent
      # Guess parameters by normal equation
      init_para = ginv(A) %*% (t(X_tilta) %*% Y)


# Generate lambda by doing cross validation
  lambda_list = 10^seq(from = -2, to = 2, by = 0.2)
  cv_list = matrix(0, length(lambda_list), 1)
  cv_result = leave_one_out_cv(X_tilta, Y, lambda_list)

  best_lambda = cv_result$best_lambda
  plot(log10(lambda_list), cv_result$CV_list,type = 'l', col = 'red')
  par(new = F)
  
# Use Best lambda to solve for regression parameters with coordinate descent
  regress_result = coord_descent(A = A, B = B, c = c,
                                 lambda = best_lambda, init_x0 = init_para,
                                 max_iter = 200, max_dist = 10^-3, plt = 0)
  
  regress_para = regress_result$min_point
  regress_norm = norm(regress_para,c('1'))


# Compute MSE
  mse = compute_mse(X_tilta, Y, regress_para)
  predictY = X_tilta %*% regress_para
  plot(Y, predictY)