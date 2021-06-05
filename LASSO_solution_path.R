library(MASS)
rm(list=ls())
dev.off(dev.list()["RStudioGD"])

source('coordinate_descent.R')


# # Use coordinate descent for real LASSO regression problem
  raw_data = read.csv('diabetes.csv')
  X = data.matrix(raw_data[,1:10])
  Y = data.matrix(raw_data[,11])

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
      p = dim(X_tilta)[2]
    # initialize regression parameters for coordinate descent
      # Guess parameters by normal equation
      init_para = ginv(A) %*% (t(X_tilta) %*% Y)

    # Generate lambda for doing cross validation
      lambda_list = 10^seq(from = -8, to = 2.75, by = 0.01)

      # Initialize regression parameter list for monitoring
        regress_para_list = list()
      # Initialize norm list for monitoring
        regress_para_norm_list = list()
      # Initialize MSE list for monitoring
        mse_list = list()
      for (idx in 1: length(lambda_list))
      {
        lambda = lambda_list[idx]

        # Solve for regression parameters with coordinate descent
          regress_result = coord_descent(A = A, B = B, c = c,
                                 lambda = lambda, init_x0 = init_para,
                                 max_iter = 200, max_dist = 10^-3, plt = 0)
          regress_para = regress_result$min_point
          regress_norm = norm(regress_para,c('1'))

          regress_para_list = cbind(regress_para_list, regress_para)
          regress_para_norm_list = cbind(regress_para_norm_list, regress_norm)

        # Compute MSE
          mse = compute_mse(X_tilta, Y, regress_para)
          mse_list = cbind(mse_list, mse)
      }

    # Plotting LASSO results
    plot(lambda_list, mse_list)
    par(new = F)
    plot(lambda_list, regress_para_norm_list)
    par(new = F)
    for (i in 1:p)
    {
      plot(lambda_list, regress_para_list[i,],
           xlim = range(c(0,max(lambda_list))), ylim = range(c(-0.7,0.7)),
           type = 'l', col = rgb(runif(1),runif(1),runif(1)))
      par(new = T)
    }
    par(new = F)
    for (i in 1:p)
    {
      plot(regress_para_norm_list, regress_para_list[i,],
           xlim = range(c(0,max(as.numeric(regress_para_norm_list)))),
           ylim = range(c(-0.7,0.7)),
           type = 'l', col = rgb(runif(1),runif(1),runif(1)))
      par(new = T)
    }
