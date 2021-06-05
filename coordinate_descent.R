compute_y <- function(A, B, c, lambda, x)
{
  y = t(x) %*% A %*% x + t(B) %*% x + c + lambda * norm(x, '1')
  return(y)
}

coord_descent <- function(A, B, c, lambda, init_x0,
                          max_iter = 100, max_dist = 10^-5, plt = 0)
{
  library(shape)
  #Check if A is positive definite
  eig_A <- eigen(A)$values
  
  for (idx in eig_A)
  {
    if (idx <= 0)
    {
      print('Input matrix A is not positive definite!')
      return()
    }
  }
  nrow = dim(A)[1]
  ncol = dim(A)[2]
  
  prev_y = compute_y(A,B,c, lambda, init_x0)
  x = init_x0
  iter_num = 0
  distance_list = list() # Check if distance to converging to 0
  while (T)
  {
    if (plt == 1) # Draw starting point if plt == 1
    {
      plot(x[1],x[2], xlim = range(c(-3,3)), ylim = range(c(-3,3)))
      start_x_arrow = x[1]
      start_y_arrow = x[2]
      par(new = TRUE)
    }
    
    for (idx in 1:nrow)
    {
      a = x
      a[idx] = 0
      
      if ((B[idx] + (A[idx,] + A[,idx]) %*% a)  < -lambda)
      {
        x[idx] = (-lambda - B[idx] - (A[idx,] + A[,idx]) %*% a) / (A[idx,] + A[,idx])[idx]
      }
      else if ((B[idx] + (A[idx,] + A[,idx]) %*% a) > lambda)
      {
        x[idx] = (lambda - B[idx] - (A[idx,] + A[,idx]) %*% a) / (A[idx,] + A[,idx])[idx]
      }
      else if ((-lambda <= (B[idx] + (A[idx,] + A[,idx]) %*% a)) & ((B[idx] + (A[idx,] + A[,idx]) %*% a) <= lambda))
      {
        x[idx] = 0
      }
      if (plt == 1) # Draw plot if plt == 1
      {
        # Draw the solution path for illustration
        plot(x[1],x[2], xlim = range(c(-3,3)), ylim = range(c(-3,3)))
        par(new = TRUE)
        stop_x_arrow = x[1]
        stop_y_arrow = x[2]
        Arrows(start_x_arrow, start_y_arrow, x1 = stop_x_arrow, 
               y1 = stop_y_arrow, col = 'red', arr.type = 'triangle',
               arr.width = 0.1, arr.length = 0.15)
        start_x_arrow = stop_x_arrow
        start_y_arrow = stop_y_arrow
        par(new = TRUE)
      }
    }
    
    y = compute_y(A,B,c, lambda, x)
    distance = norm((y-prev_y))
    distance_list = append(distance_list, distance)
    prev_y = y
    
    if ((distance <= max_dist) || (iter_num == max_iter))
    {
      result_list = list('min_point' = x, 'Optimize value' = y, 'distance' = distance_list)
        if (plt == 1)
        {
          par(new = F)
          iteration = seq(from = 1, to = length(distance_list), by = 1)
          plot(iteration, distance_list)
        }
        
        if (iter_num == max_iter)
        {
          print('Iteration reach maximum, function return')
        }
      print(paste0('Optimized value = ', y))
      return (result_list)
      break
    }

    iter_num = iter_num + 1
    print(paste0('Iteration ', iter_num,' ,Current y = ', y))
    
  }
}

compute_mse <- function(inputX, inputY, regress_para)
{
  predictY = inputX %*% regress_para
  #plot(inputY, predictY)
  mse = mean((predictY - inputY)^2)
  return(mse)
}

leave_one_out_cv <- function(inputX, inputY, lambda_list)
{
  for (idx in 1: length(lambda_list))
  {
    lambda = lambda_list[idx]
    pred_err = 0
    
    for (j in 1: dim(inputX)[1]) # Leave One Out Cross Validation
    {
      train_X = inputX[-j,]
      train_Y = inputY[-j]
      A_train = t(train_X) %*% train_X
      B_train = -2 * t(train_X) %*% train_Y
      c_train = norm(train_Y, '2')^2
      
      valid_X = inputX[j,]
      valid_y = inputY[j]
      
      # Solve for regression parameters with coordinate descent with Training set
      regress_result = coord_descent(A = A_train, B = B_train, c = c_train,
                                     lambda= lambda, init_x0 = init_para,
                                     max_iter = 200, max_dist = 10^-5, plt = 0)
      # Fitted parameters
      regress_para = regress_result$min_point
      
      # Accumulate prediction error for Validation set for selected lambda
      pred_err = pred_err + compute_mse(valid_X, valid_y, regress_para)
    }
    cv_list[idx] = pred_err / dim(inputX)[1]
  }
  #min_cv_lambda = min(cv_list)
  min_cv_lambda = lambda_list[which.min(cv_list)]
  cv_result = list('best_lambda' = min_cv_lambda, 'CV_list' = cv_list)
  return(cv_result)
}


# # Test case
# A = matrix(c(4,1,0,2), nrow = 2)
# B = matrix(c(-2,-4), nrow = 2)
# 
# c = 0
# 
# init_x0 = matrix(c(3,-3), nrow = 2)
# 
# lambda = 10
# 
# result = coord_descent(A = A, B = B, c = c,
#                        lambda= lambda, init_x0 = init_x0, plt = 1)

