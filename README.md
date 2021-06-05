# Coordinate-Descent-for-solving-LASSO-QP
Coordinate descent algorithm for solving Quadratic optimization with L1 constraint.

<a href="https://www.codecogs.com/eqnedit.php?latex=\arg_{x}&space;\,&space;\min_{}&space;x^{T}Ax&space;&plus;&space;B^{T}x&space;&plus;&space;c&space;&plus;&space;\lambda&space;\left&space;\|x&space;\right&space;\|_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\arg_{x}&space;\,&space;\min_{}&space;x^{T}Ax&space;&plus;&space;B^{T}x&space;&plus;&space;c&space;&plus;&space;\lambda&space;\left&space;\|x&space;\right&space;\|_{1}" title="\arg_{x} \, \min_{} x^{T}Ax + B^{T}x + c + \lambda \left \|x \right \|_{1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=where&space;\;&space;A\in&space;\mathbb{R}^{p\times&space;p}\:&space;\succeq&space;0,&space;\;B&space;\in&space;\mathbb{R}^{p\times1},&space;\;x&space;\in&space;\mathbb{R}^{p\times1},&space;\;c&space;\in&space;\mathbb{R},&space;\;&space;\lambda&space;\in&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?where&space;\;&space;A\in&space;\mathbb{R}^{p\times&space;p}\:&space;\succeq&space;0,&space;\;B&space;\in&space;\mathbb{R}^{p\times1},&space;\;x&space;\in&space;\mathbb{R}^{p\times1},&space;\;c&space;\in&space;\mathbb{R},&space;\;&space;\lambda&space;\in&space;\mathbb{R}" title="where \; A\in \mathbb{R}^{p\times p}\: \succeq 0, \;B \in \mathbb{R}^{p\times1}, \;x \in \mathbb{R}^{p\times1}, \;c \in \mathbb{R}, \; \lambda \in \mathbb{R}" /></a>

The approach for solving this problem is through coordinate descent, which is optimizing one direction at a time. Here, my implementation is walkthrough x from x1 to xp in order. (A faster implementation could possibly be evaluating the gradient of p directions and choose the one with steepest descent).

This project consists of 3 R files.

1. coordinate_descent.R  
    This file constis of 4 functions.
    
    **1.1 compute_y <- function(A, B, c, lambda, x)**  
        Returns the value of y based on the quadratic equation given above.  
        
        Parameters: A: array_like  
                       p * p dimension input array.  
                       
                    B: array_like  
                       p * 1 dimension input array.  
                       
                    c: scalar  
                    
                    lambda: scalar  
                    
                    x: array_like  
                       p * 1 dimension input array  
                       
        Returns:    y: scalar
                       
                    Example:
                       A = matrix(c(4,1,0,2), nrow = 2)
                       B = matrix(c(-2,-4), nrow = 2)
                       c = 0
                       lambda = 10  
                       init_x0 = matrix(c(3,-3), nrow = 2)
                       prev_y = compute_y(A,B,c, lambda, init_x0)
                       
    **1.2 coord_descent <- function(A, B, c, lambda, init_x0, max_iter, max_dist, plt)**  
        Perform coordinate descent and return the minimization point x, the optimized value and the L2 distance path.  
        
        Parameters: A: array_like  
                       p * p dimension input array. Function will check and return false if not positive definite.  
                       
                    B: array_like  
                       p * 1 dimension input array.  
                       
                    c: scalar  
                    
                    lambda: scalar  
                    
                    x: array_like  
                       p * 1 dimension input array
                    
                    max_iter: scalar
                       By default, max_iter = 100. Terminate algorithm and return if the number of iteration reaches max_iter
                            
                    max_dist: scalar
                       By default, max_dist = 10^-5. Terminate algorithm and return if the distance moved between iterations is less than max_dist
                   
                    plt: 0 or 1  
                       By default, plt = 0. plt = 1 plots the descent path for 2 dimesional case (Only enable this when p = 2)
                       
        Returns:    result_list: a list consists of 3 results  
                   
                    result_list$min_point : array_like  
                       p * 1 dimension output array. The optimized value for x
                   
                    result_list$Optimize value : scalar
                       The optimization value y.
                   
                    result_list$distance : list
                       A list consists of the distance moved per iteration
                       
                    Example:
                       A = matrix(c(4,1,0,2), nrow = 2)
                       B = matrix(c(-2,-4), nrow = 2)
                       c = 0
                       lambda = 10
                       init_x0 = matrix(c(3,-3), nrow = 2)
                       
                       result = coord_descent(A = A, B = B, c = c, lambda= lambda, init_x0 = init_x0, plt = 1)
                       
    **1.3 compute_mse <- function(inputX, inputY, regress_para)**  
        Calculate mean squared error (MSE) for LASSO problem.      
        
        Parameters: inputX: array_like  
                       n * p dimension input array, where n = the number of examples, p = the number of features.  
                       
                    inputY: array_like  
                       n * 1 dimension input array, the corresponding label for ith row of inputX.  
                    
                    regress_para: array_like  
                       p * 1 dimension input array, the fitted regression parameters from LASSO.  
                       
        Returns:    mse: a scalar
                   
    **1.4 leave_one_out_cv <- function(inputX, inputY, lambda_list)**  
        Perform leave one out cross-validation on quadratic programming with input data X, input Y and a set of given lambda.      
        
        Parameters: inputX: array_like  
                       n * p dimension input array, where n = the number of examples, p = the number of features.  
                       
                    inputY: array_like  
                       n * 1 dimension input array, the corresponding label for ith row of inputX.  
                    
                    regress_para: array_like  
                       k * 1 dimension input array. The number of lambda, k is defined by the user.  
                       
        Returns:    cv_list: a list consists of 2 results
        
                    cv_list$best_lambda : scalar  
                       The best lambda chosen by cross-validation
                   
                    cv_list$CV_list : list
                       A list consists of k * 1 of MSE result from given lambda.

2. LASSO_cross_validation.R  
    This file loads the data set "diabetes.csv" where there is 442 examples and 11 features. The goal is to predict Y using the other 10 features.
    It will convert the LASSO regression problem to QP problem and solve with coordinate descent and leave one out cross-validation.
    The fitted Y is plotted against actual Y.
    


3. LASSO_solution_path.R  
    This file loads the data set "diabetes.csv" where there is 442 examples and 11 features. The goal is to predict Y using the other 10 features.
    It will convert the LASSO regression problem to QP problem and solve with coordinate descent and leave one out cross-validation.
    This file shows LASSO solution path and serve as checking for correct LASSO implementation.
    
