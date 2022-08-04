# Coordinate-Descent-for-solving-LASSO-QP
Coordinate descent algorithm for solving Quadratic optimization with L1 constraint.

$$
arg_x \text{ min }x^TAx+B^Tx+c+\lambda||x||_1
$$

$$
\text{ where }A\in\mathscr{R}^{p*p}\succeq0, B\in\mathscr{R}^{p*1}, x\in\mathscr{R}^{p*1}, c\in\mathscr{R}, \lambda \in\mathscr{R}
$$

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
    \
    \
    \

Convert LASSO to Quadratic Programming problem:

LASSO problem is as follows,    
$$
\displaystyle\sum_{i=1}^{n} \bigg(y_i-\beta_0-\displaystyle\sum_{j=1}^{p}x_{ij}b_j\bigg)^2+\lambda\displaystyle\sum_{j=1}^{p}|b_j|
$$

$$
[Y-\tilde{X}\tilde{\beta}]^T[Y-\tilde{X}\tilde{\beta}]+\lambda||\beta||_1\\\text{where }X\in\mathscr{R}^{n*p}, Y\in\mathscr{R}^{n*1}, \beta\in\mathscr{R}^{p*1};\text
{ modify to } \tilde{X}\in\mathscr{R}^{n*(p+1)}, \tilde{\beta}\in\mathscr{R}^{(p+1)*1}
$$

$$
\tilde{X} = \begin{bmatrix*}
  1 &  \\
  1 &  \\
  . & X\\
  . &  \\
  1 &  \\
 \end{bmatrix*},
 \tilde{\beta} = \begin{bmatrix*}
  \beta_0\\
  \beta\\
 \end{bmatrix*},
$$

$$
[Y - \tilde{X}\tilde{\beta}]^T[Y - \tilde{X}\tilde{\beta}] = ||Y||^2_2+\tilde{\beta}^T(\tilde{X^T}\tilde{X})\tilde{\beta}-2Y^T\tilde{X}\tilde{\beta}
$$

$$
\therefore\text{ min }[Y-\tilde{X}\tilde{\beta}]^T[Y-\tilde{X}\tilde{\beta}]+\lambda||\beta||_1 = \tilde{\beta}^T(\tilde{X}^TX)\tilde{\beta}-2Y^T\tilde{X}\tilde{\beta}+||Y||^2_2+\lambda||{\beta}||_1
$$

Therefore the LASSO problem can be expressed as quadratic programming problem by the following formulation:
$$
\text{min }\tilde{\beta}^T(\tilde{X}^TX)\tilde{\beta}-2Y^T\tilde{X}\tilde{\beta}+||Y||^2_2+\lambda||{\beta}||_1=\text{ min }x^TAx+B^Tx+c+\lambda||x||_1\\\text{where }x = \tilde{\beta}, A = \tilde{X}^T\tilde{X}, B^T=-2Y^T\tilde{X}, c = ||Y||^2_2
$$
Perform coordinate descent on QP problem:  
$$
\text{ min }f(x) = x^TAx+B^Tx+c+\lambda||x||_1
$$

$$
\frac{\mathrm \partial}{\mathrm \partial x} ( x^TAx+B^Tx )=(A+A^T)x+B
$$

$$
\frac{\mathrm \partial}{\mathrm \partial x} ( \lambda||x||_1)=\lambda\displaystyle\sum_{i=1}^{n} =
  \begin{cases}
    1       & \quad \text{if } x_i > 0 \\
    0       & \quad \text{if } x_i = 0 \\
    -1  & \quad \text{if } x_i < 0 
  \end{cases}
$$

$$
\therefore\frac{\mathrm \partial f}{\mathrm \partial x_i} =[i^{th}\text{ row of }A+i^{th}\text{ row of }A^T]x+b_i+\lambda\text{ sign}(x_i)
$$

$$
\text{Set }\frac{\mathrm \partial f}{\mathrm \partial x_i}=0,[i^{th}\text{ row of }A+i^{th}\text{ row of }A^T]x_i + \displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_i+b_i+\lambda\text{ sign}(x_i)=0
$$

$$
\text{Case 1 : if } x_i >0; \text{then }[i^{th}\text{ row of }A+i^{th}\text{ row of }A^T]x_i > 0
$$

$$
\text{Implies}\displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_j + b_i + \lambda < 0
$$

$$
\text{Hence, if }\displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_j + b_i< -\lambda
$$

$$
\text{Then, }x_i = \frac{-\lambda - b_i -\displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_j}{[i^{th}\text{ row of }A+i^{th}\text{ row of }A^T]}
$$


$$
\text{Case 2 : if } x_i <0; \text{then }[i^{th}\text{ row of }A+i^{th}\text{ row of }A^T]x_i < 0
$$

$$
\text{Implies}\displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_j + b_i - \lambda > 0
$$

$$
\text{Hence, if }\displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_j + b_i> \lambda
$$

$$
\text{Then, }x_i = \frac{\lambda - b_i -\displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_j}{[i^{th}\text{ row of }A+i^{th}\text{ row of }A^T]}
$$


$$
\text{Case 3 : if } \displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_j + b_i>= -\lambda \text{ and }\displaystyle\sum_{j\not{=}i}[j^{th}\text{ row of }A+j^{th}\text{ row of }A^T]x_j + b_i<= \lambda
$$

$$
Then, x_i = 0
$$



