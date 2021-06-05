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
\
\
\

Convert LASSO to Quadratic Programming problem:
  
LASSO problem is as follows,    
<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i&space;=&space;1}^{n}&space;\left&space;(&space;y_{i}&space;-&space;\beta&space;_{0}&space;-&space;\sum_{j&space;=&space;1}^{p}x_{ij}b_{j}&space;\right)^{2}&space;&plus;&space;\lambda&space;\sum_{j&space;=&space;1}^{p}\left&space;|&space;b_{j}\right&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i&space;=&space;1}^{n}&space;\left&space;(&space;y_{i}&space;-&space;\beta&space;_{0}&space;-&space;\sum_{j&space;=&space;1}^{p}x_{ij}b_{j}&space;\right)^{2}&space;&plus;&space;\lambda&space;\sum_{j&space;=&space;1}^{p}\left&space;|&space;b_{j}\right&space;|" title="\sum_{i = 1}^{n} \left ( y_{i} - \beta _{0} - \sum_{j = 1}^{p}x_{ij}b_{j} \right)^{2} + \lambda \sum_{j = 1}^{p}\left | b_{j}\right |" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]^{T}\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]&space;&plus;&space;\lambda&space;\left&space;\|\beta\right&space;\|&space;_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]^{T}\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]&space;&plus;&space;\lambda&space;\left&space;\|\beta\right&space;\|&space;_{1}" title="\left [ Y - \tilde{X} \tilde{\beta } \right ]^{T}\left [ Y - \tilde{X} \tilde{\beta } \right ] + \lambda \left \|\beta\right \| _{1}" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=where&space;\;&space;X\in&space;\mathbb{R}^{n\times&space;p},&space;\;Y&space;\in&space;\mathbb{R}^{n\times1},&space;\beta&space;\in&space;\mathbb{R}^{p\times1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?where&space;\;&space;X\in&space;\mathbb{R}^{n\times&space;p},&space;\;Y&space;\in&space;\mathbb{R}^{n\times1},&space;\beta&space;\in&space;\mathbb{R}^{p\times1}" title="where \; X\in \mathbb{R}^{n\times p}, \;Y \in \mathbb{R}^{n\times1}, \beta \in \mathbb{R}^{p\times1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=modify\;&space;\;to\;\tilde{X}\in&space;\mathbb{R}^{n\times&space;(p&space;&plus;&space;1)},&space;\tilde{\beta}&space;\in&space;\mathbb{R}^{(p&plus;1)\times1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?modify\;&space;\;to\;\tilde{X}\in&space;\mathbb{R}^{n\times&space;(p&space;&plus;&space;1)},&space;\tilde{\beta}&space;\in&space;\mathbb{R}^{(p&plus;1)\times1}" title="modify\; \;to\;\tilde{X}\in \mathbb{R}^{n\times (p + 1)}, \tilde{\beta} \in \mathbb{R}^{(p+1)\times1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tilde{X}&space;=&space;\begin{pmatrix}&space;1&space;&&space;&&space;\\&space;1&space;&&space;&&space;\\&space;.&space;&&space;X&space;&&space;\\&space;.&space;&&space;&&space;\\&space;1&space;&&space;&&space;\\&space;\end{pmatrix},&space;\tilde{\beta&space;}&space;=&space;\begin{pmatrix}&space;\beta_{0}\\&space;\\&space;\beta\\&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\tilde{X}&space;=&space;\begin{pmatrix}&space;1&space;&&space;&&space;\\&space;1&space;&&space;&&space;\\&space;.&space;&&space;X&space;&&space;\\&space;.&space;&&space;&&space;\\&space;1&space;&&space;&&space;\\&space;\end{pmatrix},&space;\tilde{\beta&space;}&space;=&space;\begin{pmatrix}&space;\beta_{0}\\&space;\\&space;\beta\\&space;\end{pmatrix}" title="\tilde{X} = \begin{pmatrix} 1 & & \\ 1 & & \\ . & X & \\ . & & \\ 1 & & \\ \end{pmatrix}, \tilde{\beta } = \begin{pmatrix} \beta_{0}\\ \\ \beta\\ \end{pmatrix}" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta}&space;\right&space;]^{T}\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]&space;=&space;\left&space;\|&space;Y\right&space;\|_{2}^{2}&space;&plus;&space;\tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta}&space;\right&space;]^{T}\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]&space;=&space;\left&space;\|&space;Y\right&space;\|_{2}^{2}&space;&plus;&space;\tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta}" title="\left [ Y - \tilde{X} \tilde{\beta} \right ]^{T}\left [ Y - \tilde{X} \tilde{\beta } \right ] = \left \| Y\right \|_{2}^{2} + \tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\therefore&space;\min_{}\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]^{T}\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]&space;&plus;&space;\lambda&space;\left&space;\|\beta\right&space;\|&space;_{1}&space;=\tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta}&space;&plus;&space;\left&space;\|Y\right&space;\|_{2}^{2}&plus;&space;\lambda&space;\left&space;\|\beta\right&space;\|&space;_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\therefore&space;\min_{}\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]^{T}\left&space;[&space;Y&space;-&space;\tilde{X}&space;\tilde{\beta&space;}&space;\right&space;]&space;&plus;&space;\lambda&space;\left&space;\|\beta\right&space;\|&space;_{1}&space;=\tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta}&space;&plus;&space;\left&space;\|Y\right&space;\|_{2}^{2}&plus;&space;\lambda&space;\left&space;\|\beta\right&space;\|&space;_{1}" title="\small \therefore \min_{}\left [ Y - \tilde{X} \tilde{\beta } \right ]^{T}\left [ Y - \tilde{X} \tilde{\beta } \right ] + \lambda \left \|\beta\right \| _{1} =\tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta} + \left \|Y\right \|_{2}^{2}+ \lambda \left \|\beta\right \| _{1}" /></a>

  
Therefore the LASSO problem can be expressed as quadratic programmming problem by the following formulation:
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\min_{}&space;\tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta}&space;&plus;&space;\left&space;\|Y\right&space;\|_{2}^{2}&plus;&space;\lambda&space;\left&space;\|\beta\right&space;\|&space;_{1}&space;=&space;\min_{}&space;x^{T}Ax&space;&plus;&space;B^{T}x&space;&plus;&space;c&space;&plus;&space;\lambda&space;\left&space;\|x&space;\right&space;\|_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\min_{}&space;\tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta}&space;&plus;&space;\left&space;\|Y\right&space;\|_{2}^{2}&plus;&space;\lambda&space;\left&space;\|\beta\right&space;\|&space;_{1}&space;=&space;\min_{}&space;x^{T}Ax&space;&plus;&space;B^{T}x&space;&plus;&space;c&space;&plus;&space;\lambda&space;\left&space;\|x&space;\right&space;\|_{1}" title="\small \min_{} \tilde{\beta^{T}}(\tilde{X}^{T}\tilde{X})\tilde{\beta}-2Y^{T}\tilde{X}\tilde{\beta} + \left \|Y\right \|_{2}^{2}+ \lambda \left \|\beta\right \| _{1} = \min_{} x^{T}Ax + B^{T}x + c + \lambda \left \|x \right \|_{1}" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=where&space;\;&space;x&space;=&space;\tilde{\beta},\;&space;A&space;=&space;\tilde{X}^{T}\tilde{X},&space;B^{T}&space;=&space;-2Y^{T}\tilde{X},&space;c&space;=&space;\left&space;\|Y\right&space;\|_{2}^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?where&space;\;&space;x&space;=&space;\tilde{\beta},\;&space;A&space;=&space;\tilde{X}^{T}\tilde{X},&space;B^{T}&space;=&space;-2Y^{T}\tilde{X},&space;c&space;=&space;\left&space;\|Y\right&space;\|_{2}^{2}" title="where \; x = \tilde{\beta},\; A = \tilde{X}^{T}\tilde{X}, B^{T} = -2Y^{T}\tilde{X}, c = \left \|Y\right \|_{2}^{2}" /></a>
\
\
\
Perform coordinate descent on QP problem:  
\
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\min_{}&space;\;&space;f(x)&space;=&space;x^{T}Ax&space;&plus;&space;B^{T}x&space;&plus;&space;c&space;&plus;&space;\lambda&space;\left&space;\|x&space;\right&space;\|_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\min_{}&space;\;&space;f(x)&space;=&space;x^{T}Ax&space;&plus;&space;B^{T}x&space;&plus;&space;c&space;&plus;&space;\lambda&space;\left&space;\|x&space;\right&space;\|_{1}" title="\small \min_{} \; f(x) = x^{T}Ax + B^{T}x + c + \lambda \left \|x \right \|_{1}" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\frac{\partial}{\partial&space;x}&space;(x^{T}Ax&space;&plus;&space;B^{T}x)&space;=&space;(A&space;&plus;&space;A^{T})x&space;&plus;&space;B" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\frac{\partial}{\partial&space;x}&space;(x^{T}Ax&space;&plus;&space;B^{T}x)&space;=&space;(A&space;&plus;&space;A^{T})x&space;&plus;&space;B" title="\small \frac{\partial}{\partial x} (x^{T}Ax + B^{T}x) = (A + A^{T})x + B" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\frac{\partial}{\partial&space;x}&space;(\lambda&space;\left&space;\|&space;x&space;\right&space;\|_{1})&space;=&space;\lambda&space;\sum_{i&space;=&space;1}^{n}\begin{cases}&space;1&space;&&space;\text{&space;if&space;}&space;x_{i}&space;>&space;0&space;\\&space;0&space;&&space;\text{&space;if&space;}&space;x_{i}&space;=&space;0\\&space;-1&space;&&space;\text{&space;if&space;}&space;x_{i}<0&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\frac{\partial}{\partial&space;x}&space;(\lambda&space;\left&space;\|&space;x&space;\right&space;\|_{1})&space;=&space;\lambda&space;\sum_{i&space;=&space;1}^{n}\begin{cases}&space;1&space;&&space;\text{&space;if&space;}&space;x_{i}&space;>&space;0&space;\\&space;0&space;&&space;\text{&space;if&space;}&space;x_{i}&space;=&space;0\\&space;-1&space;&&space;\text{&space;if&space;}&space;x_{i}<0&space;\end{cases}" title="\small \frac{\partial}{\partial x} (\lambda \left \| x \right \|_{1}) = \lambda \sum_{i = 1}^{n}\begin{cases} 1 & \text{ if } x_{i} > 0 \\ 0 & \text{ if } x_{i} = 0\\ -1 & \text{ if } x_{i}<0 \end{cases}" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\therefore&space;\frac{\partial&space;f}{\partial&space;x_{i}}&space;=&space;\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]x&space;&plus;&space;b_{i}&space;&plus;&space;\lambda\;sign\;{x_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\therefore&space;\frac{\partial&space;f}{\partial&space;x_{i}}&space;=&space;\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]x&space;&plus;&space;b_{i}&space;&plus;&space;\lambda\;sign\;{x_{i}}" title="\small \therefore \frac{\partial f}{\partial x_{i}} = \left [ i^{th} \;row\;of\;A + i^{th}\;row\;of\;A^{T} \right ]x + b_{i} + \lambda\;sign\;{x_{i}}" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;Set\;\frac{\partial&space;f}{\partial&space;x_{i}}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;Set\;\frac{\partial&space;f}{\partial&space;x_{i}}&space;=&space;0" title="\small Set\;\frac{\partial f}{\partial x_{i}} = 0" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]x_{i}&space;&plus;&space;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{i}&plus;&space;b_{i}&space;&plus;&space;\lambda\;sign\;{x_{i}}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]x_{i}&space;&plus;&space;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{i}&plus;&space;b_{i}&space;&plus;&space;\lambda\;sign\;{x_{i}}&space;=&space;0" title="\left [ i^{th} \;row\;of\;A + i^{th}\;row\;of\;A^{T} \right ]x_{i} + \sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{i}+ b_{i} + \lambda\;sign\;{x_{i}} = 0" /></a>
\
\
\
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;Case\;1:&space;\;&space;if\;x_{i}&space;>&space;0;\;then\;\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]x_{i}&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;Case\;1:&space;\;&space;if\;x_{i}&space;>&space;0;\;then\;\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]x_{i}&space;>&space;0" title="\small Case\;1: \; if\;x_{i} > 0;\;then\;\left [ i^{th} \;row\;of\;A + i^{th}\;row\;of\;A^{T} \right ]x_{i} > 0" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex==>\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}&space;&plus;&space;\lambda&space;<&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=>\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}&space;&plus;&space;\lambda&space;<&space;0" title="=>\sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{j}+ b_{i} + \lambda < 0" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex==>if&space;\;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}&space;<&space;-&space;\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=>if&space;\;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}&space;<&space;-&space;\lambda" title="=>if \;\sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{j}+ b_{i} < - \lambda" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex==>&space;Then\;\;x_{i}&space;=&space;\frac{-&space;\lambda\&space;-b_{i}-\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}}{\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=>&space;Then\;\;x_{i}&space;=&space;\frac{-&space;\lambda\&space;-b_{i}-\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}}{\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]}" title="=> Then\;\;x_{i} = \frac{- \lambda\ -b_{i}-\sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{j}}{\left [ i^{th} \;row\;of\;A + i^{th}\;row\;of\;A^{T} \right ]}" /></a>
\
\
\
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;Case\;2:&space;\;&space;if\;x_{i}&space;<&space;0;\;then\;\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]x_{i}&space;<&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;Case\;2:&space;\;&space;if\;x_{i}&space;<&space;0;\;then\;\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]x_{i}&space;<&space;0" title="\small Case\;2: \; if\;x_{i} < 0;\;then\;\left [ i^{th} \;row\;of\;A + i^{th}\;row\;of\;A^{T} \right ]x_{i} < 0" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex==>\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}&space;-&space;\lambda&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=>\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}&space;-&space;\lambda&space;>&space;0" title="=>\sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{j}+ b_{i} - \lambda > 0" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex==>if&space;\;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}&space;>&space;\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=>if&space;\;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}&space;>&space;\lambda" title="=>if \;\sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{j}+ b_{i} > \lambda" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex==>&space;Then\;\;x_{i}&space;=&space;\frac{\lambda\&space;-b_{i}-\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}}{\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=>&space;Then\;\;x_{i}&space;=&space;\frac{\lambda\&space;-b_{i}-\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}}{\left&space;[&space;i^{th}&space;\;row\;of\;A&space;&plus;&space;i^{th}\;row\;of\;A^{T}&space;\right&space;]}" title="=> Then\;\;x_{i} = \frac{\lambda\ -b_{i}-\sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{j}}{\left [ i^{th} \;row\;of\;A + i^{th}\;row\;of\;A^{T} \right ]}" /></a>
\
\
\
<a href="https://www.codecogs.com/eqnedit.php?latex=Case\;3:&space;\;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}>=&space;-&space;\lambda&space;\;\;and\;\;&space;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}<=&space;\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Case\;3:&space;\;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}>=&space;-&space;\lambda&space;\;\;and\;\;&space;\sum_{j\neq&space;i}^{}\left&space;[&space;j^{th}&space;\;row\;of\;A&space;&plus;&space;j^{th}\;row\;of\;A^{T}&space;\right&space;]x_{j}&plus;&space;b_{i}<=&space;\lambda" title="Case\;3: \;\sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{j}+ b_{i}>= - \lambda \;\;and\;\; \sum_{j\neq i}^{}\left [ j^{th} \;row\;of\;A + j^{th}\;row\;of\;A^{T} \right ]x_{j}+ b_{i}<= \lambda" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=Then&space;\;\;x_{i}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Then&space;\;\;x_{i}&space;=&space;0" title="Then \;\;x_{i} = 0" /></a>
  
