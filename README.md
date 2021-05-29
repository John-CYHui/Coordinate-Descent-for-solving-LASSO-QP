# Coordinate-Descent-for-solving-LASSO-QP
Coordinate descent algorithm for solving Quadratic optimization with L1 constraint.

<a href="https://www.codecogs.com/eqnedit.php?latex=\arg_{x}&space;\,&space;\min_{}&space;x^{T}Ax&space;&plus;&space;B^{T}x&space;&plus;&space;c&space;&plus;&space;\lambda&space;\left&space;\|x&space;\right&space;\|_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\arg_{x}&space;\,&space;\min_{}&space;x^{T}Ax&space;&plus;&space;B^{T}x&space;&plus;&space;c&space;&plus;&space;\lambda&space;\left&space;\|x&space;\right&space;\|_{1}" title="\arg_{x} \, \min_{} x^{T}Ax + B^{T}x + c + \lambda \left \|x \right \|_{1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=where&space;\;&space;A\in&space;\mathbb{R}^{p\times&space;p}\:&space;\succeq&space;0,&space;\;B&space;\in&space;\mathbb{R}^{p\times1},&space;\;x&space;\in&space;\mathbb{R}^{p\times1},&space;\;c&space;\in&space;\mathbb{R},&space;\;&space;\lambda&space;\in&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?where&space;\;&space;A\in&space;\mathbb{R}^{p\times&space;p}\:&space;\succeq&space;0,&space;\;B&space;\in&space;\mathbb{R}^{p\times1},&space;\;x&space;\in&space;\mathbb{R}^{p\times1},&space;\;c&space;\in&space;\mathbb{R},&space;\;&space;\lambda&space;\in&space;\mathbb{R}" title="where \; A\in \mathbb{R}^{p\times p}\: \succeq 0, \;B \in \mathbb{R}^{p\times1}, \;x \in \mathbb{R}^{p\times1}, \;c \in \mathbb{R}, \; \lambda \in \mathbb{R}" /></a>

The approach for solving this problem is through coordinate descent, which is optimizing one direction at a time. Here, my implementation is walkthrough x from x1 to xp in order. (A faster implementation could possibly be evaluating the gradient of p directions and choose the one with steepest descent).

This project consists of 3 R files.

1. coordinate_descent.R
    This file constis of 4 functions.
    
    1.1 compute_y <- function(A, B, c, lambda, x)
        Returns the value of y based on the quadratic equation given above.
        Parameters: A: array_like
                       p * p dimension input array. Function return false if not positive definite.
                       e.g. A = matrix(c(4,1,0,2), nrow = 2)
                    B: array_like
                       P * 1 dimension input array.
                       e.g. B = matrix(c(-2,-4), nrow = 2)
                    c: scalar
                       e.g. c = 0
                       
    1.2 coord_descent <- function(A, B, c, lambda, init_x0, max_iter = 100, max_dist = 10^-5, plt = 0)
