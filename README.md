# Optimization-With-Recursion
This repo contains code for building a simple feed-forward neural network from scratch.

This code should look very familiar to those of you who have taken the deep learning specialization course as taught by Andrew Ng.
The only major difference is that this implementation uses recursion in lieu of for loops to optimize the NN. This approach allows
us to use the call stack as a cache, thus eliminating the need for a (dict, list, tuple) cache.

The actual code for doing this is within the "forward_and_backward_propagate" function - the actual recursive function.
