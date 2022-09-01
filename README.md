# Simulated Annealing on Travelling Salesperson Problem

Tested against eil51 and kroA100 instances with best results of 437 and 26672 respectively.

Initial temperature is dynamically set to when 94% of swappings are accepted.

# Swapping Neighbors
Until the temperature reaches 20% of initial temperature, it uses the swapping technique of change a neighbor (each city has 2 diferrent neighbors) to the closest neighbor.

Below 20%, the whole city is swapped with another one randomly.
