# kshortestpaths_parallel
PDC Project Spring 2024

## README

This README file provides instructions for compiling and executing the program for finding K shortest paths in graphs using MPI and OpenMP.

### Compilation:

1. Ensure you have MPI and OpenMP installed on your system.
2. Compile the program using the following command:
   ```
   mpicc -fopenmp -o k_shortest_paths k_shortest_paths.c
   ```

### Execution:

1. Prepare a csv file containing the edge information for the graph in the format `(node1name, node2name, weight)`.
2. Run the program using MPI with the following command:
   ```
   mpiexec -n <num_processes> ./k_shortest_paths filename.csv
   ```
   Replace `<num_processes>` with the number of MPI processes you want to use.
   Replace `filename` with the name of the file for the code to be executed on.

### Additional Notes:

- The program will read the `graph.csv` file, preprocess the data, and find the K shortest paths between 10 randomly selected nodes in the graph.
- You can modify the `NUM_RANDOM_NODES` and `K` constants in the code to change the number of random nodes and the number of shortest paths to find, respectively.
- Ensure that your MPI implementation supports the `mpiexec` command and that the necessary environment variables are set correctly.

### Example:

Suppose you have a graph file `graph.csv` with the following contents:

```
node1 node2 1
node2 node1 3
node3 node1 4
node4 node5 6
```

To compile the program and run it with 4 MPI processes:

1. Compile the program:
   ```
   mpicc -fopenmp -o k_shortest_paths k_shortest_paths.c
   ```
2. Run the program:
   ```
   mpiexec -n 4 ./k_shortest_paths graph.csv
   ```

This will execute the program, finding the K shortest paths between randomly selected nodes in the graph and displaying the results. Adjust the number of MPI processes and other parameters as needed for your specific setup.
