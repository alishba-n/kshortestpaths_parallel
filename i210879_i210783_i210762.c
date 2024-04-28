#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define INF INT_MAX //inf variable to store int_max
#define NUM_RANDOM_NODES 10 //random nodes
#define MAX_NODES 8000

//edge to store source name, target name and weight of thr edge
typedef struct {
    int weight;
    char source[100];
    char target[100];
} Edge;

//names of the node with unique numbers
typedef struct {
    int number; 
    char name[100];
} Node;

//preprocess the csv file
void preprocess_file(char *filename, int*** adjacency_matrix, int* num_nodes)
{
    Edge edges[MAX_NODES];
    Node nodes[MAX_NODES];
    
    char line[100];
    int num_edges = 0;
    int unique_nodes = 0; //number of unique nodes
    
    //open the input file
    FILE* fp = fopen(filename, "r");
    if (fp == NULL)
    {
        perror("Error opening input file");
        return;
    }
    
    //open the output file
    char* outputfile = "output.txt";
    FILE* op = fopen(outputfile, "w");
    if (op == NULL)
    {
        perror("Error opening output file");
        return;
    }
    
    fgets(line, sizeof(line), fp); //skip the first line
    
    //assign numbers to sources
    while (fgets(line, sizeof(line), fp))
    {
        char* source = strtok(line, ",");
        char* target = strtok(NULL, ",");
        int weight = atoi(strtok(NULL, ","));
        strtok(NULL, ",");

        int source_num = -1;
        for (int i = 0; i < unique_nodes; i++) 
        {
            if (strcmp(nodes[i].name, source) == 0)
            {
                source_num = i;
                break;
            }
        }

        //if the source node is not found, add it to the nodes array
        if (source_num == -1)
        {
            strcpy(nodes[unique_nodes].name, source);
            nodes[unique_nodes].number = unique_nodes;
            source_num = unique_nodes;
            unique_nodes++;
        }
    }
    
    //reset the file pointer
    fseek(fp, 0, SEEK_SET);
    
    int numbers = unique_nodes;
    unique_nodes = 0;
    
    fgets(line, sizeof(line), fp); //skip the first line
    
    //now assign numbers to the target nodes
    while (fgets(line, sizeof(line), fp))
    {
    	char* source = strtok(line, ",");
        char* target = strtok(NULL, ",");
        int weight = atoi(strtok(NULL, ","));
        strtok(NULL, ",");
        
        int source_num = -1;
        for (int i = 0; i < numbers; i++) 
        {
            if (strcmp(nodes[i].name, source) == 0)
            {
                source_num = i;
                break;
            }
        }

        //if the source node is not found, add it to the nodes array
        if (source_num == -1)
        {
            strcpy(nodes[numbers].name, source);
            nodes[numbers].number = numbers;
            source_num = numbers;
            numbers++;
        }
        
        fprintf(op, "%d\t", source_num);
        
        int target_num = -1;
        for (int i = 0; i < numbers; i++) 
        {
            if (strcmp(nodes[i].name, target) == 0)
            {
                target_num = i;
                break;
            }
        }

        //if the target node is not found, add it to the nodes array
        if (target_num == -1)
        {
            strcpy(nodes[numbers].name, target);
            nodes[numbers].number = numbers;
            target_num = numbers;
            numbers++;
            
            //printf("NODES: %d\n", numbers);
        }
        
        //write to the file
        fprintf(op, "%d\t%d\n", target_num, weight);

        strcpy(edges[num_edges].source, source);
        strcpy(edges[num_edges].target, target);
        edges[num_edges].weight = weight;
        num_edges++;
    }
   
    fclose(fp);
    
    unique_nodes = numbers;
    *num_nodes = unique_nodes;
    *adjacency_matrix = malloc(unique_nodes * sizeof(int*));
    for (int i = 0; i < unique_nodes; i++)
    {
        (*adjacency_matrix)[i] = malloc(unique_nodes * sizeof(int));
        for (int j = 0; j < unique_nodes; j++)
        {
            if (i == j)
            {
                (*adjacency_matrix)[i][j] = 0;
            }
            else
            {
                (*adjacency_matrix)[i][j] = INF;
            }
        }
    }

    for (int i = 0; i < num_edges; i++)
	{
    		int source_num = -1;
    		int target_num = -1;
    
    		for (int j = 0; j < unique_nodes; j++)
    		{
        		if (strcmp(nodes[j].name, edges[i].source) == 0)
       	 			{
            				source_num = j;
        			}
        			
        		if (strcmp(nodes[j].name, edges[i].target) == 0)
        			{
            				target_num = j;
        			}
        			
        		if (source_num != -1 && target_num != -1)
        			{
            				break;
        			}
    		}

    		// Check if source_num or target_num are out of bounds
    		if (source_num == -1 || target_num == -1 || source_num >= unique_nodes || target_num >= unique_nodes)
    		{
        		continue; // Skip this edge
        		printf("%d, %d \n", source_num, target_num);
    		}

    		(*adjacency_matrix)[source_num][target_num] = edges[i].weight;
	}
}

void read_file(char *filename, int*** adjacency_matrix, int* num_nodes)
{
    FILE *file = fopen(filename, "r"); //open the input file for reading
    if (!file)
    {
        printf("Error opening file.\n");
        exit(1);
    }

    int source, target, weight;
    int max_node = -1; //for num nodes tracker
    while (fscanf(file, "%d\t%d\t%d", &source, &target, &weight) == 3) //read source num, target num, weight
    {
        if (source > max_node)
        {
            max_node = source; //source num greater found
        }
        
        
        if (target > max_node)
        {
            max_node = target; //target num greater found
        }
    }

    fclose(file);
    
    //intialize the adjacency matrix passed as argument according to the num nodes
    *num_nodes = max_node + 1;
    *adjacency_matrix = malloc(*num_nodes * sizeof(int*));
    for (int i = 0; i < *num_nodes; i++)
    {
        (*adjacency_matrix)[i] = malloc(*num_nodes * sizeof(int));
        for (int j = 0; j < *num_nodes; j++)
        {
            if (i == j)
            {
                (*adjacency_matrix)[i][j] = 0; //self loops set to 0
            }
            else
            {
                (*adjacency_matrix)[i][j] = INF; //all loops to INF unless filled
            }
        }
    }
    
    //fill the adjacency matrix
    file = fopen(filename, "r");
    while (fscanf(file, "%d\t%d\t%d", &source, &target, &weight) == 3)
    {
        (*adjacency_matrix)[source][target] = weight;
    }

    fclose(file);
}

//fill random nodes array with random source and destination (node pairs)
void random_assign(int*** random_nodes, int num_nodes)
{
    (*random_nodes) = malloc(NUM_RANDOM_NODES * sizeof(int*));
    for (int i = 0; i < NUM_RANDOM_NODES; i++)
    {
        (*random_nodes)[i] = malloc(2 * sizeof(int));
    }
    
    srand(time(NULL));     
    for (int i = 0; i < NUM_RANDOM_NODES; i++)
    {
    	for(int j=0; j< 2; j++)
    	{
        	(*random_nodes)[i][j] = rand() % num_nodes;
        }
    }
}

void findKShortest_seriel(int **g, int k, int source_node, int destination_node, int rank, int num_nodes)
{
    //vector to store distances
    int dis[num_nodes][k];
    int i, j;
    
    for (i = 0; i < num_nodes; i++)
    {
        for (j = 0; j < k; j++)
        {
            dis[i][j] = INF; //initialize distances to infinity
        }
    }

    //initialization of priority queue
    struct Pair
    {
        int first; //distance
        int second; //node
    } pq[num_nodes]; //priority queue array

    int pq_size = 0; //size of priority queue
    pq[pq_size].first = 0; //distance of source node to itself is 0
    pq[pq_size].second = source_node; //source node
    pq_size++;
    
    dis[source_node][0] = 0; //sistance from source to itself is 0
    
    //main loop to find shortest paths
    while (pq_size > 0)
    {
        //extracting node with minimum distance from priority queue
        int u = pq[0].second; //source node
        int d = pq[0].first; //distance to node u
        pq[0] = pq[pq_size - 1]; //replace first element with the last
        pq_size--;

        //if destination node is reached and the kth shortest path to it has been found, exit
        if (u == destination_node && dis[u][k - 1] < d)
        {
            break;
        }

        //traversing the adjacency matrix
        for (i = 0; i < num_nodes; i++)
        {
            if (g[u][i] == 0 || g[u][i] == INF)
            {
                continue; //skip if no connection or infinite distance
            }

            int dest = i; //destination node
            int cost = g[u][i]; //cost from u to dest

            //checking if current path is shorter than previously found paths
            if (d + cost < dis[dest][k - 1])
            {
                dis[dest][k - 1] = d + cost; //update distance to destination

                //sorting the distances of destination node in ascending order
                for (j = 0; j < k; j++)
                {
                    int l;
                    for (l = j + 1; l < k; l++)
                    {
                        if (dis[dest][j] > dis[dest][l])
                        {
                            int temp = dis[dest][j];
                            dis[dest][j] = dis[dest][l];
                            dis[dest][l] = temp;
                        }
                    }
                }

                //pushing elements to priority queue
                {
                    pq[pq_size].first = d + cost;
                    pq[pq_size].second = dest;
                    pq_size++;
                }
            }
        }
    }
    
    //printing results
    printf("K shortest paths from node %d to node %d:\n", source_node, destination_node);
    for (j = 0; j < k; j++)
    {
        printf("Path %d: %d\n", j + 1, dis[destination_node][j]);
    }

    printf("\n");
}


void findKShortest(int **g, int k, int source_node, int destination_node, int rank, int num_nodes)
{
    //vector to store distances
    int dis[num_nodes][k];
    int i, j;

    #pragma omp parallel
    {
    	#pragma omp for
    	for (i = 0; i < num_nodes; i++)
    	{
        	for (j = 0; j < k; j++)
        	{
            		dis[i][j] = INF; //initialize distances to infinity
        	}
    	}
    }

    //initialization of priority queue
    struct Pair
    {
        int first; //distance
        int second; //node
    } pq[num_nodes]; //priority queue array

    int pq_size = 0; //size of priority queue
    pq[pq_size].first = 0; //distance of source node to itself is 0
    pq[pq_size].second = source_node; //source node
    pq_size++;

    dis[source_node][0] = 0; //distance from source to itself is 0

    //main loop to find shortest paths
    while (pq_size > 0)
    {
        //extracting node with minimum distance from priority queue
        int u = pq[0].second; //source node
        int d = pq[0].first; //distance to node u
        pq[0] = pq[pq_size - 1]; //replace first element with the last
        pq_size--;

        //if destination node is reached and the kth shortest path to it has been found, exit
        if (u == destination_node && dis[u][k - 1] < d)
        {
            break;
        }

        //traversing the adjacency matrix
        for (i = 0; i < num_nodes; i++)
        {
            if (g[u][i] == 0 || g[u][i] == INF)
            {
                continue; //skip if no connection or infinite distance
            }

            int dest = i; //destination node
            int cost = g[u][i]; //cost from u to dest

            //checking if current path is shorter than previously found paths
            if (d + cost < dis[dest][k - 1])
            {
                dis[dest][k - 1] = d + cost; //update distance to destination

                //sorting the distances of destination node in ascending order
                for (j = 0; j < k; j++)
                {
                    int l;
                    for (l = j + 1; l < k; l++)
                    {
                        if (dis[dest][j] > dis[dest][l])
                        {
                            int temp = dis[dest][j];
                            dis[dest][j] = dis[dest][l];
                            dis[dest][l] = temp;
                        }
                    }
                }

                //pushing elements to priority queue
                //#pragma omp critical
                {
                    pq[pq_size].first = d + cost;
                    pq[pq_size].second = dest;
                    pq_size++;
                }
            }
        }
    }
    
    //printing results
    printf("K shortest paths from node %d to node %d:\n", source_node, destination_node);
    for (j = 0; j < k; j++)
    {
        printf("Path %d: %d\n", j + 1, dis[destination_node][j]);
    }

    printf("\n");
}


int main(int argc, char *argv[])
{
    //MPI initializeion
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char * filename = argv[1];
    
    int** matrix = NULL; //adjacency matrix for the graph
    int num_nodes = 0; //number of nodes in the graph 

    int k = 3; //num shortest paths
    
    //random nodes pair array ([10][2] in this case)
    int ** random_nodes = (int**)malloc(NUM_RANDOM_NODES * sizeof(int*));
    for (int i = 0; i < NUM_RANDOM_NODES; i++)
    {
           random_nodes[i] = (int*)malloc(2 * sizeof(int));
    }
    
    clock_t start, end; //for keeping track of the start and end of execution
    
    double* process_times = (double*)malloc(size * sizeof(double)); //to store the process execution times 
    
    //rank 0 will fill adjacency graph, num nodees and initialzie the random nodes
    if(rank == 0)
    {
    	preprocess_file(filename, &matrix, &num_nodes);
        //read_file("output.txt", &matrix, &num_nodes); //read and fill adjacency graph and store num nodes
        random_assign(&random_nodes, num_nodes); //fill randome nodes array
        
        ////seriel execution
        printf("Seriel Execution: \n");
        
        start = clock();
        
        for(int i=0; i< NUM_RANDOM_NODES; i++)
        {
            findKShortest_seriel(matrix, k, random_nodes[i][0], random_nodes[i][1], rank, num_nodes); //without openmp
        }
        
        end = clock();
    
    	double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    	printf("Time taken (Seriel Execution): %f\n\n", time_taken);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //broadcast num nodes to all processes by rank 0
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //other processes will initalize matrix according to num nodes recieved through broadcast
    if(rank != 0) 
    {
        matrix = (int**)malloc(num_nodes * sizeof(int*));
        for (int i = 0; i < num_nodes; i++)
        {
            matrix[i] = (int*)malloc(num_nodes * sizeof(int));
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //broadcast the filled matrix by rank 0 to all processes
    for (int i = 0; i < num_nodes; i++)
    {
        MPI_Bcast(matrix[i], num_nodes, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //broadcast the filled random nodes array by rank 0 to all processes
    for(int i=0; i<NUM_RANDOM_NODES; i++)
    {
        MPI_Bcast(random_nodes[i], 2, MPI_INT, 0, MPI_COMM_WORLD);
    }       
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //parallel execution
    start = clock();

    findKShortest(matrix, k, random_nodes[rank][0], random_nodes[rank][1], rank, num_nodes); //openmp used to parallelize loops
    
    end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //gather the time taken for execution by all processes
    MPI_Gather(&time_taken, 1, MPI_DOUBLE, process_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    //display the total time for paralell execution
    if (rank == 0)
    {
        double total_time_taken = 0.0;
        for (int i = 0; i < size; i++)
        {
            total_time_taken += process_times[i];
        }
        printf("Total Time taken (Parallel): %f\n", total_time_taken);
    }

    MPI_Finalize(); //terminate the MPI

    return 0;
}
