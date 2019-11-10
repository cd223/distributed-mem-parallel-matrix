#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

/* Structs for optional program arguments. */
struct prog_opts_t
{
    int mat_dimension;
    double rel_precision;
    bool info_mode;
    bool verbose_mode;
} opts;

/* Parse values of optional command line arguments. */
void parse_opts(int argc, char **argv)
{
    static const char *options = "d:p:iv";
    int opt = getopt(argc, argv, options);
    while(opt != -1) {
        switch( opt ) {
            case 'd': 
                opts.mat_dimension = atoi(optarg);  // -d [int]
                break;
            case 'p':
                opts.rel_precision = atof(optarg);  // -p [double]
                break;
            case 'i':
                opts.info_mode = true;              // -i [No specifier]
                break;
            case 'v':
                opts.verbose_mode = true;           // -v [No specifier]
                break;
            default:
                break;
        }
        opt = getopt(argc, argv, options);
    }
}

/* Display matrix produced at each step by relaxation process. */
void display_matrix(double *mat, int mat_size, int num_cells, 
                    int iteration_num)
{
    // Positive non-zero iteration_num means matrix relaxed
    if(iteration_num == -1) {
        printf("\nINFO: ___Initial Matrix_______\n");
    } else if(iteration_num > 0) {
        printf("\nINFO: ___Relaxed Matrix (%d iterations)_______\n",
            iteration_num);
    } else {
        printf("\nINFO: ___Iteration Complete_______\n");
    }
    for (int n = 0; n < num_cells; ++n) {
        if (n % mat_size == 0) {
            printf("\n");
        }
        printf("%f ", mat[n]);
    }
    printf("\n");
}

/* Display program configuration, including number of MPI processes. */
void display_program_config(int num_procs)
{
    printf("\nINFO: Arguments to be used in program:\n"
            "   - Matrix Dimensions (-d):       %dx%d\n"
            "   - Relaxation Precision (-p):    %f\n"
            "   - Info Mode (-i):               %s\n"
            "   - Verbose Mode (-v):            %s\n"
            "\nINFO: %d processes running...\n",
            opts.mat_dimension, opts.mat_dimension, opts.rel_precision,
            opts.info_mode ? "YES" : "NO", opts.verbose_mode ? "YES" : "NO",
            num_procs);
}

/* Display how many rows each process has been allocated (and where they are)*/
void display_problem_breakdown(int rank, int num_rows, int rows_from_top) 
{
    printf("\nINFO: MPI proc #%d allocated %d row(s) of matrix to relax"
           " (starts %d row(s) from top).\n", rank, num_rows, rows_from_top);
}

/* Error checks for standard initial MPI calls - returns 1 on error, else 0 */ 
int setup_mpi(int argc, char **argv, int *num_procs, int *curr_rank,
              bool *is_root_proc)
{
    int err = MPI_Init(&argc, &argv);
    // Initialise connections between processes for later communication
    if (err != MPI_SUCCESS) {
        printf("\nERROR: Problem initialising MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, err);
        return 1;
    }
    // Determine number of processes/tasks (nodes x tasks per node on SLURM)
    err = MPI_Comm_size(MPI_COMM_WORLD, num_procs);
    if (err != MPI_SUCCESS) {
        printf("\nERROR: Problem determining number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, err);
        return 1;
    }
    // Determine process rank (curr_rank == 0 if root)
    err = MPI_Comm_rank(MPI_COMM_WORLD, curr_rank);
    if (err != MPI_SUCCESS) {
        printf("\nERROR: Problem determining current process rank.\n");
        MPI_Abort(MPI_COMM_WORLD, err);
        return 1;
    }
    *is_root_proc = (*curr_rank == 0) ? true : false; // If root, rank is 0
    return 0;
}

/* Calculates if element 'cell' is on the edge of matrix */
bool is_cell_on_edge(int dim, int total, int cell)
{
    return (cell % dim == 0                              // left
            || (cell+1) % dim == 0                       // right
            || cell < dim                                // top
            || ((cell < total) && cell >= dim*(dim-1))); // bottom
}

/* 
    Indicates whether prev and new cell values within precision range.
        | 0 = Within precision, 1 = Outside precision
*/
int calc_prec(double prev_val, double new_val, double precision)
{
    double difference = fabs(prev_val-new_val);
	return (difference > precision);
}

/*  Cell relaxation - calculates average of a cell's 4 neighbours. */
double cell_relax(double *mat, int cell, int mat_size)
{
	double total = mat[cell-1]              // Cell to left
                 + mat[cell+1]              // Cell to right
                 + mat[cell - mat_size]     // Cell above
                 + mat[cell + mat_size];    // Cell below
    return total / (double) 4;
}

/* Extract program variables from command line and set default values. */
void extract_program_opts(int argc, char **argv)
{
    // Set default values in case of invalid input
    opts.mat_dimension = 50;
    opts.rel_precision = 0.01;
    opts.info_mode = false;
    opts.verbose_mode = false;
    parse_opts(argc, argv);
}

/* Divide total matrix between all processes - split equally as possuble */
void divide_matrix_between_procs(int *s_count, int *r_count, int *s_shift,
                                 int *r_shift, int num_procs,
                                 int rows_per_proc, int rows_left, 
                                 int mat_size, bool display_verbose_info)
{
    // Each proc reads 2 more rows than it relaxes (to read above/below)
    for (int p = 0; p < num_procs; p++) {
        // Non-zero row remainder if rows can't be split equally
        int row_remainder = p < rows_left? 1 : 0;

        // 'Send' = rows to scatter, 'Receive' = rows to gather 
        int rows_to_send = rows_per_proc + row_remainder + 2;
        int rows_to_receive = rows_to_send - 2;
        s_count[p] = mat_size * rows_to_send;    // What is read (cell count)
        r_count[p] = mat_size * rows_to_receive; // What is relaxed (cell count)
        
        // Calculate shifts in send and receive counts
        s_shift[p] = p > 0? r_count[p-1] + s_shift[p-1] : 0;
        r_shift[p] = mat_size + s_shift[p];

        // Print what work this process has been allocated
        if (display_verbose_info) {
            int shift = r_shift[p] / mat_size;
            display_problem_breakdown(p, rows_to_receive, shift);
        }
	}
}

/* Returns 1 if relaxed matrix includes any cells which are not within precision */
int matrix_relax(double* mat_read, double *mat_write, int init_pos, int final_pos,
                 int mat_size, int recshift, double prec)
{
    int outside_prec = 0; // Used to check if any values outside precision
    int total_cells = final_pos-init_pos;

    for (int n=init_pos; n < final_pos; n++) {
        int curr_cell = recshift + (n - init_pos);

        // If non-edge cell, relax and check the new precision, updating flag
        if (!is_cell_on_edge(mat_size, total_cells, curr_cell)) {
            double prev_val = mat_write[(n - init_pos)];
            double new_val = cell_relax(mat_read, n, mat_size);
            mat_write[(n - init_pos)] = new_val;
            outside_prec = outside_prec || calc_prec(prev_val, new_val, prec);
        } 
        // If edge cell, no relaxing needed - keep old value
        else {
            mat_write[(n - init_pos)] = mat_read[n]; 
        }
    }
    return outside_prec;
}

int main(int argc, char **argv)
{
    extract_program_opts(argc, argv);       // Determine values of program args
    int mat_size = opts.mat_dimension;
    bool info_mode = opts.info_mode;
    bool verbose_mode = opts.verbose_mode;
    double prec = opts.rel_precision;

    int num_procs, curr_rank;               // Number/rank of procs
    bool is_root_proc = false;              // True if rank==0
    double *mat_overall = NULL;             // Matrix to be split
    double *mat_read = NULL;                // Sub-matrix each proc reads from
    double *mat_write = NULL;               // Sub-matrix each proc writes to

    int outside_prec = 0;                   // Set by root proc (with MPI_LOR)
    int next_iteration_needed = 0;          // Set by each proc if not finished
    int n_cells = mat_size * mat_size;      // Total cell count
    int inner_size = mat_size - 2;          // Size without each outer row/col

    // Initialise MPI, determine number of processes and current process rank
    if (setup_mpi(argc, argv, &num_procs, &curr_rank, &is_root_proc)) {
        return 1;
    }

    // Flag for displaying state of matrix after iterations
    bool display_info = info_mode && is_root_proc;  
    // Flag for displaying state and extra work division details
    bool display_verbose_info = verbose_mode && is_root_proc; 

    // Create arrays to divide matrix between procs
    size_t arr_size = num_procs * sizeof(int);
    int *s_count = malloc(arr_size);  // Num items root SENDS to each proc
    int *r_count = malloc(arr_size);  // Num items root RECEIVES from each proc
    int *s_shift = malloc(arr_size);  // Where in matrix root SENDS from
    int *r_shift = malloc(arr_size);  // Where in matrix root RECEIVES from
    if(s_count==NULL || s_shift==NULL || r_count==NULL || r_shift==NULL) {
        printf("\nERROR: Allocation of memory for arrays failed.\n");
        return 1;
    }

    // Split rows of overall matrix equally as possible between processes
    divide_matrix_between_procs(s_count, r_count, s_shift, r_shift, num_procs,
                                inner_size/num_procs, inner_size % num_procs,
                                mat_size, display_verbose_info);
    
    // Calculate where in matrix to start and stop relaxing for current proc
    int init_pos = mat_size;
    int final_pos = r_count[curr_rank] + init_pos;
    
    // If root process, setup overall/main matrix
    if(is_root_proc) {
        size_t overall_matrix_size = n_cells * sizeof(double);
        mat_overall = malloc(overall_matrix_size);
        if(mat_overall==NULL) {
            printf("\nERROR: Allocation of memory to main matrix failed.\n");
            return 1;
        }
        for (int n = 0; n < n_cells; ++n) {
            mat_overall[n] = is_cell_on_edge(mat_size, n_cells, n) ? 1.0 : 0.0;
        }
        if(display_verbose_info) {
            display_program_config(num_procs);
        }
        if(display_info || display_verbose_info) {
            display_matrix(mat_overall, mat_size, n_cells, -1);
        }
    }

    // Create read/write matrices for each process of different size
    // mat_read populated by MPI_Scatterv, mat_write populated by MPI_Gatherv
    size_t read_size = s_count[curr_rank] * sizeof(double);
    size_t write_size = r_count[curr_rank] * sizeof(double);
    mat_read = malloc(read_size);
    mat_write = malloc(write_size);
    if (mat_read==NULL || mat_write==NULL) {
        printf("\nERROR: Allocation of memory for submatrices failed.\n");
        return 1;
    }
    
    int iteration_num = 0;
    int err = 0;
    
    while(true) {
        // Split mat_overall in (maybe uneven) chunks across each mat_read
        err = MPI_Scatterv(mat_overall, s_count, s_shift, MPI_DOUBLE, mat_read,
                     s_count[curr_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Check for errors
        if (err != MPI_SUCCESS) {
            printf("\nERROR: Problem scattering mat_overall to processes.\n");
            MPI_Abort(MPI_COMM_WORLD, err);
            return 1;
        }

        // matrix_relax returns 1 if precision not reached on all cells
        next_iteration_needed = matrix_relax(mat_read, mat_write, init_pos,
                                            final_pos, mat_size, 
                                            r_shift[curr_rank], prec);
        iteration_num++;

        // Return relaxed cells to root proc, which writes this to mat_overall
        err = MPI_Gatherv(mat_write, r_count[curr_rank], MPI_DOUBLE, mat_overall,
                    r_count, r_shift, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Check for errors
        if (err != MPI_SUCCESS) {
            printf("\nERROR: Problem gathering relaxed cells from processes.\n");
            MPI_Abort(MPI_COMM_WORLD, err);
            return 1;
        }

        // Perform logical OR across all procs' next_iteration_needed flag to 
        // check if we need another iteration or can break to print matrix
        err = MPI_Allreduce(&next_iteration_needed, &outside_prec, 1, MPI_INT, 
                      MPI_LOR, MPI_COMM_WORLD);
        // Check for errors
        if (err != MPI_SUCCESS) {
            printf("\nERROR: Problem reducing OR across processes' flags.\n");
            MPI_Abort(MPI_COMM_WORLD, err);
            return 1;
        }
        
        // If precision not outside limits (outside_prec==0), all relaxed!
        if (!outside_prec) {
            break;
        } 
        // Otherwise loop again, resetting each proc's next_iteration_needed flag
        else { 
            next_iteration_needed = 0;
            if (display_info || display_verbose_info) {
                display_matrix(mat_overall, mat_size, n_cells, 0);
            }
        }
    }

    // Display final relaxed matrix
    if (display_info || display_verbose_info) {
        display_matrix(mat_overall, mat_size, n_cells, iteration_num);
    }

    // Free memory allocated to matrices
    free(s_shift);
    free(r_shift);
    free(s_count);
    free(r_count);
    free(mat_read);
    free(mat_write);
    if(is_root_proc) {
        free(mat_overall);
    }
    MPI_Finalize(); // Tidy up MPI - inc. implicit call to MPI_Barrier
    return 0;
}