#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#define DEBUG 1

#define N 1024

int main(int argc, char *argv[]) {

    int i, j, numprocs, rank;

    float vector[N];

    struct timeval tv1, tv2, tc1, tc2, tc3, tc4;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block_size = (int) (ceil((1.0 * N) / numprocs));

    float result[block_size * numprocs];

    float matrix[block_size * numprocs][N];

    float matl[block_size][N];

    float resultl[block_size];

    if (rank == 0) {
        /* Initialize Matrix and Vector */
        for (i = 0; i < N; i++) {
            vector[i] = i;
            for (j = 0; j < N; j++) {
                matrix[i][j] = i + j;
            }
        }
    }

    gettimeofday(&tc1, NULL);

    MPI_Scatter(matrix, block_size * N, MPI_FLOAT, matl, block_size * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tc2, NULL);

    int comm_time1 = (tc2.tv_usec - tc1.tv_usec) + 1000000 * (tc2.tv_sec - tc1.tv_sec);

    gettimeofday(&tv1, NULL);


    for (i = 0; i < block_size; i++) {
        resultl[i] = 0;
        for (j = 0; j < N; j++) {
            resultl[i] += matl[i][j] * vector[j];
        }
    }

    gettimeofday(&tv2, NULL);

    int computing_time = (tv2.tv_usec - tv1.tv_usec) + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    gettimeofday(&tc3, NULL);

    MPI_Gather(resultl, block_size, MPI_FLOAT, result, block_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tc4, NULL);

    int comm_time2 = (tc4.tv_usec - tc3.tv_usec) + 1000000 * (tc4.tv_sec - tc3.tv_sec);

    if (rank == 0) {
        /*Display result */
        if (DEBUG) {
            for (i = 0; i < N; i++) {
                printf(" %f \t ", result[i]);
            }
        } else {
            printf("[Process - %d] Computing Time (seconds) = %lf | Comm Time (seconds) = %lf\n", rank,
                   (double) computing_time / 1E6, (double) (comm_time1 + comm_time2) / 1E6);
        }
    } else {
        if (!DEBUG) {
            printf("[Process - %d] Computing Time (seconds) = %lf | Comm Time (seconds) = %lf\n", rank,
                   (double) computing_time / 1E6, (double) (comm_time1 + comm_time2) / 1E6);
        }
    }

    MPI_Finalize();

    return 0;
}

