// mpi_quicksort.c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include "quicksort.h"

#define N 32  // Total elements to sort

int main(int argc, char** argv) {
    int rank, size;
    int arr[N], *local_arr;
    int local_n = N;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_n = N / size;
    local_arr = (int*)malloc(local_n * sizeof(int));

    if (rank == 0) {
        srand(time(NULL));
        printf("Unsorted array:\n");
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 100;
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    double start = MPI_Wtime();

    // Distribute chunks
    MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort local chunk
    quicksort(local_arr, 0, local_n - 1);

    // Gather sorted chunks back
    MPI_Gather(local_arr, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Final full sort to merge all chunks
        quicksort(arr, 0, N - 1);

        double end = MPI_Wtime();

        printf("Fully sorted array:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        printf("Execution Time: %f seconds\n", end - start);
    }

    free(local_arr);
    MPI_Finalize();
    return 0;
}
