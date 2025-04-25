#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/cl.h>

#define ARRAY_SIZE 1200
#define CL_TARGET_OPENCL_VERSION 300

void checkError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        printf("Error: %s (%d)\n", msg, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

const char* quick_sort_kernel = 
"__kernel void quicksort(__global int* array, int left, int right) {\n"
"    int i = left + get_global_id(0);\n"
"    if (i >= right) return;\n"
"    \n"
"    int pivot = array[right];\n"
"    int index = left;\n"
"    \n"
"    for (int j = left; j < right; j++) {\n"
"        if (array[j] <= pivot) {\n"
"            int temp = array[index];\n"
"            array[index] = array[j];\n"
"            array[j] = temp;\n"
"            index++;\n"
"        }\n"
"    }\n"
"    \n"
"    int temp = array[index];\n"
"    array[index] = array[right];\n"
"    array[right] = temp;\n"
"}\n";

int main(int argc, char** argv) {  // Fixed this line
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = ARRAY_SIZE / size;
    int *data = NULL;
    int *sub_data = (int*)malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        data = (int*)malloc(ARRAY_SIZE * sizeof(int));
        printf("Unsorted array: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data[i] = rand() % 100;
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    MPI_Scatter(data, chunk_size, MPI_INT, sub_data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    checkError(err, "clGetDeviceIDs");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");
    
    const cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    checkError(err, "clCreateCommandQueueWithProperties");

    program = clCreateProgramWithSource(context, 1, &quick_sort_kernel, NULL, &err);
    checkError(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    checkError(err, "clBuildProgram");

    kernel = clCreateKernel(program, "quicksort", &err);
    checkError(err, "clCreateKernel");

    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                          chunk_size * sizeof(int), sub_data, &err);
    checkError(err, "clCreateBuffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    checkError(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(int), &(int){0});
    checkError(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 2, sizeof(int), &(int){chunk_size-1});
    checkError(err, "clSetKernelArg");

    size_t global_size = chunk_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    checkError(err, "clEnqueueNDRangeKernel");
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, 
                            chunk_size * sizeof(int), sub_data, 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer");

    MPI_Gather(sub_data, chunk_size, MPI_INT, data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Final CPU quicksort
        void qsort_local(int *arr, int left, int right) {
            if (left < right) {
                int pivot = arr[right];
                int i = left;
                for (int j = left; j < right; j++) {
                    if (arr[j] <= pivot) {
                        int temp = arr[i];
                        arr[i] = arr[j];
                        arr[j] = temp;
                        i++;
                    }
                }
                int temp = arr[i];
                arr[i] = arr[right];
                arr[right] = temp;
                qsort_local(arr, left, i-1);
                qsort_local(arr, i+1, right);
            }
        }
        qsort_local(data, 0, ARRAY_SIZE-1);

        printf("Sorted array: ");
        for (int i = 0; i < ARRAY_SIZE; i++) printf("%d ", data[i]);
        printf("\n");
        
        double end_time = MPI_Wtime();
        printf("Total execution time: %f seconds\n", end_time - start_time);
        
        free(data);
    }

    // Cleanup
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(sub_data);
    MPI_Finalize();
    return 0;
} 
