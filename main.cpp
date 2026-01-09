#include <mpi.h>
#include <omp.h>

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <fstream>

// ReLU activation
inline double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

int main(int argc, char** argv) {

    // --- MPI initialization with thread support ---
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI does not support required threading level" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Neural network dimensions ---
    const int input_size  = 1024;
    const int hidden_size = 512;
    const int output_size = 10;

    // --- Data allocation ---
    std::vector<double> input(input_size);
    std::vector<double> hidden(hidden_size, 0.0);
    std::vector<double> output(output_size, 0.0);

    std::vector<double> weights_input_hidden(input_size * hidden_size);
    std::vector<double> weights_hidden_output(hidden_size * output_size);

    // --- Initialization (rank 0 only) ---
    if (rank == 0) {
        for (int i = 0; i < input_size; ++i)
            input[i] = static_cast<double>(rand()) / RAND_MAX;

        for (auto& w : weights_input_hidden)
            w = static_cast<double>(rand()) / RAND_MAX;

        for (auto& w : weights_hidden_output)
            w = static_cast<double>(rand()) / RAND_MAX;
    }

    // --- Broadcast input and weights ---
    MPI_Bcast(input.data(), input_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights_input_hidden.data(),
              input_size * hidden_size,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights_hidden_output.data(),
              hidden_size * output_size,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- Timing start ---
    double start_time = MPI_Wtime();

    // --- MPI decomposition of hidden layer ---
    int neurons_per_proc = hidden_size / size;
    int start = rank * neurons_per_proc;
    int end   = (rank == size - 1) ? hidden_size : start + neurons_per_proc;

    // --- Compute hidden layer (OpenMP) ---
    #pragma omp parallel for schedule(static)
    for (int j = start; j < end; ++j) {
        double sum = 0.0;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights_input_hidden[i * hidden_size + j];
        }
        hidden[j] = relu(sum);
    }

    // --- Combine partial hidden layers ---
    MPI_Allreduce(MPI_IN_PLACE,
                  hidden.data(),
                  hidden_size,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    // --- Compute output layer (OpenMP) ---
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < output_size; ++k) {
        double sum = 0.0;
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden[j] * weights_hidden_output[j * output_size + k];
        }
        output[k] = sum;
    }

    // --- Timing end ---
    double end_time = MPI_Wtime();

    // --- Save results (rank 0 only) ---
    if (rank == 0) {
        std::ofstream file;
        file.open("results.csv", std::ios::app);
        file << size << ","
             << omp_get_max_threads() << ","
             << (end_time - start_time)
             << std::endl;
        file.close();

        // Optional display
        std::cout << "Execution finished (MPI="
                  << size << ", OMP="
                  << omp_get_max_threads() << ")" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
