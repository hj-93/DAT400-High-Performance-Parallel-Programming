//
//  nn_mpi.cpp
//
//  To compile: mpicxx -std=c++11 -O3 -fopenmp -o train_mpi
//  To run: ./train_mpi

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <mpi.h>

#include "deep_core.h"
#include "vector_ops.h"

vector<string> split(const string &s, char delim)
{
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim))
    {
        tokens.push_back(item);
    }
    return tokens;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    // Number of processes
    int num_processors;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);

    // Number of current process
    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    string line;
    vector<string> line_v;
    int len;
    cout << "Loading data ...\n";
    vector<float> X_train;
    vector<float> y_train;
    ifstream myfile("train.txt");
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(), 0);
            for (unsigned i = 0; i < 10; ++i)
            {
                if (i == digit)
                {
                    y_train.push_back(1.);
                }
                else
                    y_train.push_back(0.);
            }

            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i)
            {
                X_train.push_back(strtof((line_v[i]).c_str(), 0));
            }
        }
        X_train = X_train / 255.0;
        myfile.close();
    }

    else
        cout << "Unable to open file" << '\n';

    int xsize = static_cast<int>(X_train.size());
    int ysize = static_cast<int>(y_train.size());

    // Some hyperparameters for the NN
    int BATCH_SIZE = 256 / num_processors;
    float lr = .01 / BATCH_SIZE;

    // Random initialization of the weights in root process
    vector<float> W1;
    vector<float> W2;
    vector<float> W3;

    if (process_id == 0)
    {
        W1 = random_vector(784 * 128);
        W2 = random_vector(128 * 64);
        W3 = random_vector(64 * 10);
    }
    else
    {
        W1.reserve(784 * 128);
        W2.reserve(128 * 64);
        W3.reserve(64 * 10);
    }

    // Broadcast initial weights
    int status_w1 = MPI_Bcast((void *)W1.data(), 784 * 128, MPI_FLOAT, 0, MPI_COMM_WORLD);
    int status_w2 = MPI_Bcast((void *)W2.data(), 128 * 64, MPI_FLOAT, 0, MPI_COMM_WORLD);
    int status_w3 = MPI_Bcast((void *)W3.data(), 64 * 10, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (process_id == 0)
    {
        cout << "status_w1: " << status_w1 << endl;
    }

    if (process_id == 0 && (status_w1 || status_w2 || status_w3))
    {
        cout << "Broadcast of initial weights failed." << endl;
        exit(-1);
    }

    std::chrono::time_point<std::chrono::system_clock> t1, t2;
    cout << "Training the model ...\n";
    for (unsigned i = 0; i < 1000; ++i)
    {
        t1 = std::chrono::system_clock::now();
        
        // Building batches of input variables (X) and labels (y)
        int randindx = rand() % (42000 - BATCH_SIZE);
        vector<float> b_X;
        vector<float> b_y;
        for (unsigned j = randindx * 784; j < (randindx + BATCH_SIZE) * 784; ++j)
        {
            b_X.push_back(X_train[j]);
        }
        for (unsigned k = randindx * 10; k < (randindx + BATCH_SIZE) * 10; ++k)
        {
            b_y.push_back(y_train[k]);
        }

        // Feed forward
        vector<float> a1 = relu(dot(b_X, W1, BATCH_SIZE, 784, 128));
        vector<float> a2 = relu(dot(a1, W2, BATCH_SIZE, 128, 64));
        vector<float> yhat = softmax(dot(a2, W3, BATCH_SIZE, 64, 10), 10);

        // Back propagation
        vector<float> dyhat = (yhat - b_y);
        // dW3 = a2.T * dyhat
        vector<float> dW3 = dot(transform(&a2[0], BATCH_SIZE, 64), dyhat, 64, BATCH_SIZE, 10);
        // dz2 = dyhat * W3.T * relu'(a2)
        vector<float> dz2 = dot(dyhat, transform(&W3[0], 64, 10), BATCH_SIZE, 10, 64) * reluPrime(a2);
        // dW2 = a1.T * dz2
        vector<float> dW2 = dot(transform(&a1[0], BATCH_SIZE, 128), dz2, 128, BATCH_SIZE, 64);
        // dz1 = dz2 * W2.T * relu'(a1)
        vector<float> dz1 = dot(dz2, transform(&W2[0], 128, 64), BATCH_SIZE, 64, 128) * reluPrime(a1);
        // dW1 = X.T * dz1
        vector<float> dW1 = dot(transform(&b_X[0], BATCH_SIZE, 784), dz1, 784, BATCH_SIZE, 128);

        // Allreduce the weight deltas to all processes
        vector<float> dW1_aggr = process_id == 0 ? W1 - lr * dW1 : -lr * dW1;
        vector<float> dW2_aggr = process_id == 0 ? W2 - lr * dW2 : -lr * dW2;
        vector<float> dW3_aggr = process_id == 0 ? W3 - lr * dW3 : -lr * dW3;

        int status_allreduce_w1 = MPI_Allreduce(dW1_aggr.data(), W1.data(), 784 * 128, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        int status_allreduce_w2 = MPI_Allreduce(dW2_aggr.data(), W2.data(), 128 * 64, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        int status_allreduce_w3 = MPI_Allreduce(dW3_aggr.data(), W3.data(), 64 * 10, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (process_id == 0)
        {
            cout << "status_allreduce_w1: " << status_allreduce_w1 << endl;
        }

        if (process_id == 0 && (status_allreduce_w1 || status_allreduce_w2 || status_allreduce_w3))
        {
            cout << "All-reduce failed." << endl;
            exit(-1);
        }

        if ((process_id == 0) && (i + 1) % 100 == 0)
        {
            cout << "Predictions:"
                 << "\n";
            print(yhat, 10, 10);
            cout << "Ground truth:"
                 << "\n";
            print(b_y, 10, 10);
            vector<float> loss_m = yhat - b_y;
            float loss = 0.0;
            for (unsigned k = 0; k < BATCH_SIZE * 10; ++k)
            {
                loss += loss_m[k] * loss_m[k];
            }
            t2 = std::chrono::system_clock::now();
            chrono::duration<double> elapsed_seconds = t2 - t1;
            double ticks = elapsed_seconds.count();
            cout << "Iteration #: " << i << endl;
            cout << "Iteration Time: " << ticks << "s" << endl;
            cout << "Loss: " << loss / BATCH_SIZE << endl;
            cout << "*******************************************" << endl;
        };
    };

    MPI_Finalize();

    return 0;
}
