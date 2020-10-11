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
#include "deep_core.h"
#include "vector_ops.h"
#include <mpi.h>


vector<string> split(const string &s, char delim) {
  stringstream ss(s);
  string item;
  vector<string> tokens;
  while (getline(ss, item, delim)) {
    tokens.push_back(item);
  }
  return tokens;
}

int main(int argc, char * argv[]) {
  // MPI: INIT
  int mpirank = 0;
  int p = 0, root = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

  string line;
  vector<string> line_v;
  if (mpirank == 0) cout << "Loading data ...\n";
  vector<float> X_train;
  vector<float> y_train;
  ifstream myfile ("train.txt");
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      line_v = split(line, '\t');
      int digit = strtof((line_v[0]).c_str(),0);
      for (unsigned i = 0; i < 10; ++i) {
        if (i == digit)
        {
          y_train.push_back(1.);
        }
        else y_train.push_back(0.);
      }
      
      int size = static_cast<int>(line_v.size());
      for (unsigned i = 1; i < size; ++i) {
        X_train.push_back(strtof((line_v[i]).c_str(),0));
      }
    }
    X_train = X_train/255.0;
    myfile.close();
  }
  
  else cout << "Unable to open file" << '\n';
  
  int xsize = static_cast<int>(X_train.size());
  int ysize = static_cast<int>(y_train.size());
  
  // Some hyperparameters for the NN
  int BATCH_SIZE = 256;
  int P_SIZE = BATCH_SIZE / p;
  float lr = .01/BATCH_SIZE;
  // Random initialization of the weights
  vector <float> W1 = random_vector(784*128);
  vector <float> W2 = random_vector(128*64);
  vector <float> W3 = random_vector(64*10);

  // MPI: Consistent initial random weights matrix among processes
  MPI_Bcast(W1.data(), W1.size(), MPI_FLOAT, root, MPI_COMM_WORLD);
  MPI_Bcast(W2.data(), W2.size(), MPI_FLOAT, root, MPI_COMM_WORLD);
  MPI_Bcast(W3.data(), W3.size(), MPI_FLOAT, root, MPI_COMM_WORLD);

  std::chrono::time_point<std::chrono::system_clock> t1,t2;     
  if (mpirank == 0) cout << "Training the model ...\n";
  for (unsigned i = 0; i < 1000; ++i) {    
    t1 = std::chrono::system_clock::now();    
    // Building batches of input variables (X) and labels (y)
    int randindx = rand() % (42000-BATCH_SIZE);
    // MPI: Consistent randindx among processes
    MPI_Bcast(&randindx, 1, MPI_INT, root, MPI_COMM_WORLD);

    vector<float> b_X;
    vector<float> b_y;
    for (unsigned j = randindx*784; j < (randindx+BATCH_SIZE)*784; ++j){
      b_X.push_back(X_train[j]);
    }
    for (unsigned k = randindx*10; k < (randindx+BATCH_SIZE)*10; ++k){
      b_y.push_back(y_train[k]);
    }

    // Feed forward
    // MPI: Distribute batch to different processes
    vector<float> sub_b_X (b_X.begin() + mpirank * P_SIZE * 784, b_X.begin() + (mpirank + 1) * P_SIZE * 784);
    vector<float> a1 = relu(dot( sub_b_X, W1, P_SIZE, 784, 128 ));
    vector<float> a2 = relu(dot( a1, W2, P_SIZE, 128, 64 ));
    vector<float> yh = dot( a2, W3, P_SIZE, 64, 10 );

    a1.resize(a1.size() * p, 0);
    a2.resize(a2.size() * p, 0);
    yh.resize(yh.size() * p, 0);
    MPI_Allgather(a1.data(), a1.size()/p, MPI_FLOAT, a1.data(), a1.size()/p, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(a2.data(), a2.size()/p, MPI_FLOAT, a2.data(), a2.size()/p, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(yh.data(), yh.size()/p, MPI_FLOAT, yh.data(), yh.size()/p, MPI_FLOAT, MPI_COMM_WORLD);

    vector<float> yhat = softmax(yh, 10);

    // Back propagation
    vector<float> t_tmp, sub_p;
    vector<float> dyhat = (yhat - b_y);
    // dW3 = a2.T * dyhat
    t_tmp = std::move(transform( &a2[0], BATCH_SIZE, 64 ));
    sub_p = std::move(vector<float>(t_tmp.begin() + mpirank * 64/p * BATCH_SIZE, t_tmp.begin() + (mpirank + 1) * 64/p * BATCH_SIZE));
    vector<float> dW3 = dot(sub_p, dyhat, 64/p, BATCH_SIZE, 10);
    dW3.resize(dW3.size() * p, 0);
    MPI_Gather(dW3.data(), dW3.size()/p, MPI_FLOAT, dW3.data(), dW3.size()/p, MPI_FLOAT, root, MPI_COMM_WORLD);

    // dz2 = dyhat * W3.T * relu'(a2)
    t_tmp = std::move(transform( &W3[0], 64, 10 ));
    sub_p = std::move(vector<float>(dyhat.begin() + mpirank * P_SIZE * 10, dyhat.begin() + (mpirank + 1) * P_SIZE * 10));
    vector<float> dz2 = dot(sub_p, t_tmp, P_SIZE, 10, 64)* reluPrime(a2);
    dz2.resize(dz2.size() * p, 0);
    MPI_Allgather(dz2.data(), dz2.size()/p, MPI_FLOAT, dz2.data(), dz2.size()/p, MPI_FLOAT, MPI_COMM_WORLD);

    // dW2 = a1.T * dz2
    t_tmp = std::move(transform( &a1[0], BATCH_SIZE, 128 ));
    sub_p = std::move(vector<float>(t_tmp.begin() + mpirank * 128/p * BATCH_SIZE, t_tmp.begin() + (mpirank + 1) * 128/p * BATCH_SIZE));
    vector<float> dW2 = dot(sub_p, dz2, 128/p, BATCH_SIZE, 64);
    dW2.resize(dW2.size() * p, 0);
    MPI_Gather(dW2.data(), dW2.size()/p, MPI_FLOAT, dW2.data(), dW2.size()/p, MPI_FLOAT, root, MPI_COMM_WORLD);

    // dz1 = dz2 * W2.T * relu'(a1)
    t_tmp = std::move(transform( &W2[0], 128, 64 ));
    sub_p = std::move(vector<float>(dz2.begin() + mpirank * P_SIZE * 64, dz2.begin() + (mpirank + 1) * P_SIZE * 64));
    vector<float> dz1 = dot(sub_p, t_tmp, P_SIZE, 64, 128) * reluPrime(a1);
    dz1.resize(dz1.size() * p, 0);
    MPI_Allgather(dz1.data(), dz1.size()/p, MPI_FLOAT, dz1.data(), dz1.size()/p, MPI_FLOAT, MPI_COMM_WORLD);

    // dW1 = X.T * dz1
    t_tmp = std::move(transform( &b_X[0], BATCH_SIZE, 784 ));
    sub_p = std::move(vector<float>(t_tmp.begin() + mpirank * 784/p * BATCH_SIZE, t_tmp.begin() + (mpirank + 1) * 784/p * BATCH_SIZE));
    vector<float> dW1 = dot(sub_p, dz1, 784/p, BATCH_SIZE, 128);
    dW1.resize(dW1.size() * p, 0);
    MPI_Gather(dW1.data(), dW1.size()/p, MPI_FLOAT, dW1.data(), dW1.size()/p, MPI_FLOAT, root, MPI_COMM_WORLD);

    // Updating the parameters
   if (mpirank == 0) {
        W3 = W3 - lr * dW3;
        W2 = W2 - lr * dW2;
        W1 = W1 - lr * dW1;
    }
    MPI_Bcast(W1.data(), W1.size(), MPI_FLOAT, root, MPI_COMM_WORLD);
    MPI_Bcast(W2.data(), W2.size(), MPI_FLOAT, root, MPI_COMM_WORLD);
    MPI_Bcast(W3.data(), W3.size(), MPI_FLOAT, root, MPI_COMM_WORLD);

    if ((mpirank == 0) && (i+1) % 100 == 0){
      cout << "Predictions:" << "\n";
      print ( yhat, 10, 10 );
      cout << "Ground truth:" << "\n";
      print ( b_y, 10, 10 );      
      vector<float> loss_m = yhat - b_y;
      float loss = 0.0;
      for (unsigned k = 0; k < BATCH_SIZE*10; ++k){
        loss += loss_m[k]*loss_m[k];
      }      
      t2 = std::chrono::system_clock::now();
      chrono::duration<double> elapsed_seconds = t2-t1;
      double ticks = elapsed_seconds.count();
      cout << "Iteration #: "  << i << endl;
      cout << "Iteration Time: "  << ticks << "s" << endl;
      cout << "Loss: " << loss/BATCH_SIZE << endl;
      cout << "*******************************************" << endl;
    };      
  };
  MPI_Finalize();
  return 0;
}
