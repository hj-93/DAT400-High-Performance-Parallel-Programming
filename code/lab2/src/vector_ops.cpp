#include <random>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iostream>
#include "vector_ops.h" 

std::chrono::microseconds AutoProfiler::totalExecTime = std::chrono::microseconds(0);
extern int num_partitions;

//#define BLOCK_TILE
#define USE_PTHREAD

#ifdef USE_PTHREAD
struct gemm_thread_args
{
  vector<float>* output;
  const vector<float>* m1;
  const vector<float>* m2;
  int m1_rows;
  int m1_columns;
  int m2_columns;
  int row_start;
  int row_end;
};

void *dot_block (void *args) {
    gemm_thread_args* curr_args = (gemm_thread_args*) args;
    for( int row = curr_args->row_start; row < curr_args->row_end; ++row ) {
        for( int k = 0; k < curr_args->m1_columns; ++k ) {
            for( int col = 0; col < curr_args->m2_columns; ++col ) {
               (*curr_args->output)[row * curr_args->m2_columns + col ] += (*curr_args->m1)[ row * curr_args->m1_columns + k ] * (*curr_args->m2)[k * curr_args->m2_columns + col];
            }
        }
    }
    return nullptr;
}
#endif


using namespace std;
void print ( const vector <float>& m, int n_rows, int n_columns ) {
    
    /*  "Couts" the input vector as n_rows x n_columns matrix.
     Inputs:
     m: vector, matrix of size n_rows x n_columns
     n_rows: int, number of rows in the left matrix m1
     n_columns: int, number of columns in the left matrix m1
     */
    
    for( int i = 0; i != n_rows; ++i ) {
        for( int j = 0; j != n_columns; ++j ) {
            cout << m[ i * n_columns + j ] << " ";
        }
        cout << '\n';
    }
    cout << endl;
}

int argmax ( const vector <float>& m ) {
    return distance(m.begin(), max_element(m.begin(), m.end()));
}


vector<float> random_vector(const int size)
{
    /*  Generates a random vector with uniform distribution
     Inputs:
     size: the vector size
     */
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> distribution(0.0, 0.05);
    static default_random_engine generator;

    vector<float> data(size);
    generate(data.begin(), data.end(), [&]() { return distribution(generator); });
    return data;
}

vector <float> operator+(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the elementwise sum of two vectors.
     Inputs:
     m1: a vector
     m2: a vector
     Output: a vector, sum of the vectors m1 and m2.
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m1[i] + m2[i];
    };
    
    return sum;
}

vector <float> operator-(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the difference between two vectors.
     Inputs:
     m1: vector
     m2: vector
     Output: vector, m1 - m2, difference between two vectors m1 and m2.
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m1[i] - m2[i];
    };
    
    return difference;
}

vector <float> operator*(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the product of two vectors (elementwise multiplication).
     Inputs:
     m1: vector
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1[i] * m2[i];
    };
    
    return product;
}

vector <float> operator*(const float m1, const vector <float>& m2){
    
    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1 * m2[i];
    };
    
    return product;
}

vector <float> operator/(const vector <float>& m2, const float m1){
    
    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m2[i] / m1;
    };
    
    return product;
}

vector <float> transform (float *m, const int C, const int R) {
    
    /*  Returns a transpose matrix of input matrix.
     Inputs:
     m: vector, input matrix
     C: int, number of columns in the input matrix
     R: int, number of rows in the input matrix
     Output: vector, transpose matrix mT of input matrix m
     */
    
    vector <float> mT (C*R);
    
    for(unsigned n = 0; n != C*R; n++) {
        unsigned i = n/C;
        unsigned j = n%C;
        mT[n] = m[R*j + i];
    }
    
    return mT;
}

vector <float> dot (const vector <float>& m1, const vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns) {
    
    /*  Returns the product of two matrices: m1 x m2.
     Inputs:
     m1: vector, left matrix of size m1_rows x m1_columns
     m2: vector, right matrix of size m1_columns x m2_columns (the number of rows in the right matrix
     must be equal to the number of the columns in the left one)
     m1_rows: int, number of rows in the left matrix m1
     m1_columns: int, number of columns in the left matrix m1
     m2_columns: int, number of columns in the right matrix m2
     Output: vector, m1 * m2, product of two vectors m1 and m2, a matrix of size m1_rows x m2_columns
     */

    AutoProfiler profiler;

    vector <float> output (m1_rows*m2_columns, 0);
#if defined(BLOCK_TILE)
    const int block_size = 64 / sizeof(float); // 64 = common cache line size
    int N = m1_rows;
    int M = m1_columns;
    int K = m2_columns;
    for (int i0 = 0; i0 < N; i0 += block_size) {
        int imax = i0 + block_size > N ? N : i0 + block_size;

        for (int j0 = 0; j0 < M; j0 += block_size) {
            int jmax = j0 + block_size > M ? M : j0 + block_size;

            for (int k0 = 0; k0 < K; k0 += block_size) {
                int kmax = k0 + block_size > K ? K : k0 + block_size;

                for (int j1 = j0; j1 < jmax; ++j1) {
                    int sj = K * j1;

                    for (int i1 = i0; i1 < imax; ++i1) {
                        int mi = M * i1;
                        int ki = K * i1;

                        for (int k1 = k0; k1 < kmax; ++k1) {
                            output[ki + k1] += m1[mi + j1] * m2[sj + k1];
                        }
                    }
                }
            }
        }
    }
#elif defined(USE_PTHREAD)
    pthread_t threads[num_partitions];

    for (int i = 0; i < num_partitions; ++i) {
      auto row_start = i * m1_rows/num_partitions;
      auto row_end   = (i == num_partitions - 1) ?  m1_rows : row_start + m1_rows/num_partitions;

      gemm_thread_args* args = new gemm_thread_args( { &output,
                                                       &m1,
                                                       &m2,
                                                       m1_rows,
                                                       m1_columns,
                                                       m2_columns,
                                                       row_start,
                                                       row_end,
                                                     } );

      pthread_create( &threads[i], NULL, &dot_block, args );
    }
    for (int i = 0; i < num_partitions; ++i) {
      pthread_join( threads[i], NULL );
    }
#else
    for( int row = 0; row < m1_rows; ++row ) {
        for( int k = 0; k < m1_columns; ++k ) {
            for( int col = 0; col < m2_columns; ++col ) {
                output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
            }
        }
    }
#endif
  return output;
}



