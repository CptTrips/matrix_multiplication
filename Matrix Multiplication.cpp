/* Matrix Multiplication.cpp : This file contains the 'main' function. Program execution begins and ends there.
    This project is about implementing 'fast' (O(n^\omega) where 2 <= \omega < 3) matrix multiplication algorithms and making their x64 code performant.
*/

#include <iostream>
#include <chrono>

// Matrices are assumed square!
struct Matrix {
    unsigned int n;
    float* array;
    Matrix(unsigned int size, float* input_array) : n(size), array(input_array)
    {}
    Matrix(unsigned int size) : n(size)
    {  
        array = new float[n * n];
    }
    Matrix()
    {
        //std::cout << "empty constructor\n";
        array = nullptr;
        n = 0;
    }
    // Copy constructor
    Matrix(const Matrix& A)
    {
        n = A.n;
        array = new float[n * n];
        for (unsigned int i = 0; i < n * n; i++)
        {
            array[i] = A.array[i];
        }
        std::cout << "copy constructor\n";
    }
    // Assignment operator
    Matrix& operator=(const Matrix& A)
    {
        n = A.n;
		delete[] array;
        array = new float[n * n];
        for (unsigned int i = 0; i < n * n; i++)
        {
            array[i] = A.array[i];
        }

        std::cout << "assignment operator\n";
        return *this;
    }
    // Move constructor
    Matrix(Matrix&& A)
    {
        n = A.n;
        array = A.array;

        A.n = 0;
        A.array = nullptr;
        //std::cout << "move constructor\n";
    }
    // Move assignment
    Matrix& operator=(Matrix&& A)
    {
        if (this != &A)
        {
            delete[] array;

            n = A.n;
            array = A.array;

            A.n = 0;
            A.array = nullptr;
        }
        //std::cout << "move assignment\n";

        return *this;
    }
    ~Matrix()
    {
		delete[] array;
    }
};

void print_mat(const Matrix& A)
{
    for (unsigned int i = 0; i < A.n; i++)
    {
        std::cout << "    ";
        for (unsigned int j = 0; j < A.n; j++)
            std::cout << A.array[j * A.n + i] << " ";
        std::cout << std::endl;
    }
}

float element(const Matrix& A, unsigned int i, unsigned int j)
{
    return A.array[j * A.n + i];
}

Matrix mat_add(const Matrix& A, const Matrix& B)
{
    Matrix C(A.n);

    for (unsigned int i = 0; i < A.n * A.n; i++)
    {
        C.array[i] = B.array[i] + A.array[i];
    }

    return C;
}

Matrix scalar_mul(const float x, const Matrix& A)
{
    unsigned int size = A.n * A.n;

    Matrix B(A.n);

    for (unsigned int i = 0; i < size; i++)
    {
        B.array[i] = x * A.array[i];
    }

    return B;
}

Matrix mat_sub(const Matrix& A, const Matrix& B)
{
    Matrix C(A.n);

    for (unsigned int i = 0; i < A.n * A.n; i++)
    {
        C.array[i] = A.array[i] - B.array[i];
    }

    return C;
}

Matrix random_matrix(unsigned int n)
{

    Matrix A(n);

    for (unsigned int i = 0; i < n * n; i++) {
        A.array[i] = 1 / (float)rand();
    }

    return A;
}

void fill_submatrices(const Matrix& A, Matrix* submatrix_array)
{
        for (unsigned int s = 0; s < 4; s++)
        {
            Matrix* H = &submatrix_array[s];
            H->n = A.n / 2;

            delete[] H->array;

            H->array = new float[H->n * H->n];

            unsigned int row_offset, column_offset;

            if (s == 2 or s == 3)
                row_offset = H->n;
            else
                row_offset = 0;

            if (s == 1 or s == 3)
                column_offset = H->n;
            else
                column_offset = 0;

            for (unsigned int i = 0; i < H->n; i++)
            {
                for (unsigned int j = 0; j < H->n; j++)
                {
                    H->array[j * H->n + i] = A.array[(j + column_offset) * A.n + i + row_offset];
                }
            }
        }

}

void fill_matrix(Matrix& A, Matrix* submatrix_array)
{
        for (unsigned int s = 0; s < 4; s++)
        {
            Matrix* H = &submatrix_array[s];

            unsigned int row_offset, column_offset;

            if (s == 2 or s == 3)
                row_offset = H->n;
            else
                row_offset = 0;

            if (s == 1 or s == 3)
                column_offset = H->n;
            else
                column_offset = 0;

            for (unsigned int i = 0; i < H->n; i++)
            {
                for (unsigned int j = 0; j < H->n; j++)
                {
                    A.array[(j + column_offset) * A.n + i + row_offset] = H->array[j * H->n + i];
                }
            }
        }

}


Matrix strassen_mat_mul(const Matrix& A, const Matrix&  B)
{
    unsigned int size = A.n * A.n;

    Matrix C(A.n);

    // subdivide A & B into 2x2 blocks
    // Assume A.n = B.n = 2^m for some m
    if (A.n > 2)
    {
        Matrix A_submatrix_list[4]; // A11, A12, A21, A22;
        Matrix B_submatrix_list[4];
        
        fill_submatrices(A, A_submatrix_list);
        fill_submatrices(B, B_submatrix_list);

        Matrix M1 = strassen_mat_mul(mat_add(A_submatrix_list[0], A_submatrix_list[3]), mat_add(B_submatrix_list[0], B_submatrix_list[3]));
        Matrix M2 = strassen_mat_mul(mat_add(A_submatrix_list[2], A_submatrix_list[3]), B_submatrix_list[0]);
        Matrix M3 = strassen_mat_mul(A_submatrix_list[0], mat_sub(B_submatrix_list[1], B_submatrix_list[3]));
        Matrix M4 = strassen_mat_mul(A_submatrix_list[3], mat_sub(B_submatrix_list[2], B_submatrix_list[0]));
        Matrix M5 = strassen_mat_mul(mat_add(A_submatrix_list[0], A_submatrix_list[1]), B_submatrix_list[3]);
        Matrix M6 = strassen_mat_mul(mat_sub(A_submatrix_list[2], A_submatrix_list[0]), mat_add(B_submatrix_list[0], B_submatrix_list[1]));
        Matrix M7 = strassen_mat_mul(mat_sub(A_submatrix_list[1], A_submatrix_list[3]), mat_add(B_submatrix_list[2], B_submatrix_list[3]));

        // Fill C
        Matrix C_submatrix_array[4];
        C_submatrix_array[0] = mat_add(M1, mat_add(M4, mat_sub(M7, M5))); //C11
        C_submatrix_array[1] = mat_add(M3, M5); //C12
        C_submatrix_array[2] = mat_add(M2, M4); //C21
        C_submatrix_array[3] = mat_add(M1, mat_sub(mat_add(M3, M6), M2)); //C22

        // Need to use std::array or similar to let the array use move semantics
        //Matrix C_submatrix_array[4] = { C11, C12, C21, C22 };

        fill_matrix(C, C_submatrix_array);
    }
    else if (A.n == 2)
    {
        float M1 = (element(A, 0, 0) + element(A, 1, 1)) * (element(B, 0, 0) + element(B, 1,1));
        float M2 = (element(A, 1, 0) + element(A, 1, 1)) * element(B, 0, 0);
        float M3 = element(A, 0, 0) * (element(B, 0, 1) - element(B, 1, 1));
        float M4 = element(A, 1, 1) * (element(B, 1, 0) - element(B, 0, 0));
        float M5 = (element(A, 0, 0) + element(A, 0, 1)) * element(B, 1, 1);
        float M6 = (element(A, 1, 0) - element(A, 0, 0)) * (element(B, 0, 0) + element(B, 0, 1));
        float M7 = (element(A, 0, 1) - element(A, 1, 1)) * (element(B, 1, 0) + element(B, 1, 1));

        C.array[0] = M1 + M4 - M5 + M7;
        C.array[1] = M2 + M4;
        C.array[2] = M3 + M5;
        C.array[3] = M1 - M2 + M3 + M6;
    }
    else
		throw "matrix decomposition error";


    return C;

}

Matrix naive_mat_mul(const Matrix& A, const Matrix& B)
{
    unsigned int size = A.n * A.n;

    Matrix C(A.n);

    // C row
    for (unsigned int i = 0; i < C.n; i++)
    {
        // C column
        for (unsigned int j = 0; j < C.n; j++)
        {

            float sum = 0;

            // A dot B
            for (unsigned int k = 0; k < A.n; k++)
            {
                sum += element(A, i, k) * element(B, k, j);
            }

            C.array[j * C.n + i] = sum;

        }
        
    }

    return C;

}



int main()
{
    std::cout << "Hello World!\n";

    Matrix identity;
    identity.n = 2;
    identity.array = new float[4] { 1, 0, 0, 1 };
    Matrix x;
    x.n = 2;
    x.array = new float[4] { 0, 1, 1, 0 };
    Matrix y;
    y.n = 2;
    y.array = new float[4] { 0, 1, -1, 0 };
    Matrix z;
    z.n = 2;
    z.array = new float[4] { 1, 0, 0, -1 };

    unsigned int samples = pow(2, 7);

    for (unsigned int n = 2; n <= 8; n++)
    {

        unsigned int mat_size = pow(2, n);

        auto start = std::chrono::high_resolution_clock::now();

        for (unsigned int i = 0; i < samples; i++)
        {

            Matrix A = random_matrix(mat_size);

            Matrix B = random_matrix(mat_size);

            Matrix C = strassen_mat_mul(A, B);
        }

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << mat_size << 'x' << mat_size << " Elapsed time: " << elapsed.count() << " s\n";

    }
}
