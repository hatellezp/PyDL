//
// Created by horacio on 15/09/2020.
//
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_linalg.h>

#include <fftw3.h>

#include <stdio.h>
#include <errno.h>

// I will use this ?
#define VALGRAPH_TOLERANCE 0.000001
// the method for computing the bound
#define FULL_V 0
#define SINGLE_V 1

#define UR_FORWARD 0
#define UR_BACKWARDS 1

#define M_SCALER 1.1
#define B_TRANSLATER 1.

#ifndef PYDL_CORE_VALGRAPH_H
#define PYDL_CORE_VALGRAPH_H

enum State {
    UNKOWN = -1,
    SUCCESS = 0,
    FAILURE = 1,
};

typedef struct {
   enum State state;
   double value;
   char* message;
}result;

void say_hello();
void test_allocation();
void test_double_array_sum(size_t n, const double * a, const double * b, double * c);

result solve_dynamic_array(size_t n, double ** arr, double ** container, double a, double b, double c);
result solve_static_array(size_t n, double * arr[n*n], double * container[n], double a, double b, double c);

result find_bound_dynamic(size_t n, double *(* arr), double tolerance);
result find_bound_static(size_t n, double (* arr)[n*n], double tolerance);

result find_bound_dynamic_noref(size_t n, double *arr, double tolerance);
result find_bound_static_noref(size_t n, double arr[n*n], double tolerance);

double find_bound_dynamic_noref_only_double(size_t n, double *arr, double tolerance);

#endif //PYDL_CORE_VALGRAPH_H
