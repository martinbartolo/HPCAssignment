#include <stdio.h>
#include <omp.h>

int main()
{
    omp_set_num_threads(4);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                 printf("Section 1: I am thread %d\n", omp_get_thread_num());
                 printf("Section 1: I am thread %d\n", omp_get_thread_num());
            }

            #pragma omp section
            {
             printf("Section 2: I am thread %d\n", omp_get_thread_num());
             printf("Section 2: I am thread %d\n", omp_get_thread_num());
            }
        }
}