#ifndef METHODINPUT_H
#define METHODINPUT_H

class MethodInput {
public:
  double *a;
  double *b;
  int N, M, P;
  int outputLength;
  MethodInput(int N, int M, int P) {
    this->a = (double*)malloc(sizeof(double) * M * N);
    this->b = (double*)malloc(sizeof(double) * N * P);
    this->N = N;
    this->M = M;
    this->P = P;
    this->outputLength = M * P;
  }
  ~MethodInput() {
    free(this->a);
    free(this->b);
  }
};


#endif
