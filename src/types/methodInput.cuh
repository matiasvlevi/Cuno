#ifndef METHODINPUT_H
#define METHODINPUT_H

class MethodInput {
public:
  float *a;
  float *b;
  int N, M, P;
  int outputLength;
  MethodInput(int N, int M, int P) {
    this->a = (float*)malloc(sizeof(float) * M * N);
    this->b = (float*)malloc(sizeof(float) * N * P);
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