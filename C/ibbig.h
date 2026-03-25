#ifndef CLUSTERCOVSC_H
#define CLUSTERCOVSC_H

void clusterCovsC(
    double *covMat, int *group, int *noCovs, int *noSigs, 
    double *alpha, int *noPop, int *maxStag, double *mutation, 
    double *SR, int *max_SP, double *SP);

#endif
