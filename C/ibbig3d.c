#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "ibbig3d.h"

double getScore3D(int *positions, int actualGenes, double *covTen, int noRows, int noCols, int noDepth, double alpha){
	
	double score = 0;
    
	int actualCols = 0;
	int actualDepth = 0;
	
	int *cols = malloc(sizeof(int) * noCols);
	int *depths = malloc(sizeof(int) * noDepth);
	//int cols[noCols];
	//int depths[noDepth];
	
    for (int i = 0; i < actualGenes; i++){
        if (positions[i] < noCols){
            cols[actualCols++] = positions[i];
        }else{
            depths[actualDepth++] = positions[i] - noCols;
        }
    }
	
    if (actualCols == 0 || actualDepth == 0)
        return 0;
	
    for (int r = 0; r < noRows; r++)
	{
        double pScore = 0;
		
        for (int c = 0; c < actualCols; c++)
		{
            for (int d = 0; d < actualDepth; d++)
			{
				int col = cols[c];
                int dep = depths[d];
                int idx = r + noRows * col + noRows * noCols * dep;
                pScore += covTen[idx];
            }
        }
		
		double denom = actualCols * actualDepth;
        double p1 = pScore / denom;
		
        if (p1 > 0.25)
		{
            double eScore;
			
            if (p1 >= 0.999999)
			{
                eScore = 1;
            }
			else
			{
                double p0 = 1 - p1;
                eScore = 1 + p1 * log2(p1) + p0 * log2(p0);
            }
			
            score += pScore * pow(eScore, alpha);
        }
    }
	
	free(cols);
	free(depths);
	
    return score;
}
 
void initializePop(long int *pop, int noPop, double *covTen, int noCovs, int noDepth, int noSigs, double alpha){
   
   int noGenes = noCovs + noDepth;
   
   int index=0;
   int r1,r2,i,j;
   
   //initialize population with zeros
   	for (int i = 0; i < noGenes; i++){
   	  for (int j = 0; j < noPop; j++){
  	     pop[i * noPop + j] = 0; 	
  	  }
   }
   
   double score;
   int actualGenes;
   int x=0;
   
   while (x<(noPop*100) && index<noPop){
      actualGenes=2;
	  r1 = rand() % noCovs; 
      r2 = rand() % noDepth; 
      int positions[2]= {r1, noCovs + r2};   
      score=getScore3D(positions,actualGenes,covTen,noSigs,noCovs,noDepth,alpha);
      //printf("%f %d %d %d\n",score,r1,r2,actualCovs);
      if (score>0){
         pop[r1 * noPop + index] = 1;
         pop[(noCovs + r2) *noPop + index] = 1;
         index++;
      }
      x++;
   }    
}

void mysort(double* scores, int* order, int N){
  int i, j, t1;
  double v, t;

  if(N<=1) return;

  // Partition elements
  v = scores[0];
  i = 0;
  j = N;
  for(;;)
  {
    while(scores[++i] < v && i < N) { }
    while(scores[--j] > v) { }
    if(i >= j) break;
    t = scores[i]; scores[i] = scores[j]; scores[j] = t;
    t1 = order[i]; order[i] = order[j]; order[j] = t1;
  }
  t = scores[i-1]; scores[i-1] = scores[0]; scores[0] = t;
  t1 = order[i-1]; order[i-1] = order[0]; order[0] = t1;
  mysort(scores, order, i-1);
  mysort(scores+i, order+i,  N-i);
}



void sortPop(long int *pop, int noGenes, int noPop, double *scores){
   int i,j;  	
   int order[noPop];
   for (i=0;i<noPop;i++){
      order[i]=i;
   }  
   mysort(scores,order,noPop);
   int sortedPop[noGenes*noPop];
   for (i=0;i<noPop;i++){
      for (j=0;j<noGenes;j++){
         sortedPop[j*noPop+i]=pop[j*noPop+order[i]];	 
      }
   } 
   for (i=0;i<noPop*noGenes;++i){
   	  pop[i]=sortedPop[i];
   }   
   
}

//parent selection
int selectParent(double *SP){
   double r=rand();
   r/=RAND_MAX;
   int index=0;
   while(SP[index]<r){
      index++;
   }
   return(index);
}

//generates the next generation of children
void generateChildren(long int *pop,
                      double *covTen,
                      double *scores,
                      int noPop,
                      int noGenes,
                      int noCovs,
                      int noDepth,
                      int noSigs,
                      double SR,
                      int maxSP,
                      double *SP,
                      double mutation,
                      double alpha)
{

	long int *newPop = malloc(sizeof(long int) * noGenes * noPop);
	//long int newPop[noGenes * noPop];
	double newScores[noPop];

    newScores[0] = scores[noPop - 1];

    for (int i = 0; i < noGenes; i++)
        newPop[i * noPop] = pop[(i + 1) * noPop - 1];

    int sucChildren = 1;
    int index = 1;

    int threshold = RAND_MAX / noGenes;
    int mutThreshold = (double)RAND_MAX * mutation;

    while (sucChildren < noPop)
    {
		index++;

        int par1 = selectParent(SP);
        int par2 = selectParent(SP);

		int child[noGenes];
		int positions[noGenes];

        int actualGenes = 0;

		int crosspoint = rand() % (noGenes - 1) + 1;
        //int crosspoint = rand() % (noGenes - 2) + 1;

        for (int i = 0; i < noGenes; i++)
        {
            if (i < crosspoint)
                child[i] = pop[i * noPop + par1];
            else
                child[i] = pop[i * noPop + par2];

            if (rand() <= mutThreshold && rand() <= threshold)
                child[i] = (child[i] + 1) % 2;

            if (child[i] == 1)
                positions[actualGenes++] = i;
        }

        double childScore = getScore3D(
            positions,
            actualGenes,
            covTen,
            noSigs,
			noCovs,
            noDepth,
            alpha);

		if (childScore > 0)
        {
            newScores[sucChildren] = childScore;

            for (int i = 0; i < noGenes; i++)
                newPop[i * noPop + sucChildren] = child[i];

            sucChildren++;
        }

		/*

        if (childScore > scores[par1] || childScore > scores[par2])
        {
            newScores[sucChildren] = childScore;

            for (int i = 0; i < noGenes; i++)
                newPop[i * noPop + sucChildren] = child[i];

            sucChildren++;
        }
		
		*/
		
    }

    for (int i = 0; i < noGenes * noPop; i++)
        pop[i] = newPop[i];

    for (int i = 0; i < noPop; i++)
        scores[i] = newScores[i];
	
	free(newPop);
	
}

void clusterCovs3DC(double *covTen,
                    int *group,
                    int *noCovs,
                    int *noSigs,
                    int *noDepth,
                    double *alpha,
                    int *noPop,
                    int *maxStag,
                    double *mutation,
                    double *SR,
                    int *max_SP,
                    double *SP)
{

    srand((unsigned)time(NULL));

    int noGenes = (*noCovs) + (*noDepth);

	long int *pop = malloc(sizeof(long int) * noGenes * (*noPop));
	//long int pop[noGenes * (*noPop)];

    initializePop(pop, *noPop, covTen, *noCovs, *noDepth, *noSigs, *alpha);

    double scores[*noPop];

	int positions[noGenes];

    for (int j = 0; j < *noPop; j++)
    {
        int actualGenes = 0;

        for (int i = 0; i < noGenes; i++)
            if (pop[i * (*noPop) + j] == 1)
                positions[actualGenes++] = i;
		
        scores[j] = getScore3D(positions, actualGenes, covTen, *noSigs, *noCovs, *noDepth, *alpha);
    }
	
    sortPop(pop, noGenes, *noPop, scores);

    double maxScore = 0;
    int stag = 0;

    while (stag < *maxStag)
    {
		
        generateChildren(pop, covTen, scores, *noPop, noGenes, *noCovs, *noDepth, *noSigs, *SR, *max_SP, SP, *mutation, *alpha);

        sortPop(pop, noGenes, *noPop, scores);

        if (scores[*noPop - 1] > maxScore)
        {
            stag = 0;
            maxScore = scores[*noPop - 1];
        }
        else
            stag++;
    }

    for (int i = 0; i < noGenes; i++)
        group[i] = pop[(i + 1) * (*noPop) - 1];
	
	free(pop);
	
}


