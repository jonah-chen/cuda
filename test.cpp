#include <stdio.h>
#include <stdlib.h>

int main(){
   int *a = new int[3];
   a[0] = 1;
   a[1] = 2;
   a[2] = 3;

   unsigned char* b = (unsigned char*)a;
   
}