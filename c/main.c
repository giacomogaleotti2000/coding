#include <stdio.h>

// when this function is called, variables a and b are created and then when the function is returned the variables are "destroed"
int sum(int a, int b) {
    return a+b;
}

int main(void) { // void indicates that the function doesnt have arguments
    printf("hello world %d\n", sum(10, 20));
    return 0;
}