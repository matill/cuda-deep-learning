#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <iostream>


typedef short               i16;
typedef int                 i32;
typedef long                i64;
typedef long long          i128;
typedef unsigned short      u16;
typedef unsigned            u32;
typedef unsigned long       u64;
typedef unsigned long long  u128;
typedef float f32;
typedef double f64;


#define MIN(a, b) \
    ((a) < (b) ? (a) : (b))

#define ASSERT_NOT_NULL(pointer) \
    if (!((pointer) != NULL)) { \
        printf("Assertion error %s:%d. %s is NULL\n", __FILE__, __LINE__, #pointer); \
    }

#define ASSERT_EQ_INT(a, b) \
    if ((a) != (b)) { \
        printf("Assertion error %s:%d. %s (%d) != %s (%d)\n", __FILE__, __LINE__, #a, (a), #b, (b)); \
    }

#define ASSERT_NEQ_INT(a, b) \
    if ((a) == (b)) { \
        printf("Assertion error %s:%d. %s (%d) == %s (%d)\n", __FILE__, __LINE__, #a, (a), #b, (b)); \
    }


#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        printf("Assertion error %s:%d. %s (", __FILE__, __LINE__, #a); \
        std::cout << (a); \
        printf(") != %s (", #b); \
        std::cout << (b); \
        printf(")\n"); \
        exit(-1); \
    }


#define ASSERT_BETWEEN_FLOAT(low, between, high) \
    if (!(low <= between && between <= high)) { \
        printf("Assertion error %s:%d. %s (%f) <= %s (%f) <= %s (%f)", __FILE__, __LINE__, #low, (low), #between, (between), #high, (high)); \
    }



#endif
