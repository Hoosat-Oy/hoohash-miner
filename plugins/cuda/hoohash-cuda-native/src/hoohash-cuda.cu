#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include "keccak-tiny.c"
#include "xoshiro256starstar.c"
#include "blake3_compact.h" 

typedef uint8_t Hash[32];

typedef union _uint256_t {
    uint64_t number[4];
    uint8_t hash[32];
} uint256_t;

#define BLOCKDIM 1024
#define MATRIX_SIZE 64
#define HALF_MATRIX_SIZE 32
#define QUARTER_MATRIX_SIZE 16
#define HASH_HEADER_SIZE 72
#define HASH_SIZE 32
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

#define LT_U256(X,Y) ((X)->number[3] != (Y)->number[3] ? (X)->number[3] < (Y)->number[3] : (X)->number[2] != (Y)->number[2] ? (X)->number[2] < (Y)->number[2] : (X)->number[1] != (Y)->number[1] ? (X)->number[1] < (Y)->number[1] : (X)->number[0] < (Y)->number[0])


__constant__ uint8_t matrix[MATRIX_SIZE][MATRIX_SIZE];
__constant__ uint8_t hash_header[HASH_HEADER_SIZE];
__constant__ uint256_t target;


__device__ void blake3(uint8_t* out, const uint8_t* in) {
    Hasher hasher;
    hasher_new(&hasher);
    hasher_update(&hasher, in, 80);
    hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}

__device__ float MediumComplexNonLinear(float x) {
    return exp(sin(x) + cos(x));
}

__device__ float IntermediateComplexNonLinear(float x) {
    if (x == M_PI/2 || x == 3*M_PI/2) {
        return 0.0f; // Avoid singularity
    }
    return sin(x) * cos(x) * tan(x);
}

__device__ float HighComplexNonLinear(float x) {
    return exp(x) * log(x + 1);
}

__device__ float ComplexNonLinear(float x) {
    float transformFactor = fmod(x, 1.0f);
    if (x < 1) {
        if (transformFactor < 0.25f) {
            return MediumComplexNonLinear(x + (1 + transformFactor));
        } else if (transformFactor < 0.5f) {
            return MediumComplexNonLinear(x - (1 + transformFactor));
        } else if (transformFactor < 0.75f) {
            return MediumComplexNonLinear(x * (1 + transformFactor));
        } else {
            return MediumComplexNonLinear(x / (1 + transformFactor));
        }
    } else if (x < 10) {
        if (transformFactor < 0.25f) {
            return IntermediateComplexNonLinear(x + (1 + transformFactor));
        } else if (transformFactor < 0.5f) {
            return IntermediateComplexNonLinear(x - (1 + transformFactor));
        } else if (transformFactor < 0.75f) {
            return IntermediateComplexNonLinear(x * (1 + transformFactor));
        } else {
            return IntermediateComplexNonLinear(x / (1 + transformFactor));
        }
    } else {
        if (transformFactor < 0.25f) {
            return HighComplexNonLinear(x + (1 + transformFactor));
        } else if (transformFactor < 0.5f) {
            return HighComplexNonLinear(x - (1 + transformFactor));
        } else if (transformFactor < 0.75f) {
            return HighComplexNonLinear(x * (1 + transformFactor));
        } else {
            return HighComplexNonLinear(x / (1 + transformFactor));
        }
    }
}

__device__ void matrixMultiplication(const uint8_t* hash, uint8_t* result) {
    float vector[MATRIX_SIZE];
    float product[MATRIX_SIZE];

    for (int i = 0; i < HASH_SIZE; i++) {
        vector[2*i] = (float)(hash[i] >> 4);
        vector[2*i+1] = (float)(hash[i] & 0x0F);
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        float sum = 0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            sum += (float)matrix[i][j] * ComplexNonLinear(vector[j]);
        }
        product[i] = sum;
    }

    for (int i = 0; i < HASH_SIZE; i++) {
        uint16_t high = (uint16_t)fmodf(product[2*i], 16);
        uint16_t low = (uint16_t)fmodf(product[2*i+1], 16);
        result[i] = hash[i] ^ ((high << 4) | low);
    }
}

extern "C" {
    __global__ void heavy_hash(const uint64_t nonce_mask, const uint64_t nonce_fixed, const uint64_t nonces_len, uint8_t random_type, void* states, uint64_t *final_nonce) {
        int nonceId = threadIdx.x + blockIdx.x * blockDim.x;
        if (nonceId < nonces_len) {
            if (nonceId == 0) *final_nonce = 0;
            uint64_t nonce;
            switch (random_type) {
                case RANDOM_LEAN:
                    nonce = ((uint64_t *)states)[0] ^ nonceId;
                    break;
                case RANDOM_XOSHIRO:
                default:
                    nonce = xoshiro256_next(((ulonglong4 *)states) + nonceId);
                    break;
            }
            nonce = (nonce & nonce_mask) | nonce_fixed;

            uint8_t input[80];
            memcpy(input, hash_header, HASH_HEADER_SIZE);
            memcpy(input + HASH_HEADER_SIZE, (uint8_t *)(&nonce), 8);

            uint256_t result;
            blake3(result.hash, input);

            uint8_t multiplied[HASH_SIZE];
            matrixMultiplication(result.hash, multiplied);

            blake3(result.hash, multiplied);
            
            if (LT_U256(&result, &target)) {
                atomicCAS((unsigned long long int*)final_nonce, 0, (unsigned long long int)nonce);
            }
        }
    }
}