#include <stdint.h>
#include <assert.h>
#include "keccak-tiny.c"
#include "xoshiro256starstar.c"
#include "blake3.h" // Ensure you have a CUDA-compatible Blake3 implementation

#define M_PI 3.14159265358979323846f
#define MATRIX_SIZE 128
#define HALF_MATRIX_SIZE 64
#define QUARTER_MATRIX_SIZE 32
#define HASH_SIZE 32
#define HASH_HEADER_SIZE 72
#define BLOCKDIM 1024

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

__constant__ uint16_t matrix[MATRIX_SIZE][MATRIX_SIZE];
__constant__ uint8_t hash_header[HASH_HEADER_SIZE];
__constant__ uint256_t target;

__device__ float ComplexNonLinear(float x) {
    float transformFactor = fmodf(x, 1.0f);
    if (x < 1) {
        if (transformFactor < 0.25f)
            return expf(sinf(x + (1 + transformFactor)) + cosf(x + (1 + transformFactor)));
        else if (transformFactor < 0.5f)
            return expf(sinf(x - (1 + transformFactor)) + cosf(x - (1 + transformFactor)));
        else if (transformFactor < 0.75f)
            return expf(sinf(x * (1 + transformFactor)) + cosf(x * (1 + transformFactor)));
        else
            return expf(sinf(x / (1 + transformFactor)) + cosf(x / (1 + transformFactor)));
    } else if (x < 10) {
        if (x == M_PI/2 || x == 3*M_PI/2) return 0.0f;
        if (transformFactor < 0.25f)
            return sinf(x + (1 + transformFactor)) * cosf(x + (1 + transformFactor)) * tanf(x + (1 + transformFactor));
        else if (transformFactor < 0.5f)
            return sinf(x - (1 + transformFactor)) * cosf(x - (1 + transformFactor)) * tanf(x - (1 + transformFactor));
        else if (transformFactor < 0.75f)
            return sinf(x * (1 + transformFactor)) * cosf(x * (1 + transformFactor)) * tanf(x * (1 + transformFactor));
        else
            return sinf(x / (1 + transformFactor)) * cosf(x / (1 + transformFactor)) * tanf(x / (1 + transformFactor));
    } else {
        if (transformFactor < 0.25f)
            return expf(x + (1 + transformFactor)) * logf(x + (1 + transformFactor) + 1);
        else if (transformFactor < 0.5f)
            return expf(x - (1 + transformFactor)) * logf(x - (1 + transformFactor) + 1);
        else if (transformFactor < 0.75f)
            return expf(x * (1 + transformFactor)) * logf(x * (1 + transformFactor) + 1);
        else
            return expf(x / (1 + transformFactor)) * logf(x / (1 + transformFactor) + 1);
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

__device__ bool LT_U256(const uint256_t* a, const uint256_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->number[i] != b->number[i]) {
            return a->number[i] < b->number[i];
        }
    }
    return false;
}

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

        uint8_t hash[HASH_SIZE];
        blake3(input, 80, hash, HASH_SIZE);

        uint8_t multiplied[HASH_SIZE];
        matrixMultiplication(hash, multiplied);

        blake3(multiplied, HASH_SIZE, hash, HASH_SIZE);

        uint256_t result;
        memcpy(result.hash, hash, HASH_SIZE);

        if (LT_U256(&result, &target)) {
            atomicCAS((unsigned long long int*)final_nonce, 0, (unsigned long long int)nonce);
        }
    }
}

extern "C" {
    __global__ void heavy_hash(const uint64_t nonce_mask, const uint64_t nonce_fixed, const uint64_t nonces_len, uint8_t random_type, void* states, uint64_t *final_nonce);
}