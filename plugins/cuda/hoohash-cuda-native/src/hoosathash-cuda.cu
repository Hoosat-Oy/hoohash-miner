#include <stdint.h>
#include <assert.h>
#include "keccak-tiny.c"
#include "xoshiro256starstar.c"
#include "blake3.h" // Make sure you have the CUDA-compatible Blake3 implementation

#define M_PI 3.14159265358979323846f
#define MATRIX_SIZE 128
#define HASH_SIZE 32
#define BLOCKDIM 256

__constant__ uint16_t matrix[MATRIX_SIZE][MATRIX_SIZE];
__constant__ uint8_t hash_header[72];
__constant__ uint256_t target;

__device__ float MediumComplexNonLinear(float x) {
    return expf(sinf(x) + cosf(x));
}

__device__ float IntermediateComplexNonLinear(float x) {
    if (x == M_PI/2 || x == 3*M_PI/2) {
        return 0.0f;
    }
    return sinf(x) * cosf(x) * tanf(x);
}

__device__ float HighComplexNonLinear(float x) {
    return expf(x) * logf(x + 1);
}

__device__ float ComplexNonLinear(float x) {
    float transformFactor = fmodf(x, 1.0f);
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

__device__ void matrixMultiplication(const uint16_t mat[MATRIX_SIZE][MATRIX_SIZE], const uint8_t* hash, uint8_t* result) {
    float vector[MATRIX_SIZE];
    float product[MATRIX_SIZE] = {0};

    for (int i = 0; i < HASH_SIZE; i++) {
        vector[2*i] = (float)(hash[i] >> 4);
        vector[2*i+1] = (float)(hash[i] & 0x0F);
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            product[i] += (float)mat[i][j] * ComplexNonLinear(vector[j]);
        }
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

__global__ void hoohash_pow(uint64_t nonce_start, uint64_t nonces_per_thread, uint64_t* final_nonce) {
    uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce_base = nonce_start + thread_id * nonces_per_thread;

    uint8_t input[80];
    memcpy(input, hash_header, 72);

    for (uint64_t i = 0; i < nonces_per_thread; i++) {
        uint64_t nonce = nonce_base + i;
        memcpy(input + 72, &nonce, 8);

        uint8_t hash[HASH_SIZE];
        blake3(input, 80, hash, HASH_SIZE);

        uint8_t multiplied[HASH_SIZE];
        matrixMultiplication(matrix, hash, multiplied);

        blake3(multiplied, HASH_SIZE, hash, HASH_SIZE);

        uint256_t result;
        memcpy(result.hash, hash, HASH_SIZE);

        if (LT_U256(&result, &target)) {
            atomicCAS((unsigned long long int*)final_nonce, 0, (unsigned long long int)nonce);
            return;
        }
    }
}

extern "C" {
    void launch_hoohash_pow(uint64_t nonce_start, uint64_t total_nonces, uint64_t* final_nonce, cudaStream_t stream) {
        uint64_t threads_per_block = BLOCKDIM;
        uint64_t nonces_per_thread = 256;
        uint64_t blocks = (total_nonces + threads_per_block * nonces_per_thread - 1) / (threads_per_block * nonces_per_thread);

        hoohash_pow<<<blocks, threads_per_block, 0, stream>>>(nonce_start, nonces_per_thread, final_nonce);
    }
}