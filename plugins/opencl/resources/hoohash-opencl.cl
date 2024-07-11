#include "blake3_compact.h"

typedef uchar Hash[32];

typedef union _uint256_t {
    ulong number[4];
    uchar hash[32];
} uint256_t;

#define BLOCKDIM 1024
#define MATRIX_SIZE 64
#define HALF_MATRIX_SIZE 32
#define QUARTER_MATRIX_SIZE 16
#define HASH_HEADER_SIZE 72
#define HASH_SIZE 32

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

#define LT_U256(X,Y) ((X).number[3] != (Y)->number[3] ? (X).number[3] < (Y)->number[3] : (X).number[2] != (Y)->number[2] ? (X).number[2] < (Y)->number[2] : (X).number[1] != (Y)->number[1] ? (X).number[1] < (Y)->number[1] : (X).number[0] < (Y)->number[0])
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32

__global unsigned char matrix[MATRIX_SIZE][MATRIX_SIZE];
__global unsigned char hash_header[HASH_HEADER_SIZE];
__global uint256_t *target;

// Xoshiro256** implementation
typedef struct {
    ulong s[4];
} xoshiro256starstar_state;

ulong rotl(const ulong x, int k) {
    return (x << k) | (x >> (64 - k));
}

ulong xoshiro256starstar_next(xoshiro256starstar_state *state) {
    const ulong result = rotl(state->s[1] * 5, 7) * 9;
    const ulong t = state->s[1] << 17;

    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];

    state->s[2] ^= t;
    state->s[3] = rotl(state->s[3], 45);

    return result;
}

static void blake3(uchar* out, const uchar* in) {
    Hasher hasher;
    hasher_new(&hasher);
    hasher_update(&hasher, in, 80);
    hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}

float MediumComplexNonLinear(float x) {
    return exp(sin(x) + cos(x));
}

float IntermediateComplexNonLinear(float x) {
    if (x == M_PI_F/2 || x == 3*M_PI_F/2) {
        return 0.0f; // Avoid singularity
    }
    return sin(x) * cos(x) * tan(x);
}

float HighComplexNonLinear(float x) {
    return exp(x) * log(x + 1);
}

float ComplexNonLinear(float x) {
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

void matrixMultiplication(__global unsigned char matrix[MATRIX_SIZE][MATRIX_SIZE], const uchar* hash, uchar* result) {
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
        ushort high = (ushort)fmod(product[2*i], 16);
        ushort low = (ushort)fmod(product[2*i+1], 16);
        result[i] = hash[i] ^ ((high << 4) | low);
    }
}

__kernel void heavy_hash(const ulong nonce_mask, const ulong nonce_fixed, const ulong nonces_len, 
                         uchar random_type, __global xoshiro256starstar_state* states, __global ulong *final_nonce) {
    int nonceId = get_global_id(0);
    if (nonceId < nonces_len) {
        if (nonceId == 0) *final_nonce = 0;
        ulong nonce;
        switch (random_type) {
            case RANDOM_LEAN:
                nonce = ((__global ulong *)states)[0] ^ nonceId;
                break;
            case RANDOM_XOSHIRO:
            default:
                nonce = xoshiro256starstar_next(&states[nonceId]);
                break;
        }
        nonce = (nonce & nonce_mask) | nonce_fixed;

        uchar input[80];
        for (int i = 0; i < HASH_HEADER_SIZE; i++) {
            input[i] = hash_header[i];
        }
        for (int i = 0; i < 8; i++) {
            input[HASH_HEADER_SIZE + i] = ((uchar *)(&nonce))[i];
        }

        uint256_t result;
        blake3(result.hash, input);

        uchar multiplied[HASH_SIZE];
        matrixMultiplication(matrix, result.hash, multiplied);

        blake3(result.hash, multiplied);

        if (LT_U256(&result, target)) {
            atom_cmpxchg((volatile __global ulong*)final_nonce, 0, nonce);
        }
    }
}