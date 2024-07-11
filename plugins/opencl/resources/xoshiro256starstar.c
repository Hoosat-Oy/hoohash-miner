typedef struct {
    ulong s[4];
} xoshiro256starstar_state;

// Rotate left operation
ulong rotl(const ulong x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Generate the next random number and update the state
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

// Jump function for the generator. This is equivalent
// to 2^128 calls to next(); it can be used to generate 2^128
// non-overlapping subsequences for parallel computations.
void jump(xoshiro256starstar_state *state) {
    static const ulong JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

    ulong s0 = 0;
    ulong s1 = 0;
    ulong s2 = 0;
    ulong s3 = 0;
    for(int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & ((ulong)1 << b)) {
                s0 ^= state->s[0];
                s1 ^= state->s[1];
                s2 ^= state->s[2];
                s3 ^= state->s[3];
            }
            xoshiro256starstar_next(state);
        }
        
    state->s[0] = s0;
    state->s[1] = s1;
    state->s[2] = s2;
    state->s[3] = s3;
}

// Long jump function for the generator. This is equivalent to
// 2^192 calls to next(); it can be used to generate 2^64 starting points,
// from each of which jump() will generate 2^64 non-overlapping
// subsequences for parallel distributed computations.
void long_jump(xoshiro256starstar_state *state) {
    static const ulong LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

    ulong s0 = 0;
    ulong s1 = 0;
    ulong s2 = 0;
    ulong s3 = 0;
    for(int i = 0; i < sizeof(LONG_JUMP) / sizeof(*LONG_JUMP); i++)
        for(int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & ((ulong)1 << b)) {
                s0 ^= state->s[0];
                s1 ^= state->s[1];
                s2 ^= state->s[2];
                s3 ^= state->s[3];
            }
            xoshiro256starstar_next(state);
        }
        
    state->s[0] = s0;
    state->s[1] = s1;
    state->s[2] = s2;
    state->s[3] = s3;
}

// Initialize the state with a seed
void seed_xoshiro256starstar(xoshiro256starstar_state *state, ulong seed) {
    // A simple seed initialization. For better initialization, 
    // use a splitmix64-based initialization as recommended by the authors.
    state->s[0] = seed;
    state->s[1] = seed ^ 0x1234567890abcdef;
    state->s[2] = seed ^ 0xfedcba9876543210;
    state->s[3] = seed ^ 0xf0e1d2c3b4a59687;

    // Perform a few iterations to mix the state
    for (int i = 0; i < 16; i++) {
        xoshiro256starstar_next(state);
    }
}

// Generate a random float in the range [0, 1)
float xoshiro256starstar_next_float(xoshiro256starstar_state *state) {
    return (xoshiro256starstar_next(state) >> 11) * (1.0f / (float)(1ULL << 53));
}

// Example kernel using xoshiro256**
__kernel void random_numbers(__global float *output, ulong seed, int count) {
    int gid = get_global_id(0);
    xoshiro256starstar_state rng_state;

    // Initialize the RNG state for this work item
    seed_xoshiro256starstar(&rng_state, seed + gid);

    // Generate random numbers
    for (int i = gid; i < count; i += get_global_size(0)) {
        output[i] = xoshiro256starstar_next_float(&rng_state);
    }
}