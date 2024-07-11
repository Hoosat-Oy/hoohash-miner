#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <string.h>

#define OUT_LEN 32
#define KEY_LEN 32
#define BLOCK_LEN 64
#define CHUNK_LEN 1024

#define CHUNK_START (1 << 0)
#define CHUNK_END (1 << 1)
#define PARENT (1 << 2)
#define ROOT (1 << 3)
#define KEYED_HASH (1 << 4)
#define DERIVE_KEY_CONTEXT (1 << 5)
#define DERIVE_KEY_MATERIAL (1 << 6)

__device__ __constant__ uint32_t IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

__device__ __constant__ uint8_t MSG_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

__device__ void g(uint32_t *state, int a, int b, int c, int d, uint32_t mx, uint32_t my)
{
    state[a] = state[a] + state[b] + mx;
    state[d] = __funnelshift_r(state[d] ^ state[a], state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = __funnelshift_r(state[b] ^ state[c], state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = __funnelshift_r(state[d] ^ state[a], state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = __funnelshift_r(state[b] ^ state[c], state[b] ^ state[c], 7);
}

__device__ void round(uint32_t *state, const uint32_t *m)
{
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

__device__ void permute(uint32_t *m)
{
    uint32_t permuted[16];
    for (int i = 0; i < 16; i++)
    {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    memcpy(m, permuted, 16 * sizeof(uint32_t));
}

__device__ void compress(const uint32_t *chaining_value,
                         const uint32_t *block_words,
                         uint64_t counter,
                         uint32_t block_len,
                         uint32_t flags,
                         uint32_t *out)
{
    uint32_t state[16] = {
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0], IV[1], IV[2], IV[3],
        (uint32_t)counter, (uint32_t)(counter >> 32), block_len, flags};
    uint32_t block[16];
    memcpy(block, block_words, 16 * sizeof(uint32_t));

    round(state, block); // round 1
    permute(block);
    round(state, block); // round 2
    permute(block);
    round(state, block); // round 3
    permute(block);
    round(state, block); // round 4
    permute(block);
    round(state, block); // round 5
    permute(block);
    round(state, block); // round 6
    permute(block);
    round(state, block); // round 7

    for (int i = 0; i < 8; i++)
    {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }

    memcpy(out, state, 16 * sizeof(uint32_t));
}

__device__ void words_from_little_endian_bytes(const uint8_t *bytes, uint32_t *words, size_t words_len)
{
    for (size_t i = 0; i < words_len; i++)
    {
        words[i] = ((uint32_t)bytes[4 * i + 0] << 0) |
                   ((uint32_t)bytes[4 * i + 1] << 8) |
                   ((uint32_t)bytes[4 * i + 2] << 16) |
                   ((uint32_t)bytes[4 * i + 3] << 24);
    }
}

typedef struct ChunkState
{
    uint32_t chaining_value[8];
    uint64_t chunk_counter;
    uint8_t block[BLOCK_LEN];
    uint8_t block_len;
    uint8_t blocks_compressed;
    uint32_t flags;
} ChunkState;

__device__ void chunk_state_init(ChunkState *self, const uint32_t *key_words, uint64_t chunk_counter, uint32_t flags)
{
    memcpy(self->chaining_value, key_words, 8 * sizeof(uint32_t));
    self->chunk_counter = chunk_counter;
    memset(self->block, 0, BLOCK_LEN);
    self->block_len = 0;
    self->blocks_compressed = 0;
    self->flags = flags;
}

__device__ size_t chunk_state_len(const ChunkState *self)
{
    return BLOCK_LEN * self->blocks_compressed + self->block_len;
}

__device__ uint32_t chunk_state_start_flag(const ChunkState *self)
{
    return self->blocks_compressed == 0 ? CHUNK_START : 0;
}

__device__ void chunk_state_update(ChunkState *self, const uint8_t *input, size_t input_len)
{
    while (input_len > 0)
    {
        if (self->block_len == BLOCK_LEN)
        {
            uint32_t block_words[16];
            words_from_little_endian_bytes(self->block, block_words, 16);
            uint32_t out[16];
            compress(self->chaining_value, block_words, self->chunk_counter,
                     BLOCK_LEN, self->flags | chunk_state_start_flag(self), out);
            memcpy(self->chaining_value, out, 8 * sizeof(uint32_t));
            self->blocks_compressed++;
            memset(self->block, 0, BLOCK_LEN);
            self->block_len = 0;
        }

        size_t want = BLOCK_LEN - self->block_len;
        size_t take = (want < input_len) ? want : input_len;
        memcpy(self->block + self->block_len, input, take);
        self->block_len += take;
        input += take;
        input_len -= take;
    }
}

typedef struct Output
{
    uint32_t input_chaining_value[8];
    uint32_t block_words[16];
    uint64_t counter;
    uint32_t block_len;
    uint32_t flags;
} Output;

__device__ void output_chaining_value(const Output *self, uint32_t *out)
{
    uint32_t out_block[16];
    compress(self->input_chaining_value, self->block_words, self->counter,
             self->block_len, self->flags, out_block);
    memcpy(out, out_block, 8 * sizeof(uint32_t));
}

__device__ void output_root_bytes(const Output *self, uint8_t *out, size_t out_len)
{
    uint64_t output_block_counter = 0;
    while (out_len > 0)
    {
        uint32_t words[16];
        compress(self->input_chaining_value, self->block_words, output_block_counter,
                 self->block_len, self->flags | ROOT, words);
        for (size_t word = 0; word < 16 && out_len > 0; word++)
        {
            for (size_t byte = 0; byte < 4 && out_len > 0; byte++)
            {
                *out = (uint8_t)(words[word] >> (8 * byte));
                out++;
                out_len--;
            }
        }
        output_block_counter++;
    }
}

__device__ void chunk_state_output(const ChunkState *self, Output *out)
{
    uint32_t block_words[16];
    words_from_little_endian_bytes(self->block, block_words, 16);
    memcpy(out->input_chaining_value, self->chaining_value, 8 * sizeof(uint32_t));
    memcpy(out->block_words, block_words, 16 * sizeof(uint32_t));
    out->counter = self->chunk_counter;
    out->block_len = self->block_len;
    out->flags = self->flags | chunk_state_start_flag(self) | CHUNK_END;
}

__device__ void parent_output(
    const uint32_t *left_child_cv,
    const uint32_t *right_child_cv,
    const uint32_t *key_words,
    uint32_t flags,
    Output *out)
{
    memcpy(out->input_chaining_value, key_words, 8 * sizeof(uint32_t));
    memcpy(out->block_words, left_child_cv, 8 * sizeof(uint32_t));
    memcpy(out->block_words + 8, right_child_cv, 8 * sizeof(uint32_t));
    out->counter = 0;
    out->block_len = BLOCK_LEN;
    out->flags = PARENT | flags;
}

__device__ void parent_cv(
    const uint32_t *left_child_cv,
    const uint32_t *right_child_cv,
    const uint32_t *key_words,
    uint32_t flags,
    uint32_t *out)
{
    Output parent_output_var;
    parent_output(left_child_cv, right_child_cv, key_words, flags, &parent_output_var);
    output_chaining_value(&parent_output_var, out);
}

typedef struct Hasher
{
    ChunkState chunk_state;
    uint32_t key_words[8];
    uint32_t cv_stack[54][8];
    uint8_t cv_stack_len;
    uint32_t flags;
} Hasher;

__device__ void hasher_init(Hasher *self, const uint32_t *key_words, uint32_t flags)
{
    chunk_state_init(&self->chunk_state, key_words, 0, flags);
    memcpy(self->key_words, key_words, 8 * sizeof(uint32_t));
    self->cv_stack_len = 0;
    self->flags = flags;
}

__device__ void hasher_new_internal(Hasher *self, const uint *key_words, uint flags)
{
    hasher_init(self, key_words, flags);
}

__device__ void hasher_new(Hasher *self)
{
    uint iv_copy[8];
    for (int i = 0; i < 8; i++)
    {
        iv_copy[i] = IV[i];
    }
    hasher_new_internal(self, iv_copy, 0);
}

__device__ void hasher_new_keyed(Hasher *self, const uchar key[KEY_LEN])
{
    uint key_words[8];
    words_from_little_endian_bytes(key, key_words, 8);
    hasher_new_internal(self, key_words, KEYED_HASH);
}

__device__ void hasher_push_stack(Hasher *self, const uint32_t *cv)
{
    memcpy(self->cv_stack[self->cv_stack_len], cv, 8 * sizeof(uint32_t));
    self->cv_stack_len++;
}

__device__ void hasher_pop_stack(Hasher *self, uint32_t *out)
{
    self->cv_stack_len--;
    memcpy(out, self->cv_stack[self->cv_stack_len], 8 * sizeof(uint32_t));
}

__device__ void hasher_add_chunk_chaining_value(Hasher *self, uint32_t *new_cv, uint64_t total_chunks)
{
    while ((total_chunks & 1) == 0)
    {
        uint32_t parent_node[8];
        hasher_pop_stack(self, parent_node);
        parent_cv(parent_node, new_cv, self->key_words, self->flags, new_cv);
        total_chunks >>= 1;
    }
    hasher_push_stack(self, new_cv);
}

__device__ void hasher_update(Hasher *self, const uint8_t *input, size_t input_len)
{
    while (input_len > 0)
    {
        if (chunk_state_len(&self->chunk_state) == CHUNK_LEN)
        {
            uint32_t chunk_cv[8];
            Output output;
            chunk_state_output(&self->chunk_state, &output);
            output_chaining_value(&output, chunk_cv);
            uint64_t total_chunks = self->chunk_state.chunk_counter + 1;
            hasher_add_chunk_chaining_value(self, chunk_cv, total_chunks);
            chunk_state_init(&self->chunk_state, self->key_words, total_chunks, self->flags);
        }

        size_t want = CHUNK_LEN - chunk_state_len(&self->chunk_state);
        size_t take = (want < input_len) ? want : input_len;
        chunk_state_update(&self->chunk_state, input, take);
        input += take;
        input_len -= take;
    }
}

__device__ void hasher_finalize(const Hasher *self, uint8_t *out, size_t out_len)
{
    Output output;
    chunk_state_output(&self->chunk_state, &output);

    size_t parent_nodes_remaining = self->cv_stack_len;
    while (parent_nodes_remaining > 0)
    {
        parent_nodes_remaining--;
        parent_output(
            self->cv_stack[parent_nodes_remaining],
            output.input_chaining_value,
            self->key_words,
            self->flags,
            &output);
    }

    output_root_bytes(&output, out, out_len);
}
