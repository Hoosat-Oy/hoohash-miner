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

__constant uint IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

__constant uchar MSG_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

void g(uint *state, int a, int b, int c, int d, uint mx, uint my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotate(state[d] ^ state[a], 16U);
    state[c] = state[c] + state[d];
    state[b] = rotate(state[b] ^ state[c], 12U);
    state[a] = state[a] + state[b] + my;
    state[d] = rotate(state[d] ^ state[a], 8U);
    state[c] = state[c] + state[d];
    state[b] = rotate(state[b] ^ state[c], 7U);
}

void round(uint *state, const uint *m) {
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

void permute(uint *m) {
    uint permuted[16];
    for (int i = 0; i < 16; i++) {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    for (int i = 0; i < 16; i++) {
        m[i] = permuted[i];
    }
}

void compress(const uint *chaining_value, const uint *block_words,
              ulong counter, uint block_len, uint flags,
              uint *out) {
    uint state[16] = {
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0], IV[1], IV[2], IV[3],
        (uint)counter, (uint)(counter >> 32), block_len, flags
    };
    uint block[16];
    for (int i = 0; i < 16; i++) {
        block[i] = block_words[i];
    }

    round(state, block);
    permute(block);
    round(state, block);
    permute(block);
    round(state, block);
    permute(block);
    round(state, block);
    permute(block);
    round(state, block);
    permute(block);
    round(state, block);
    permute(block);
    round(state, block);

    for (int i = 0; i < 8; i++) {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }

    for (int i = 0; i < 16; i++) {
        out[i] = state[i];
    }
}

void words_from_little_endian_bytes(const uchar *bytes, uint *words, int words_len) {
    for (int i = 0; i < words_len; i++) {
        words[i] = ((uint)bytes[i*4]) | ((uint)bytes[i*4 + 1] << 8) |
                   ((uint)bytes[i*4 + 2] << 16) | ((uint)bytes[i*4 + 3] << 24);
    }
}

typedef struct {
    uint input_chaining_value[8];
    uint block_words[16];
    ulong counter;
    uint block_len;
    uint flags;
} Output;

void output_chaining_value(Output *self, uint *out) {
    compress(self->input_chaining_value, self->block_words,
             self->counter, self->block_len, self->flags, out);
    for (int i = 8; i < 16; i++) {
        out[i - 8] = out[i];
    }
}

void output_root_bytes(Output *self, uchar *out, int out_len) {
    uint words[16];
    uint output_block_counter = 0;
    for (int i = 0; i < out_len; i += 2*OUT_LEN) {
        compress(self->input_chaining_value, self->block_words,
                 output_block_counter, self->block_len,
                 self->flags | ROOT, words);
        int words_to_copy = min(16, (out_len - i + 3) / 4);
        for (int j = 0; j < words_to_copy; j++) {
            int bytes_to_copy = min(4, out_len - i - 4*j);
            for (int k = 0; k < bytes_to_copy; k++) {
                out[i + 4*j + k] = (words[j] >> (8*k)) & 0xFF;
            }
        }
        output_block_counter++;
    }
}

typedef struct {
    uint chaining_value[8];
    ulong chunk_counter;
    uchar block[BLOCK_LEN];
    uint block_len;
    uint blocks_compressed;
    uint flags;
} ChunkState;

void chunk_state_init(ChunkState *self, const uint *key_words, ulong chunk_counter, uint flags) {
    for (int i = 0; i < 8; i++) {
        self->chaining_value[i] = key_words[i];
    }
    self->chunk_counter = chunk_counter;
    for (int i = 0; i < BLOCK_LEN; i++) {
        self->block[i] = 0;
    }
    self->block_len = 0;
    self->blocks_compressed = 0;
    self->flags = flags;
}

uint chunk_state_len(const ChunkState *self) {
    return (uint)BLOCK_LEN * self->blocks_compressed + self->block_len;
}

uint chunk_state_start_flag(const ChunkState *self) {
    return self->blocks_compressed == 0 ? CHUNK_START : 0;
}

void chunk_state_update(ChunkState *self, const uchar *input, uint input_len) {
    while (input_len > 0) {
        if (self->block_len == BLOCK_LEN) {
            uint block_words[16];
            words_from_little_endian_bytes(self->block, block_words, 16);
            uint compress_out[16];
            compress(self->chaining_value, block_words, self->chunk_counter,
                     BLOCK_LEN, self->flags | chunk_state_start_flag(self),
                     compress_out);
            for (int i = 0; i < 8; i++) {
                self->chaining_value[i] = compress_out[i];
            }
            self->blocks_compressed++;
            for (int i = 0; i < BLOCK_LEN; i++) {
                self->block[i] = 0;
            }
            self->block_len = 0;
        }

        uint want = BLOCK_LEN - self->block_len;
        uint take = min(want, input_len);
        for (uint i = 0; i < take; i++) {
            self->block[self->block_len + i] = input[i];
        }
        self->block_len += take;
        input += take;
        input_len -= take;
    }
}

void chunk_state_output(const ChunkState *self, Output *out) {
    uint block_words[16];
    words_from_little_endian_bytes(self->block, block_words, 16);
    for (int i = 0; i < 8; i++) {
        out->input_chaining_value[i] = self->chaining_value[i];
    }
    for (int i = 0; i < 16; i++) {
        out->block_words[i] = block_words[i];
    }
    out->counter = self->chunk_counter;
    out->block_len = self->block_len;
    out->flags = self->flags | chunk_state_start_flag(self) | CHUNK_END;
}

void parent_output(
    const uint *left_child_cv,
    const uint *right_child_cv,
    const uint *key_words,
    uint flags,
    Output *out
) {
    for (int i = 0; i < 8; i++) {
        out->input_chaining_value[i] = key_words[i];
        out->block_words[i] = left_child_cv[i];
        out->block_words[i + 8] = right_child_cv[i];
    }
    out->counter = 0;
    out->block_len = BLOCK_LEN;
    out->flags = PARENT | flags;
}

void parent_cv(
    const uint *left_child_cv,
    const uint *right_child_cv,
    const uint *key_words,
    uint flags,
    uint *out
) {
    Output parent_out;
    parent_output(left_child_cv, right_child_cv, key_words, flags, &parent_out);
    output_chaining_value(&parent_out, out);
}

typedef struct {
    ChunkState chunk_state;
    uint key_words[8];
    uint cv_stack[54 * 8];
    uint cv_stack_len;
    uint flags;
} Hasher;

void hasher_init(Hasher *self, const uint *key_words, uint flags) {
    chunk_state_init(&self->chunk_state, key_words, 0, flags);
    for (int i = 0; i < 8; i++) {
        self->key_words[i] = key_words[i];
    }
    for (int i = 0; i < 54 * 8; i++) {
        self->cv_stack[i] = 0;
    }
    self->cv_stack_len = 0;
    self->flags = flags;
}

void hasher_new_internal(Hasher *self, const uint *key_words, uint flags) {
    hasher_init(self, key_words, flags);
}

void hasher_new(Hasher *self) {
    uint iv_copy[8];
    for (int i = 0; i < 8; i++) {
        iv_copy[i] = IV[i];
    }
    hasher_new_internal(self, iv_copy, 0);
}

void hasher_new_keyed(Hasher *self, const uchar key[KEY_LEN]) {
    uint key_words[8];
    words_from_little_endian_bytes(key, key_words, 8);
    hasher_new_internal(self, key_words, KEYED_HASH);
}

void hasher_push_stack(Hasher *self, const uint *cv) {
    for (int i = 0; i < 8; i++) {
        self->cv_stack[self->cv_stack_len * 8 + i] = cv[i];
    }
    self->cv_stack_len++;
}

void hasher_pop_stack(Hasher *self, uint *out) {
    self->cv_stack_len--;
    for (int i = 0; i < 8; i++) {
        out[i] = self->cv_stack[self->cv_stack_len * 8 + i];
    }
}

void hasher_add_chunk_chaining_value(Hasher *self, const uint *new_cv, ulong total_chunks) {
    while ((total_chunks & 1) == 0) {
        uint parent_node[8];
        hasher_pop_stack(self, parent_node);
        parent_cv(parent_node, new_cv, self->key_words, self->flags, (uint *)new_cv);
        total_chunks >>= 1;
    }
    hasher_push_stack(self, new_cv);
}

void hasher_update(Hasher *self, const uchar *input, uint input_len) {
    while (input_len > 0) {
        if (chunk_state_len(&self->chunk_state) == CHUNK_LEN) {
            uint chunk_cv[8];
            Output output;
            chunk_state_output(&self->chunk_state, &output);
            output_chaining_value(&output, chunk_cv);
            ulong total_chunks = self->chunk_state.chunk_counter + 1;
            hasher_add_chunk_chaining_value(self, chunk_cv, total_chunks);
            chunk_state_init(&self->chunk_state, self->key_words, total_chunks, self->flags);
        }

        uint want = CHUNK_LEN - chunk_state_len(&self->chunk_state);
        uint take = min(want, input_len);
        chunk_state_update(&self->chunk_state, input, take);
        input += take;
        input_len -= take;
    }
}

void hasher_finalize(const Hasher *self, uchar *out, uint out_len) {
    Output output;
    chunk_state_output(&self->chunk_state, &output);
    
    uint parent_nodes_remaining = self->cv_stack_len;
    while (parent_nodes_remaining > 0) {
        parent_nodes_remaining--;
        uint parent_node[8];
        for (int j = 0; j < 8; j++) {
            parent_node[j] = self->cv_stack[parent_nodes_remaining * 8 + j];
        }
        uint chaining_value[8];
        output_chaining_value(&output, chaining_value);
        parent_output(parent_node, chaining_value, self->key_words, self->flags, &output);
    }
    output_root_bytes(&output, out, out_len);
}