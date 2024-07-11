use crate::Hash;
use blake3::Hasher as Blake3Hasher;

use crate::target::Uint256;

const BLOCK_HASH_DOMAIN: &[u8] = b"BlockHash";
const POW_HASH_DOMAIN: &[u8] = b"ProofOfWorkHash";
const HEAVY_HASH_DOMAIN: &[u8] = b"HeavyHash";

fn create_domain_key(domain: &[u8]) -> [u8; 32] {
    let mut key = [0u8; 32];
    let len = std::cmp::min(domain.len(), 32);
    key[..len].copy_from_slice(&domain[..len]);
    key
}

#[derive(Clone)]
pub struct PowHasher(Blake3Hasher);

impl PowHasher {
    #[inline(always)]
    pub fn new(pre_pow_hash: Hash, timestamp: u64) -> Self {
        let key = create_domain_key(POW_HASH_DOMAIN);
        let mut hasher = Blake3Hasher::new_keyed(&key);
        hasher.update(&pre_pow_hash.to_le_bytes());
        hasher.update(&timestamp.to_le_bytes());
        PowHasher(hasher)
    }

    #[inline(always)]
    pub fn finalize_with_nonce(& self, nonce: u64) -> Hash {
        let mut hasher = self.0.clone();
        hasher.update(&nonce.to_le_bytes());
        Hash::from_le_bytes(*(hasher.finalize().as_bytes()))
    }
}

#[derive(Clone, Copy)]
pub struct HeavyHasher;

impl HeavyHasher {
    pub fn hash(in_hash: Hash) -> Uint256 {
        let key = create_domain_key(HEAVY_HASH_DOMAIN);
        let mut hasher = Blake3Hasher::new_keyed(&key);
        hasher.update(&in_hash.to_le_bytes());
        Hash::from_le_bytes(*(hasher.finalize().as_bytes()))
    }
}

#[derive(Clone)]
pub struct HeaderHasher(Blake3Hasher);

impl HeaderHasher {
    pub fn new() -> Self {
        let key = create_domain_key(BLOCK_HASH_DOMAIN);
        HeaderHasher(Blake3Hasher::new_keyed(&key))
    }

    pub fn write<A: AsRef<[u8]>>(&mut self, data: A) {
        self.0.update(data.as_ref());
    }

    pub fn finalize(self) -> Uint256 {
        Hash::from_le_bytes(*(self.0.finalize().as_bytes()))
    }
}

pub trait Hasher {
    fn update<A: AsRef<[u8]>>(&mut self, data: A) -> &mut Self;
}

impl Hasher for HeaderHasher {
    fn update<A: AsRef<[u8]>>(&mut self, data: A) -> &mut Self {
        self.write(data);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_hash() {
        let timestamp: u64 = 5435345234;
        let nonce: u64 = 432432432;
        let pre_pow_hash = Hash::from_le_bytes([42; 32]);
        let hasher = PowHasher::new(pre_pow_hash, timestamp);
        let hash = hasher.finalize_with_nonce(nonce);
        
        // Add assertions here to verify the hash
    }

    #[test]
    fn test_heavy_hash() {
        let val = Hash::from_le_bytes([42; 32]);
        let hash = HeavyHasher::hash(val);
        
        // Add assertions here to verify the hash
    }

    #[test]
    fn test_header_hash() {
        let mut hasher = HeaderHasher::new();
        hasher.write(b"test data");
        let hash = hasher.finalize();
        
        // Add assertions here to verify the hash
    }
}