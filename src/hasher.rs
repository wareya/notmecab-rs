pub struct Hasher(u64);

// This is basically FNV, just extended to do 4 bytes at a time.
impl Hasher {
    pub fn new() -> Self
    {
        Hasher(0xcbf29ce484222325)
    }

    pub fn write_u32(&mut self, value : u32)
    {
        let value = value as u64;
        self.0 = (self.0 ^ value).wrapping_mul(0x100000001b3);
    }

    pub fn finish(&self) -> u64
    {
        self.0
    }
}
