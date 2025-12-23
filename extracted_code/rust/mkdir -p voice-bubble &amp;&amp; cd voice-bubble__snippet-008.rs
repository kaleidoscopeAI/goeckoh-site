use bumpalo::Bump;

struct AudioPool {
    arena: Bump,
    frames: Vec<AudioFrame>,
}

impl AudioPool {
    fn new() -> Self {
        Self {
            arena: Bump::new(),
            frames: Vec::with_capacity(100),
        }
    }
    
    fn allocate_frame(&self, samples: usize) -> AudioFrame {
        let ptr = self.arena.alloc_slice_fill_default(samples);
        AudioFrame::from_raw(ptr)
    }
}
