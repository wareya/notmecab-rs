use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;

/// A blob of bytes.
pub struct Blob {
    _data: Box<dyn AsRef<[u8]> + Sync + Send>,
    pointer: *const u8,
    length: usize
}

unsafe impl Sync for Blob {}
unsafe impl Send for Blob {}

impl Deref for Blob {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        // Technically we could just dereference the data handle
        // and grab the reference from there, but that would be
        // potentially slower as the data is boxed.
        unsafe {
            std::slice::from_raw_parts(self.pointer, self.length)
        }
    }
}

impl AsRef<[u8]> for Blob {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.deref()
    }
}

impl Blob {
    /// Constructs a new `Blob`. The `data` object will be kept alive until
    /// the `Blob` is dropped.
    ///
    /// # Examples
    /// ```rust
    /// # use notmecab::Blob;
    /// static STATIC_SLICE: &[u8] = &[1, 2, 3];
    /// let blob_1 = Blob::new(STATIC_SLICE);
    /// assert_eq!(blob_1.len(), 3);
    /// assert_eq!(&*blob_1, &[1, 2, 3]);
    ///
    /// let vector: Vec<u8> = vec![4, 5];
    /// let blob_2 = Blob::new(vector);
    /// assert_eq!(blob_2.len(), 2);
    /// assert_eq!(&*blob_2, &[4, 5])
    /// ```
    pub fn new(data: impl AsRef<[u8]> + Sync + Send + 'static) -> Self {
        let data = Box::new(data);
        let slice: &[u8] = (*data).as_ref();
        let pointer = slice.as_ptr();
        let length = slice.len();
        Blob {
            _data: data,
            pointer,
            length
        }
    }

    /// Opens a file at a given path and creates a `Blob` from it. Will use `mmap`.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let fp = File::open(path)?;
        Self::from_file(&fp)
    }

    /// Creates a `Blob` from a `File`. Will use `mmap`.
    pub fn from_file(fp: &File) -> io::Result<Self> {
        let mmap = unsafe {
            memmap::Mmap::map(fp)?
        };

        Ok(Self::new(mmap))
    }
}
