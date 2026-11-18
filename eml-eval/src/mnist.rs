//! The MNIST digits, read from the original gzip-compressed IDX files.

use std::{
    fs::File,
    io::{self, Read},
    path::Path,
    slice::ChunksExact,
};

use flate2::read::GzDecoder;

/// Images with raw pixel values 0–255, row by row, and their labels.
pub struct Digits {
    pub pixels: Vec<f64>,
    pub labels: Vec<u8>,
    pub image_size: usize,
}

impl Digits {
    /// The 10,000 test digits in `dir` (`t10k-images-idx3-ubyte.gz`,
    /// `t10k-labels-idx1-ubyte.gz`).
    pub fn test_set(dir: &Path) -> io::Result<Self> {
        let (dims, pixels) = read_idx(&dir.join("t10k-images-idx3-ubyte.gz"))?;
        let (_, labels) = read_idx(&dir.join("t10k-labels-idx1-ubyte.gz"))?;
        let image_size = dims[1..].iter().product();
        if dims[0] != labels.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "images and labels differ in number",
            ));
        }
        Ok(Digits {
            pixels: pixels.into_iter().map(f64::from).collect(),
            labels,
            image_size,
        })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn images(&self) -> ChunksExact<'_, f64> {
        self.pixels.chunks_exact(self.image_size)
    }
}

/// Reads an IDX file of unsigned bytes: its dimensions and its data.
fn read_idx(path: &Path) -> io::Result<(Vec<usize>, Vec<u8>)> {
    let mut bytes = Vec::new();
    GzDecoder::new(File::open(path)?).read_to_end(&mut bytes)?;
    let invalid = |what: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {what}", path.display()),
        )
    };

    let [0, 0, 0x08, ndim, rest @ ..] = bytes.as_slice() else {
        return Err(invalid("not an unsigned-byte IDX file"));
    };
    let (header, data) = rest
        .split_at_checked(4 * *ndim as usize)
        .ok_or_else(|| invalid("truncated header"))?;
    let dims: Vec<usize> = header
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_be_bytes(b) as usize)
        .collect();
    if dims.is_empty() || data.len() != dims.iter().product::<usize>() {
        return Err(invalid("size does not match the header"));
    }
    Ok((dims, data.to_vec()))
}
