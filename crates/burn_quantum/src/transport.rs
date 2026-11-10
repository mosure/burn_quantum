//! Shared bounded frame transport with absolute I/O deadlines.
use crate::{Error, Result};
use serde::{Serialize, de::DeserializeOwned};
use std::io::{Read, Write};
fn err(e: impl std::fmt::Display) -> Error {
    Error::InvalidValue(format!("framed transport: {e}"))
}
pub(crate) fn read<T: DeserializeOwned>(stream: &mut impl Read, limit: usize) -> Result<T> {
    let mut header = [0u8; 8];
    stream.read_exact(&mut header).map_err(err)?;
    let n = usize::try_from(u64::from_be_bytes(header)).map_err(|_| Error::Overflow)?;
    if n == 0 || n > limit {
        return Err(Error::InvalidValue("frame exceeds byte budget".into()));
    }
    let mut bytes = vec![0; n];
    stream.read_exact(&mut bytes).map_err(err)?;
    serde_json::from_slice(&bytes).map_err(err)
}
pub(crate) fn write<T: Serialize>(stream: &mut impl Write, value: &T, limit: usize) -> Result<()> {
    // The writer rejects growth beyond the encoded byte budget during serialization.
    struct Bounded {
        data: Vec<u8>,
        limit: usize,
    }
    impl Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.data.len()) {
                return Err(std::io::Error::other("encoded frame budget"));
            }
            self.data.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut buffer = Bounded {
        data: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut buffer, value).map_err(err)?;
    stream
        .write_all(&(buffer.data.len() as u64).to_be_bytes())
        .map_err(err)?;
    stream.write_all(&buffer.data).map_err(err)?;
    stream.flush().map_err(err)
}

use std::net::TcpStream;
use std::time::{Duration, Instant};
fn io(e: impl std::fmt::Display) -> Error {
    err(e)
}
pub(crate) struct DeadlineStream {
    stream: TcpStream,
    end: Instant,
}
impl DeadlineStream {
    pub(crate) fn new(stream: TcpStream, end: Instant) -> Result<Self> {
        stream.set_nodelay(true).map_err(io)?;
        Ok(Self { stream, end })
    }
    fn remaining(&self) -> std::io::Result<Duration> {
        self.end
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "connection deadline"))
    }
}
impl Read for DeadlineStream {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.stream.set_read_timeout(Some(self.remaining()?))?;
        self.stream.read(bytes)
    }
}
impl Write for DeadlineStream {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.stream.set_write_timeout(Some(self.remaining()?))?;
        self.stream.write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}
