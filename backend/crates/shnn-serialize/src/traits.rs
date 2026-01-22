//! Serialization traits for zero-copy operations
//! 
//! Provides core traits for serializing and deserializing neuromorphic data
//! with zero-copy optimizations and deterministic memory layout.

use crate::{Result, SerializeError, Buffer, BufferMut, BinaryEncoder, BinaryDecoder};

/// Trait for types that can be serialized
pub trait Serialize {
    fn serialize(&self, encoder: &mut BinaryEncoder) -> Result<()>;
    
    fn serialized_size(&self) -> usize;
}

/// Trait for types that can be deserialized
pub trait Deserialize: Sized {
    fn deserialize(decoder: &mut BinaryDecoder) -> Result<Self>;
}

/// Zero-copy serialization trait for types with fixed layout
pub trait ZeroCopySerialize: Copy {
    /// Serialize with zero-copy by casting to bytes
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const u8,
                core::mem::size_of::<Self>(),
            )
        }
    }
    
    /// Deserialize with zero-copy by casting from bytes
    fn from_bytes(bytes: &[u8]) -> Result<&Self> {
        if bytes.len() < core::mem::size_of::<Self>() {
            return Err(SerializeError::UnexpectedEof);
        }
        
        let ptr = bytes.as_ptr() as *const Self;
        if !crate::utils::is_aligned(ptr as *const u8, core::mem::align_of::<Self>()) {
            return Err(SerializeError::AlignmentError);
        }
        
        Ok(unsafe { &*ptr })
    }
}

// Implement basic serialization for primitive types
impl Serialize for u8 {
    fn serialize(&self, encoder: &mut BinaryEncoder) -> Result<()> {
        encoder.buffer.write_u8(*self)
    }
    
    fn serialized_size(&self) -> usize {
        1
    }
}

impl Deserialize for u8 {
    fn deserialize(decoder: &mut BinaryDecoder) -> Result<Self> {
        decoder.buffer.read_u8()
    }
}

impl Serialize for u32 {
    fn serialize(&self, encoder: &mut BinaryEncoder) -> Result<()> {
        encoder.buffer.write_u32(*self)
    }

    fn serialized_size(&self) -> usize {
        4
    }
}

impl Deserialize for u32 {
    fn deserialize(decoder: &mut BinaryDecoder) -> Result<Self> {
        decoder.buffer.read_u32()
    }
}

impl Serialize for u64 {
    fn serialize(&self, encoder: &mut BinaryEncoder) -> Result<()> {
        encoder.buffer.write_u64(*self)
    }

    fn serialized_size(&self) -> usize {
        8
    }
}

impl Deserialize for u64 {
    fn deserialize(decoder: &mut BinaryDecoder) -> Result<Self> {
        decoder.buffer.read_u64()
    }
}

impl Serialize for f32 {
    fn serialize(&self, encoder: &mut BinaryEncoder) -> Result<()> {
        encoder.buffer.write_f32(*self)
    }

    fn serialized_size(&self) -> usize {
        4
    }
}

impl Deserialize for f32 {
    fn deserialize(decoder: &mut BinaryDecoder) -> Result<Self> {
        decoder.buffer.read_f32()
    }
}

#[cfg(feature = "std")]
impl Serialize for String {
    fn serialize(&self, encoder: &mut BinaryEncoder) -> Result<()> {
        let bytes = self.as_bytes();
        encoder.buffer.write_u32(bytes.len() as u32)?;
        encoder.buffer.write_bytes(bytes)
    }

    fn serialized_size(&self) -> usize {
        4 + self.len()
    }
}

#[cfg(feature = "std")]
impl Deserialize for String {
    fn deserialize(decoder: &mut BinaryDecoder) -> Result<Self> {
        let len = decoder.buffer.read_u32()? as usize;
        let bytes = decoder.buffer.read_bytes(len)?;
        core::str::from_utf8(bytes).map(|s| s.to_string()).map_err(|_| SerializeError::InvalidUtf8)
    }
}

impl Serialize for bool {
    fn serialize(&self, encoder: &mut BinaryEncoder) -> Result<()> {
        encoder.buffer.write_u8(*self as u8)
    }

    fn serialized_size(&self) -> usize {
        1
    }
}

impl Deserialize for bool {
    fn deserialize(decoder: &mut BinaryDecoder) -> Result<Self> {
        Ok(decoder.buffer.read_u8()? != 0)
    }
}

#[cfg(feature = "std")]
impl<T: Serialize> Serialize for Vec<T> {
    fn serialize(&self, encoder: &mut BinaryEncoder) -> Result<()> {
        encoder.buffer.write_u32(self.len() as u32)?;
        for item in self.iter() {
            item.serialize(encoder)?;
        }
        Ok(())
    }

    fn serialized_size(&self) -> usize {
        4 + self.iter().map(|item| item.serialized_size()).sum::<usize>()
    }
}

#[cfg(feature = "std")]
impl<T: Deserialize> Deserialize for Vec<T> {
    fn deserialize(decoder: &mut BinaryDecoder) -> Result<Self> {
        let len = decoder.buffer.read_u32()? as usize;
        let mut vec = Vec::new();
        for _ in 0..len {
            vec.push(T::deserialize(decoder)?);
        }
        Ok(vec)
    }
}

// Implement zero-copy for POD types
impl ZeroCopySerialize for u8 {}
impl ZeroCopySerialize for u16 {}
impl ZeroCopySerialize for u32 {}
impl ZeroCopySerialize for u64 {}
impl ZeroCopySerialize for f32 {}
impl ZeroCopySerialize for f64 {}