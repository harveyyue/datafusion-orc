// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Encoding/Decoding logic for writing/reading Run Length Encoded values
//! from ORC types.

use std::{
    fmt,
    io::Read,
    ops::{BitOrAssign, ShlAssign},
};

use bytes::Bytes;
use num::{traits::CheckedShl, PrimInt, Signed};
use snafu::ResultExt;

use crate::{
    column::Column,
    error::{InvalidColumnEncodingSnafu, IoSnafu, Result},
    memory::EstimateMemory,
    proto::column_encoding::Kind as ProtoColumnKind,
};

use self::{
    rle_v1::RleReaderV1,
    rle_v2::RleReaderV2,
    util::{
        get_closest_aligned_bit_width, signed_msb_decode, signed_zigzag_decode,
        signed_zigzag_encode,
    },
};

pub mod boolean;
pub mod byte;
pub mod decimal;
pub mod float;
pub mod rle_v1;
pub mod rle_v2;
pub mod timestamp;
mod util;

/// Encodes primitive values into an internal buffer, usually with a specialized run length
/// encoding for better compression.
pub trait PrimitiveValueEncoder<V>: EstimateMemory
where
    V: Copy,
{
    fn new() -> Self;

    fn write_one(&mut self, value: V);

    fn write_slice(&mut self, values: &[V]) {
        for &value in values {
            self.write_one(value);
        }
    }

    /// Take the encoded bytes, replacing it with an empty buffer.
    // TODO: Figure out how to retain the allocation instead of handing
    //       it off each time.
    fn take_inner(&mut self) -> Bytes;
}

pub trait PrimitiveValueDecoder<V>: Iterator<Item = Result<V>> {
    /// Decode out.len() values into out at a time, returning the amount of
    /// values decoded successfully.
    ///
    /// By default it relies on Iterator::next(), but hopefully this can be
    /// refactored away when it is properly implemented for all the decoders.
    // TODO: what about returning a &mut []? or taking &mut Vec<> as input?
    //       relying on return usize to indicate how many values in out are
    //       actually valid is probably not the best interface here.
    fn decode(&mut self, out: &mut [V]) -> Result<usize> {
        let mut len = 0;
        for n in out.iter_mut() {
            match self.next() {
                Some(r) => {
                    *n = r?;
                    len += 1;
                }
                None => break,
            };
        }
        Ok(len)
    }
}

pub fn get_unsigned_rle_reader<R: Read + Send + 'static>(
    column: &Column,
    reader: R,
) -> Box<dyn PrimitiveValueDecoder<i64> + Send> {
    match column.encoding().kind() {
        ProtoColumnKind::Direct | ProtoColumnKind::Dictionary => {
            Box::new(RleReaderV1::<i64, _, UnsignedEncoding>::new(reader))
        }
        ProtoColumnKind::DirectV2 | ProtoColumnKind::DictionaryV2 => {
            Box::new(RleReaderV2::<i64, _, UnsignedEncoding>::new(reader))
        }
    }
}

pub fn get_rle_reader<N: NInt, R: Read + Send + 'static>(
    column: &Column,
    reader: R,
) -> Result<Box<dyn PrimitiveValueDecoder<N> + Send>> {
    match column.encoding().kind() {
        ProtoColumnKind::Direct => Ok(Box::new(RleReaderV1::<N, _, SignedEncoding>::new(reader))),
        ProtoColumnKind::DirectV2 => Ok(Box::new(RleReaderV2::<N, _, SignedEncoding>::new(reader))),
        k => InvalidColumnEncodingSnafu {
            name: column.name(),
            encoding: k,
        }
        .fail(),
    }
}

pub trait EncodingSign: Send + 'static {
    // TODO: have separate type/trait to represent Zigzag encoded NInt?
    fn zigzag_decode<N: VarintSerde>(v: N) -> N;
    fn zigzag_encode<N: VarintSerde>(v: N) -> N;

    fn decode_signed_msb<N: NInt>(v: N, encoded_byte_size: usize) -> N;
}

pub struct SignedEncoding;

impl EncodingSign for SignedEncoding {
    #[inline]
    fn zigzag_decode<N: VarintSerde>(v: N) -> N {
        signed_zigzag_decode(v)
    }

    #[inline]
    fn zigzag_encode<N: VarintSerde>(v: N) -> N {
        signed_zigzag_encode(v)
    }

    #[inline]
    fn decode_signed_msb<N: NInt>(v: N, encoded_byte_size: usize) -> N {
        signed_msb_decode(v, encoded_byte_size)
    }
}

pub struct UnsignedEncoding;

impl EncodingSign for UnsignedEncoding {
    #[inline]
    fn zigzag_decode<N: VarintSerde>(v: N) -> N {
        v
    }

    #[inline]
    fn zigzag_encode<N: VarintSerde>(v: N) -> N {
        v
    }

    #[inline]
    fn decode_signed_msb<N: NInt>(v: N, _encoded_byte_size: usize) -> N {
        v
    }
}

pub trait VarintSerde: PrimInt + CheckedShl + BitOrAssign + Signed {
    const BYTE_SIZE: usize;

    /// Calculate the minimum bit size required to represent this value, by truncating
    /// the leading zeros.
    #[inline]
    fn bits_used(self) -> usize {
        Self::BYTE_SIZE * 8 - self.leading_zeros() as usize
    }

    /// Feeds [`Self::bits_used`] into a mapping to get an aligned bit width.
    fn closest_aligned_bit_width(self) -> usize {
        get_closest_aligned_bit_width(self.bits_used())
    }

    fn from_u8(b: u8) -> Self;
}

/// Helps generalise the decoder efforts to be specific to supported integers.
/// (Instead of decoding to u64/i64 for all then downcasting).
pub trait NInt:
    VarintSerde + ShlAssign<usize> + fmt::Debug + fmt::Display + fmt::Binary + Send + Sync + 'static
{
    type Bytes: AsRef<[u8]> + AsMut<[u8]> + Default + Clone + Copy + fmt::Debug;

    #[inline]
    fn empty_byte_array() -> Self::Bytes {
        Self::Bytes::default()
    }

    /// Should truncate any extra bits.
    fn from_i64(u: i64) -> Self;

    fn from_be_bytes(b: Self::Bytes) -> Self;

    // TODO: use num_traits::ToBytes instead
    fn to_be_bytes(self) -> Self::Bytes;

    fn add_i64(self, i: i64) -> Option<Self>;

    fn sub_i64(self, i: i64) -> Option<Self>;

    // TODO: use Into<i64> instead?
    fn as_i64(self) -> i64;

    fn read_big_endian(reader: &mut impl Read, byte_size: usize) -> Result<Self> {
        debug_assert!(
            byte_size <= Self::BYTE_SIZE,
            "byte_size cannot exceed max byte size of self"
        );
        let mut buffer = Self::empty_byte_array();
        // Read into back part of buffer since is big endian.
        // So if smaller than N::BYTE_SIZE bytes, most significant bytes will be 0.
        reader
            .read_exact(&mut buffer.as_mut()[Self::BYTE_SIZE - byte_size..])
            .context(IoSnafu)?;
        Ok(Self::from_be_bytes(buffer))
    }
}

impl VarintSerde for i16 {
    const BYTE_SIZE: usize = 2;

    #[inline]
    fn from_u8(b: u8) -> Self {
        b as Self
    }
}

impl VarintSerde for i32 {
    const BYTE_SIZE: usize = 4;

    #[inline]
    fn from_u8(b: u8) -> Self {
        b as Self
    }
}

impl VarintSerde for i64 {
    const BYTE_SIZE: usize = 8;

    #[inline]
    fn from_u8(b: u8) -> Self {
        b as Self
    }
}

impl VarintSerde for i128 {
    const BYTE_SIZE: usize = 16;

    #[inline]
    fn from_u8(b: u8) -> Self {
        b as Self
    }
}

// We only implement for i16, i32, i64 and u64.
// ORC supports only signed Short, Integer and Long types for its integer types,
// and i8 is encoded as bytes. u64 is used for other encodings such as Strings
// (to encode length, etc.).

impl NInt for i16 {
    type Bytes = [u8; 2];

    #[inline]
    fn from_i64(i: i64) -> Self {
        i as Self
    }

    #[inline]
    fn from_be_bytes(b: Self::Bytes) -> Self {
        Self::from_be_bytes(b)
    }

    #[inline]
    fn to_be_bytes(self) -> Self::Bytes {
        self.to_be_bytes()
    }

    #[inline]
    fn add_i64(self, i: i64) -> Option<Self> {
        i.try_into().ok().and_then(|i| self.checked_add(i))
    }

    #[inline]
    fn sub_i64(self, i: i64) -> Option<Self> {
        i.try_into().ok().and_then(|i| self.checked_sub(i))
    }

    #[inline]
    fn as_i64(self) -> i64 {
        self as i64
    }
}

impl NInt for i32 {
    type Bytes = [u8; 4];

    #[inline]
    fn from_i64(i: i64) -> Self {
        i as Self
    }

    #[inline]
    fn from_be_bytes(b: Self::Bytes) -> Self {
        Self::from_be_bytes(b)
    }

    #[inline]
    fn to_be_bytes(self) -> Self::Bytes {
        self.to_be_bytes()
    }

    #[inline]
    fn add_i64(self, i: i64) -> Option<Self> {
        i.try_into().ok().and_then(|i| self.checked_add(i))
    }

    #[inline]
    fn sub_i64(self, i: i64) -> Option<Self> {
        i.try_into().ok().and_then(|i| self.checked_sub(i))
    }

    #[inline]
    fn as_i64(self) -> i64 {
        self as i64
    }
}

impl NInt for i64 {
    type Bytes = [u8; 8];

    #[inline]
    fn from_i64(i: i64) -> Self {
        i as Self
    }

    #[inline]
    fn from_be_bytes(b: Self::Bytes) -> Self {
        Self::from_be_bytes(b)
    }

    #[inline]
    fn to_be_bytes(self) -> Self::Bytes {
        self.to_be_bytes()
    }

    #[inline]
    fn add_i64(self, i: i64) -> Option<Self> {
        self.checked_add(i)
    }

    #[inline]
    fn sub_i64(self, i: i64) -> Option<Self> {
        self.checked_sub(i)
    }

    #[inline]
    fn as_i64(self) -> i64 {
        self
    }
}
