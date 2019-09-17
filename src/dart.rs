use crate::HashMap;
use crate::HashSet;

use std::io::Cursor;
use std::io::Read;
use std::io::Seek;
use std::ops::Range;
use std::hash::{BuildHasherDefault, Hasher};

use super::blob::*;
use super::file::*;
use super::FormatToken;

type BuildNoopHasher = BuildHasherDefault<NoopHasher>;

#[derive(Default)]
struct NoopHasher(u64);

impl Hasher for NoopHasher {
    fn finish(&self) -> u64
    {
        self.0
    }

    fn write(&mut self, bytes : &[u8])
    {
        for &byte in bytes
        {
            self.0 = (self.0 << 8) ^ (byte as u64);
        }
    }

    fn write_u64(&mut self, value : u64)
    {
        self.0 ^= value;
    }
}

#[derive(Debug)]
pub (crate) struct Link {
    base : u32,
    check : u32
}

impl Link {
    pub (crate) fn read<T : Read>(sysdic : &mut T) -> Result<Link, &'static str>
    {
        Ok(Link{base : read_u32(sysdic)?, check : read_u32(sysdic)?})
    }
}

fn check_valid_link(links : &[Link], from : u32, to : u32) -> Result<(), i32>
{
    // check for overflow
    if to as usize >= links.len()
    {
        return Err(1);
    }
    // make sure we didn't follow a link from somewhere we weren't supposed to
    else if links[to as usize].check != from
    {
        return Err(2);
    }
    // make sure we don't follow a link back where we started
    else if links[to as usize].base == from
    {
        return Err(3);
    }
    Ok(())
}

fn check_valid_out(links : &[Link], from : u32, to : u32) -> Result<(), i32>
{
    if let Err(err) = check_valid_link(links, from, to)
    {
        return Err(err);
    }
    // don't follow links to bases that aren't outputs
    else if links[to as usize].base < 0x8000_0000
    {
        return Err(-1);
    }
    Ok(())
}

fn collect_links(links : &[Link], base : u32, collection : &mut Vec<(String, u32)>, key : &[u8])
{
    if check_valid_out(links, base, base).is_ok()
    {
        if let Ok(key) = read_str_buffer(&key)
        {
            collection.push((key, !links[base as usize].base));
        }
    }
    for i in 0..0x100
    {
        if check_valid_link(links, base, base+1+i).is_ok()
        {
            let mut newkey = key.to_owned();
            newkey.push(i as u8);
            collect_links(links, links[(base+1+i) as usize].base, collection, &newkey);
        }
    }
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
pub(crate) struct DictInfo {
    first : u32,
    end   : u32,
}

fn entries_to_tokens(entries : Vec<(String, u32)>) -> HashMap<String, DictInfo>
{
    entries.into_iter().map(|entry| {
        let first : u32 = entry.1 / 0x100;
        let end   : u32 = (entry.1 % 0x100) + first;
        (entry.0, DictInfo{first, end})
    }).collect()
}

fn collect_links_into_map(links : Vec<Link>) -> HashMap<String, DictInfo>
{
    let mut collection : Vec<(String, u32)> = Vec::new();
    collect_links(&links, links[0].base, &mut collection, &[]);
    entries_to_tokens(collection)
}

pub (crate) struct DartDict {
    pub(crate) dict : HashMap<String, DictInfo>,
    pub(crate) tokens : Vec<FormatToken>,
    contains_longer : HashSet<u64, BuildNoopHasher>,
    pub(crate) left_contexts : u32,
    pub(crate) right_contexts : u32,
    feature_bytes_range : Range<usize>,
    blob : Blob
}

impl DartDict {
    pub (crate) fn may_contain(&self, hash : u64) -> bool
    {
        self.contains_longer.contains(&hash)
    }
    pub (crate) fn dic_get<'a>(&'a self, find : &str) -> Option<&'a [FormatToken]>
    {
        if let Some(info) = self.dict.get(find)
        {
            Some(&self.tokens[info.first as usize..info.end as usize])
        }
        else
        {
            None
        }
    }
    pub (crate) fn feature_get(&self, offset : u32) -> &str
    {
        let offset = offset as usize;
        let feature_blob = &self.blob[self.feature_bytes_range.clone()];
        let slice = match feature_blob.get(offset..) {
            Some(slice) => slice,
            None => {
                // Out-of-range offset.
                return "";
            }
        };
        
        let length = slice.iter().copied().take_while(|&byte| byte != 0).count();
        let slice = &slice[..length];
        
        let is_at_char_boundary =
            slice.is_empty() || (slice[0] as i8) >= -0x40;
        
        assert!(is_at_char_boundary);
        
        // This is safe since we checked that the whole feature blob is valid
        // UTF-8 when we loaded the dictionary.
        unsafe {
            std::str::from_utf8_unchecked(slice)
        }
    }
}

pub (crate) fn load_mecab_dart_file(blob : Blob) -> Result<DartDict, &'static str> {
    let mut reader = Cursor::new(&blob);
    let dic_file = &mut reader;
    // magic
    seek_rel_4(dic_file)?;
    
    // 0x04
    let version = read_u32(dic_file)?;
    if version != 0x66
    {
        return Err("unsupported version");
    }
    
    // 0x08
    seek_rel_4(dic_file)?; // dict type - u32 sys (0), usr (1), unk (2) - we don't care and have no use for the information
    
    read_u32(dic_file)?; // number of unique somethings; might be unique lexeme surfaces, might be feature strings, we don't need it
    // 0x10
    // this information is duplicated in the matrix dic_file and we will ensure that it is consistent
    let left_contexts  = read_u32(dic_file)?;
    let right_contexts = read_u32(dic_file)?;
    
    // 0x18
    let linkbytes = read_u32(dic_file)?; // number of bytes used to store the dual-array trie
    if linkbytes%8 != 0
    {
        return Err("dictionary broken: link table stored with number of bytes that is not a multiple of 8");
    }
    let tokenbytes = read_u32(dic_file)?; // number of bytes used to store the list of tokens
    if tokenbytes%16 != 0
    {
        return Err("dictionary broken: token table stored with number of bytes that is not a multiple of 16");
    }
    // 0x20
    let feature_bytes_count = read_u32(dic_file)? as usize; // number of bytes used to store the feature string pile
    seek_rel_4(dic_file)?;
    
    let encoding = read_nstr(dic_file, 0x20)?;
    if encoding.to_lowercase() != "utf-8"
    {
        return Err("only UTF-8 dictionaries are supported. stop using legacy encodings for infrastructure!");
    }
    
    let mut links : Vec<Link> = Vec::with_capacity((linkbytes/8) as usize);
    for _i in 0..(linkbytes/8)
    {
        links.push(Link::read(dic_file)?);
    }
    
    let mut tokens : Vec<FormatToken> = Vec::with_capacity((tokenbytes/16) as usize);
    for _i in 0..(tokenbytes/16)
    {
        tokens.push(FormatToken::read(dic_file, tokens.len() as u32)?);
    }
    
    let feature_bytes_location = dic_file.seek(std::io::SeekFrom::Current(0)).unwrap() as usize;
    let feature_bytes_range = feature_bytes_location..feature_bytes_location + feature_bytes_count;
    let feature_slice = match blob.get(feature_bytes_range.clone()) {
        Some(slice) => slice,
        None => {
            return Err("dictionary broken: invalid feature bytes range");
        }
    };
    if std::str::from_utf8(feature_slice).is_err() {
        return Err("dictionary broken: feature blob is not valid UTF-8");
    }
    
    let dictionary = collect_links_into_map(links);
    
    let mut contains_longer = HashSet::with_hasher(BuildNoopHasher::default());
    for entry in dictionary.keys()
    {
        let mut hasher = crate::hasher::Hasher::new();
        for ch in entry.chars()
        {
            hasher.write_u32(ch as u32);
            contains_longer.insert(hasher.finish());
        }
    }
    
    Ok(DartDict {
        dict: dictionary,
        tokens,
        contains_longer,
        left_contexts,
        right_contexts,
        feature_bytes_range,
        blob
    })
}