use hashbrown::HashMap;
use hashbrown::HashSet;

use std::io::BufRead;
use std::io::Cursor;
use std::io::Read;
use std::io::Seek;

use std::cell::RefCell;

use super::blob::*;
use super::file::*;
use super::FormatToken;

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
    let mut dictionary : HashMap<String, DictInfo> = HashMap::new();
    for entry in entries
    {
        let first : u32 = entry.1 / 0x100;
        let end   : u32 = (entry.1 % 0x100) + first;
        dictionary.insert(entry.0, DictInfo{first, end});
    }
    
    dictionary
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
    pub(crate) contains_longer : HashSet<String>,
    pub(crate) left_contexts : u32,
    pub(crate) right_contexts : u32,
    feature_bytes_location : usize,
    feature_bytes_count : usize,
    blob : Blob,
    feature_string_cache : RefCell<HashMap<u32, String>>
}

impl DartDict {
    pub (crate) fn may_contain(&self, find : &str) -> bool
    {
        self.contains_longer.contains(find) || self.dict.contains_key(find)
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
    pub (crate) fn feature_get(&self, offset : u32) -> Result<String, &'static str>
    {
        if let Some(cached) = self.feature_string_cache.borrow().get(&offset)
        {
            return Ok(cached.clone());
        }
        if (offset as usize) < self.feature_bytes_count
        {
            let mut vec = Vec::new();
            let mut reader = Cursor::new(&self.blob);
            reader.seek(std::io::SeekFrom::Start(self.feature_bytes_location as u64 + offset as u64)).unwrap();
            reader.read_until(0, &mut vec).ok();
            let ret = read_str_buffer(&vec[..]);
            if ret.is_ok()
            {
                let ret = ret.unwrap();
                let mut cache = self.feature_string_cache.borrow_mut();
                cache.insert(offset, ret.clone());
                Ok(ret)
            }
            else
            {
                ret
            }
        }
        else
        {
            Ok("".to_string())
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
    let featurebytes = read_u32(dic_file)?; // number of bytes used to store the feature string pile
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
    
    let dictionary = collect_links_into_map(links);
    
    let mut contains_longer = HashSet::new();
    
    for entry in dictionary.keys()
    {
        if !contains_longer.contains(entry)
        {
            for (i, _) in entry.char_indices()
            {
                if i > 0
                {
                    contains_longer.insert(entry[0..i].to_string());
                }
            }
        }
    }
    
    Ok(DartDict{
        dict: dictionary,
        tokens,
        contains_longer,
        left_contexts,
        right_contexts,
        feature_bytes_location,
        feature_bytes_count : featurebytes as usize,
        blob,
        feature_string_cache : RefCell::new(HashMap::new())
    })
}