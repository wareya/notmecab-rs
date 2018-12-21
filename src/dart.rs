use std::io::BufReader;
use std::collections::HashMap;

use std::io::Read;

use super::file::*;
use super::FormatToken;

#[derive(Debug)]
pub (crate) struct Link {
    base : u32,
    check : u32
}

impl Link {
    pub (crate) fn read<T : Read>(sysdic : &mut BufReader<T>) -> Result<Link, &'static str>
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

fn collect_links_hashmap(links : &[Link], base : u32, collection : &mut Vec<(String, u32)>, key : &[u8])
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
            collect_links_hashmap(links, links[(base+1+i) as usize].base, collection, &newkey);
        }
    }
}

// for supporting unk.dic in the future

/*
struct Range {
    start : char,
    end : char,
}
fn collect_links_ranges(links : &[Link], base : u32, ranges : &mut Vec<Range>, key : Vec<u8>, start : &mut char, state : &mut u32)
{
    if check_valid_out(links, base, base).is_ok()
    {
        if let Ok(key) = read_str_buffer(&key)
        {
            if key.chars().count() != 1:
            {
                panic("range dictionary contains non-single-character entries");
            }
            let c : char = key.chars().next();
            
            let newstate = !links[base as usize].base;
            if newstate != oldstate && start != 0
            {
                
            }
        }
    }
    for i in 0..0x100
    {
        if check_valid_link(links, base, base+1+i).is_ok()
        {
            let mut newkey = key.clone();
            newkey.push(i as u8);
            collect_links_ranges(links, links[(base+1+i) as usize].base, collection, newkey, start, state);
        }
    }
}
*/

fn entries_to_tokens(entries : Vec<(String, u32)>, tokens : &[FormatToken]) -> HashMap<String, Vec<FormatToken>>
{
    let mut dictionary : HashMap<String, Vec<FormatToken>> = HashMap::new();
    for entry in entries
    {
        let mut similar_lexemes : Vec<FormatToken> = Vec::new();
        
        let first : u32 = entry.1 / 0x100;
        let count : u32 = entry.1 % 0x100;
        for i in 0..count
        {
            similar_lexemes.push(tokens[(first+i) as usize].clone());
        }
        dictionary.insert(entry.0, similar_lexemes);
    }
    
    dictionary
}

pub (crate) fn collect_links_into_hashmap(links : &[Link], tokens : &[FormatToken]) -> HashMap<String, Vec<FormatToken>>
{
    let mut collection : Vec<(String, u32)> = Vec::new();
    collect_links_hashmap(&links, links[0].base, &mut collection, &[]);
    
    entries_to_tokens(collection, tokens)
}