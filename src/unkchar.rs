use std::collections::HashMap;

use std::io::BufReader;
use std::io::Read;
use std::io::Seek;

use super::file::*;

// for loading

#[derive(Clone)]
#[derive(Copy)]
struct CharData {
    // bitfield of compatible types
    typefield : u32,
    // type to treat sequences starting with this character as
    default_type : u8,
    // group up all possible groups up to this many compatible characters
    prefix_group_len : u8,
    // group as many compatible characters as possible
    greedy_group : bool,
    // force processing this character into groups even if there is a dictionary token that passes over this character
    always_process : bool,
}

impl CharData {
    fn read(data : u32) -> CharData
    {
        CharData {
            typefield : data & 0x0003_FFFF,
            default_type : ((data >> 18) & 0xFF) as u8,
            prefix_group_len : ((data >> 26) & 0xF) as u8,
            greedy_group : ((data >> 30) & 1) != 0,
            always_process : ((data >> 31) & 1) != 0,
        }
    }
}

// for usage

#[derive(Clone)]
#[derive(Copy)]
pub (crate) struct CharType {
    typefield : u32,
    default_type : u8,
}

impl CharType {
    fn from(data : CharData) -> CharType
    {
        CharType {
            typefield : data.typefield,
            default_type : data.default_type,
        }
    }
}

impl CharType {
    fn has_type(&self, index : u8) -> bool
    {
        if index >= 32
        {
            return false;
        }
        self.typefield & (1u32<<(index)) != 0
    }
}

#[derive(Clone)]
pub (crate) struct TypeData {
    pub (crate) name : String,
    pub (crate) number : u8,
    pub (crate) prefix_group_len : u8,
    pub (crate) greedy_group : bool,
    pub (crate) always_process : bool,
}

impl TypeData {
    fn from(data : CharData, names : &Vec<String>) -> Result<TypeData, &'static str>
    {
        if data.default_type as usize >= names.len()
        {
            return Err("invalid chars.bin file");
        }
        Ok(TypeData {
            name : names[data.default_type as usize].clone(),
            number : data.default_type,
            prefix_group_len : data.prefix_group_len,
            greedy_group : data.greedy_group,
            always_process : data.always_process,
        })
    }
}

pub (crate) struct UnkChar {
    types : HashMap<u8, TypeData>,
    data : Vec<CharType>
}

impl UnkChar {
    pub (crate) fn get_type<'a>(&'a self, c : char) -> &'a TypeData
    {
        if (c as u32) < 0xFFFF
        {
            self.types.get(&self.data[c as usize].default_type).unwrap()
        }
        else
        {
            self.types.get(&0).unwrap()
        }
    }
    pub (crate) fn has_type(&self, c : char, ctype : u8) -> bool
    {
        if (c as u32) < 0xFFFF
        {
            self.data[c as usize].has_type(ctype)
        }
        else
        {
            ctype == 0
        }
    }
    pub (crate) fn always_process(&self, c : char) -> bool
    {
        self.get_type(c).always_process
    }
}

pub (crate) fn load_char_bin<T : Read + Seek>(file : &mut BufReader<T>) -> Result<UnkChar, &'static str>
{
    let num_types = read_u32(file)?;
    let mut type_names = Vec::new();
    for _ in 0..num_types
    {
        type_names.push(read_nstr(file, 0x20)?);
    }
    let mut unk_chars = UnkChar {
        types : HashMap::new(),
        data : Vec::new()
    };
    for _ in 0..0xFFFF
    {
        let bitfield = read_u32(file)?;
        let data = CharData::read(bitfield);
        if !unk_chars.types.contains_key(&data.default_type)
        {
            unk_chars.types.insert(data.default_type, TypeData::from(data, &type_names)?);
        }
        unk_chars.data.push(CharType::from(data));
    }
    Ok(unk_chars)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use super::*;
    use crate::dart;
    
    #[test]
    fn test_unkchar_load()
    {
        let mut unkdic = BufReader::new(File::open("data/unk.dic").unwrap());
        let mut unkdef = BufReader::new(File::open("data/char.bin").unwrap());
        
        dart::load_mecab_dart_file(0xEF_71_9A_03, &mut unkdic).unwrap();
        load_char_bin(&mut unkdef).unwrap();
    }
}

