use std::collections::HashMap;

use std::io::BufReader;
use std::io::Read;
use std::io::Seek;

use super::file::*;

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
    fn has_type(&self, index : u32) -> bool
    {
        self.typefield & (1u32<<(index)) != 0
    }
}

pub (crate) struct TypeData {
    name : String,
    number : u8,
    word_length : u8,
    group_up : bool,
    always_process : bool,
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
            word_length : data.word_length,
            group_up : data.group_up,
            always_process : data.always_process,
        })
    }
}

#[derive(Clone)]
#[derive(Copy)]
struct CharData {
    typefield : u32,
    default_type : u8,
    word_length : u8,
    group_up : bool,
    always_process : bool,
}

impl CharData {
    fn read(data : u32) -> CharData
    {
        CharData {
            typefield : data & 0x0003_FFFF,
            default_type : ((data >> 18) & 0xFF) as u8,
            word_length : ((data >> 26) & 0xF) as u8,
            group_up : ((data >> 30) & 1) != 0,
            always_process : ((data >> 31) & 1) != 0,
        }
    }
}

pub (crate) struct UnkChar {
    types : HashMap<u8, TypeData>,
    data : Vec<CharType>
}

pub (crate) fn load_char_bin<T : Read + Seek>(file : &mut BufReader<T>) -> Result<UnkChar, &'static str>
{
    let num_types = read_u32(file)?;
    let mut type_names = Vec::new();
    for i in 0..num_types
    {
        type_names.push(read_nstr(file, 0x20)?);
    }
    let mut unk_chars = UnkChar {
        types : HashMap::new(),
        data : Vec::new()
    };
    for i in 0..0xFFFF
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
        
        let unk_dictionary = dart::load_mecab_dart_file(0xEF_71_9A_03, &mut unkdic).unwrap();
        load_char_bin(&mut unkdef).unwrap();
    }
}

