use hashbrown::HashMap;

use std::io::Read;

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
    fn has_type(self, index : u8) -> bool
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
    fn from(data : CharData, names : &[String]) -> Result<TypeData, &'static str>
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
    pub (crate) fn get_type(&'_ self, c : char) -> &'_ TypeData
    {
        if (c as u32) < 0xFFFF
        {
            &self.types[&self.data[c as usize].default_type]
        }
        else
        {
            &self.types[&0]
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

pub (crate) fn load_char_bin<T : Read>(file : &mut T) -> Result<UnkChar, &'static str>
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
        unk_chars.types.entry(data.default_type).or_insert(TypeData::from(data, &type_names)?);
        unk_chars.data.push(CharType::from(data));
    }
    Ok(unk_chars)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;
    use crate::dart;
    use crate::blob::Blob;
    
    #[test]
    fn test_unkchar_load()
    {
        let unkdic = Blob::open("data/unk.dic").unwrap();
        let unkdef = Blob::open("data/char.bin").unwrap();
        
        dart::load_mecab_dart_file(unkdic).unwrap();
        load_char_bin(&mut Cursor::new(unkdef)).unwrap();
    }
}

