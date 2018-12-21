use std::io::BufReader;
use std::io::Read;

extern crate byteorder;
use byteorder::{LE, ReadBytesExt};

pub (crate) fn read_i16<T : Read>(f : &mut BufReader<T>) -> Result<i16, &'static str>
{
    match f.read_i16::<LE>()
    {
        Ok(val) => Ok(val),
        _ => Err("IO error")
    }
}
pub (crate) fn read_u16<T : Read>(f : &mut BufReader<T>) -> Result<u16, &'static str>
{
    match f.read_u16::<LE>()
    {
        Ok(val) => Ok(val),
        _ => Err("IO error")
    }
}
pub (crate) fn read_u32<T : Read>(f : &mut BufReader<T>) -> Result<u32, &'static str>
{
    match f.read_u32::<LE>()
    {
        Ok(val) => Ok(val),
        _ => Err("IO error")
    }
}

pub (crate) fn read_i16_buffer<T : Read>(f : &mut BufReader<T>, dst : &mut [i16]) -> Result<(), &'static str>
{
    match f.read_i16_into::<LE>(dst)
    {
        Ok(_) => Ok(()),
        _ => Err("IO error")
    }
}

fn trim_at_null(mystr : &[u8]) -> &[u8]
{
    let mut nullpos = 0usize;
    while nullpos < mystr.len() && mystr[nullpos] != 0
    {
        nullpos += 1
    }
    &mystr[..nullpos]
}

pub (crate) fn read_nstr<T : Read>(f : &mut BufReader<T>, n : usize) -> Result<String, &'static str>
{
    let mut buf = vec![0u8; n];
    
    match f.read_exact(&mut buf)
    {
        Ok(_) =>
        {
            let mystr = std::str::from_utf8(trim_at_null(&buf));
            
            if let Ok(mystr) = mystr
            {
                Ok(mystr.to_string())
            }
            else
            {
                Err("Decoding error")
            }
        }
        _ => Err("IO error")
    }
}
pub (crate) fn read_str_buffer(buf : &[u8]) -> Result<String, &'static str>
{
    let mystr = std::str::from_utf8(trim_at_null(buf));
    
    if let Ok(mystr) = mystr
    {
        Ok(mystr.to_string())
    }
    else
    {
        Err("UTF-8 decoding error")
    }
}

// this is way, WAY faster than seeking 4 bytes forward explicitly.
pub (crate) fn seek_rel_4<T : Read>(f : &mut BufReader<T>) -> Result<(), &'static str>
{
    //read_u32(f)?;
    let mut bogus = [0u8; 4];
    match f.read_exact(&mut bogus)
    {
        Ok(_) => Ok(()),
        _ => Err("IO error")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn null_padded_string_decode()
    {
        let vec = vec![0x20u8, 0x00u8, 0x00u8];
        assert_eq!(super::read_str_buffer(&vec), Ok(" ".to_string()));
    }
    #[test]
    fn null_comma_strings_decode_first_only()
    {
        let vec = vec![0x20u8, 0x00u8, 0x20u8];
        assert_eq!(super::read_str_buffer(&vec), Ok(" ".to_string()));
    }
}

