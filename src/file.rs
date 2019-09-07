use std::io::Read;

pub (crate) fn read_i16<T : Read>(f : &mut T) -> Result<i16, &'static str>
{
    read_u16(f).map(|val| val as i16)
}
pub (crate) fn read_u16<T : Read>(f : &mut T) -> Result<u16, &'static str>
{
    let mut buffer = [0; 2];
    match f.read_exact(&mut buffer)
    {
        Ok(()) => Ok(u16::from_le_bytes(buffer)),
        _ => Err("IO error")
    }
}
pub (crate) fn read_u32<T : Read>(f : &mut T) -> Result<u32, &'static str>
{
    let mut buffer = [0; 4];
    match f.read_exact(&mut buffer)
    {
        Ok(()) => Ok(u32::from_le_bytes(buffer)),
        _ => Err("IO error")
    }
}

unsafe fn as_byte_slice_mut<T>(slice : &mut [T]) -> &mut [u8]
{
    std::slice::from_raw_parts_mut(
        slice.as_mut_ptr() as *mut u8,
        slice.len() * std::mem::size_of::<T>()
    )
}

pub (crate) fn read_i16_buffer<T : Read>(f : &mut T, dst : &mut [i16]) -> Result<(), &'static str>
{
    let dst_b = unsafe { as_byte_slice_mut(dst) };
    f.read_exact(dst_b).map_err(|_| "IO error")?;

    for val in dst.iter_mut()
    {
        *val = i16::from_le(*val);
    }

    Ok(())
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

pub (crate) fn read_nstr<T : Read>(f : &mut T, n : usize) -> Result<String, &'static str>
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
pub (crate) fn seek_rel_4<T : Read>(f : &mut T) -> Result<(), &'static str>
{
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
    #[test]
    fn read_i16_buffer()
    {
        let input = &[0x12, 0x34, 0x56, 0x78];
        let mut out = [0i16, 2];
        super::read_i16_buffer(&mut &input[..], &mut out).unwrap();
        assert_eq!(out, [0x3412, 0x7856]);
    }
}

