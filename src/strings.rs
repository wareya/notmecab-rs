use std::collections::VecDeque;

use std::fs::File;
use std::io::Read;

pub fn read_file_to_string(fname : &str) -> std::io::Result<String>
{
    let mut file = File::open("grammarsimple.txt")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    return Ok(contents);
}

pub fn codepoints_len(text : &str) -> usize
{
    text.chars().count()
}

pub fn codepoints(text : &str) -> Vec<char>
{
    text.chars().collect()
}

pub fn slice(text : &str, start : i64, end : i64) -> String
{
    let chars : Vec<char> = text.chars().collect();
    let u_start = if start < 0 {chars.len() - (-start as usize)} else {start as usize};
    let u_end   = if end   < 0 {chars.len() - (-end   as usize)} else {end   as usize};
    
    if u_start >= chars.len()
    {
        return "".to_string();
    }
    else
    {
        return chars[u_start..u_end].into_iter().collect();
    }
}

pub fn unescape(text: &str) -> String
{
    let mut ret = String::with_capacity(text.len());
    let mut chars : VecDeque<char> = text.chars().collect();
    while let Some(c) = chars.pop_front()
    {
        if c != '\\'
        {
            ret.push(c);
        }
        else
        {
            if let Some(c2) = chars.pop_front()
            {
                match c2
                {
                    '\\' => {ret.push(c);}
                    'n' => {ret.push('\n');}
                    'r' => {ret.push('\r');}
                    't' => {ret.push('\t');}
                    '"' => {ret.push('"');}
                    _ => {ret.push(c); ret.push(c2);}
                }
            }
        }
    }
    return ret;
}

pub fn escape(text: &str) -> String
{
    let mut ret = String::with_capacity(text.len());
    let mut chars : VecDeque<char> = text.chars().collect();
    while let Some(c) = chars.pop_front()
    {
        match c
        {
            '\\' => {ret.push('\\');ret.push('\\');}
            '\n' => {ret.push('\\');ret.push('n');}
            '\r' => {ret.push('\\');ret.push('r');}
            '\t' => {ret.push('\\');ret.push('t');}
            '\"' => {ret.push('\\');ret.push('"');}
            _ => {ret.push(c);}
        }
    }
    return ret;
}
