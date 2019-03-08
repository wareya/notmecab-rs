use std::io::BufReader;
use std::io::BufRead;
use std::io::Read;

use std::collections::HashMap;
use std::collections::HashSet;

use crate::FormatToken;
use crate::strings::*;

#[derive(Debug)]
pub (crate) struct UserDict {
    pub(crate) dict: HashMap<String, Vec<FormatToken>>,
    pub(crate) contains_longer: HashSet<String>,
    pub(crate) features: Vec<String>,
}

impl UserDict {
    pub (crate) fn load<T : Read>(file : &mut BufReader<T>) -> Result<UserDict, &'static str>
    {
        let mut dict : HashMap<String, Vec<FormatToken>> = HashMap::new();
        let mut contains_longer = HashSet::new();
        let mut features = Vec::new();
        for (i, line) in file.lines().enumerate()
        {
            let line = line.or_else(|_| Err("IO error"))?;
            let parts : Vec<&str> = line.splitn(5, ",").into_iter().collect();
            if parts.len() != 5
            {
                continue;
            }
            let surface = parts[0].to_string();
            let surface_codepoints = codepoints(&surface);
            let left_context = parts[1].parse::<u16>().or_else(|_| Err("numeric parse error"))?;
            let right_context = parts[2].parse::<u16>().or_else(|_| Err("numeric parse error"))?;
            let cost = parts[3].parse::<i64>().or_else(|_| Err("numeric parse error"))?;
            let feature = parts[4].to_string();
            let token = FormatToken
            { left_context,
              right_context,
              pos : 0,
              cost,
              original_id : i as u32,
              feature_offset : i as u32
            };
            if let Some(list) = dict.get_mut(&surface)
            {
                list.push(token);
            }
            else
            {
                dict.insert(surface, vec!(token));
            }
            for i in 1..surface_codepoints.len()
            {
                let toinsert = surface_codepoints[0..i].iter().collect();
                contains_longer.insert(toinsert);
            }
            features.push(feature);
        }
        Ok(UserDict { dict, contains_longer, features })
    }
    
    pub (crate) fn may_contain(&self, find : &String) -> bool
    {
        self.contains_longer.contains(find) || self.dict.contains_key(find)
    }
    pub (crate) fn dic_get<'a>(&'a self, find : &String) -> Option<Vec<&'a FormatToken>>
    {
        self.dict.get(find).map(|l| l.iter().map(|x| x).collect())
    }
    pub (crate) fn feature_get(&self, offset : u32) -> String
    {
        self.features.get(offset as usize).cloned().unwrap_or_else(|| "".to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use super::*;
    
    #[test]
    fn test_unkchar_load()
    {
        let mut usrdic = BufReader::new(File::open("data/userdict.csv").unwrap());
        UserDict::load(&mut usrdic).unwrap();
    }
}

