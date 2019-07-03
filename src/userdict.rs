use std::io::BufRead;
use std::io::Read;

use hashbrown::HashMap;
use hashbrown::HashSet;

use crate::FormatToken;

#[derive(Debug)]
pub (crate) struct UserDict {
    pub(crate) dict: HashMap<String, Vec<FormatToken>>,
    pub(crate) contains_longer: HashSet<String>,
    pub(crate) features: Vec<String>,
}

impl UserDict {
    pub (crate) fn load<T : Read + BufRead>(file : &mut T) -> Result<UserDict, &'static str>
    {
        let mut dict : HashMap<String, Vec<FormatToken>> = HashMap::new();
        let mut contains_longer = HashSet::new();
        let mut features = Vec::new();
        for (i, line) in file.lines().enumerate()
        {
            let line = line.or_else(|_| Err("IO error"))?;
            let parts : Vec<&str> = line.splitn(5, ',').collect();
            if parts.len() != 5
            {
                continue;
            }
            let surface = parts[0].to_string();
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
                dict.insert(surface.clone(), vec!(token));
            }
            for (i, _) in surface.char_indices()
            {
                if i > 0
                {
                    contains_longer.insert(surface[0..i].to_string());
                }
            }
            features.push(feature);
        }
        Ok(UserDict { dict, contains_longer, features })
    }
    
    pub (crate) fn may_contain(&self, find : &str) -> bool
    {
        self.contains_longer.contains(find) || self.dict.contains_key(find)
    }
    pub (crate) fn dic_get<'a>(&'a self, find : &str) -> Option<&'a Vec<FormatToken>>
    {
        self.dict.get(find)
    }
    pub (crate) fn feature_get(&self, offset : u32) -> String
    {
        self.features.get(offset as usize).cloned().unwrap_or_else(|| "".to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use super::*;
    
    #[test]
    fn test_unkchar_load()
    {
        let mut usrdic = BufReader::new(File::open("data/userdict.csv").unwrap());
        UserDict::load(&mut usrdic).unwrap();
    }
}

