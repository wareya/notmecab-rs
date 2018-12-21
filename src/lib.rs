use std::io::BufReader;
use std::io::Read;
use std::io::Seek;

use std::collections::HashSet;
use std::collections::HashMap;

use std::str;

extern crate pathfinding;

use pathfinding::directed::astar::astar;

mod strings;
mod file;
mod dart;

use self::file::*;
use self::dart::Link;
use self::strings::*;

#[derive(Clone)]
#[derive(Debug)]
pub (crate) struct FormatToken {
    left_context : u16,
    right_context : u16,
    
    pos  : u16,
    cost : i16,
    
    original_id : u32,
    
    feature_offset : u32,
}

impl FormatToken {
    fn read<T : Read + std::io::Seek>(sysdic : &mut BufReader<T>, original_id : u32) -> Result<FormatToken, &'static str>
    {
        let ret = FormatToken
        { left_context : read_u16(sysdic)?,
          right_context : read_u16(sysdic)?,
          pos : read_u16(sysdic)?,
          cost : read_i16(sysdic)?,
          original_id,
          feature_offset : read_u32(sysdic)?,
        };
        
        // seek away a u32 of padding
        seek_rel_4(sysdic)?;
        //seek_rel(sysdic, 4);
        
        Ok(ret)
    }
}

#[derive(Clone)]
#[derive(Debug)]
#[derive(PartialEq)]
enum TokenType {
    Normal,
    BOS,
    UNK
}

#[derive(Clone)]
#[derive(Debug)]
pub struct LexerToken {
    left_context : u16,
    right_context : u16,
    
    pos  : u16,
    cost : i16,
    
    original_id : u32,
    
    feature_offset : u32,
    
    start : usize, // in codepoints
    end   : usize, // in codepoints
    
    kind : TokenType,
}

impl LexerToken {
    fn from(other : & FormatToken, start : usize, end : usize) -> LexerToken
    {
        LexerToken
        { left_context : other.left_context,
          right_context : other.right_context,
          pos : other.pos,
          cost : other.cost,
          original_id : other.original_id,
          feature_offset : other.feature_offset,
          start,
          end,
          kind : TokenType::Normal
        }
    }
    fn make_unk(start : usize, end : usize) -> LexerToken
    {
        // FIXME: read unk.dic and use real context IDs
        LexerToken
        { left_context : !0u16,
          right_context : !0u16,
          pos : 0,
          cost : 0,
          original_id : !0u32,
          feature_offset : !0u32,
          start,
          end,
          kind : TokenType::UNK
        }
    }
    fn make_bos(start : usize, end : usize) -> LexerToken
    {
        LexerToken
        { left_context : 0, // context ID of EOS/BOS
          right_context : 0,
          pos : 0,
          cost : 0,
          original_id : !0u32,
          feature_offset : !0u32,
          start,
          end,
          kind : TokenType::BOS
        }
    }
}

#[derive(Debug)]
pub struct ParserToken {
    surface : String,
    feature : String,
    
    original_id : u32,
    
    unknown : bool,
}

impl ParserToken {
    fn build(surface : String, feature : String, original_id : u32, unknown : bool) -> ParserToken
    {
        ParserToken
        { surface,
          feature,
          original_id,
          unknown
        }
    }
}

pub struct Dict {
    dictionary : HashMap<String, Vec<FormatToken>>,
    contains_longer : HashSet<String>,
    // when using parse_to_lexertokens you need to be able to access the feature string table directly, so it's public
    pub feature_string_bytes : Vec<u8>,
    
    min_edge_cost_ever : i64,

    left_edges : u16,
    right_edges : u16,
    connection_matrix : Vec<i16>, // 2d matrix encoded as 1d
}

impl Dict {
    pub fn load<T : Read + Seek, Y : Read + Seek>(sysdic : &mut BufReader<T>, matrix : &mut BufReader<Y>) -> Result<Dict, &'static str>
    {
        // magic
        let magic = read_u32(sysdic)?;
        if magic != 0xE1172181
        {
            return Err("not a mecab sys.dic file");
        }
        
        // 0x04
        let version = read_u32(sysdic)?;
        if version != 0x66
        {
            return Err("unsupported version");
        }
    
        // 0x08
        seek_rel_4(sysdic)?; // dict type - u32 sys (0), usr (1), unk (2) - we don't care and have no use for the information
        
        let _num_unknown = read_u32(sysdic)?; // number of unique somethings; might be unique lexeme surfaces, might be feature strings, we don't need it
        // 0x10
        // this information is duplicated in the matrix file and we will ensure that it is consistent
        let num_sysdic_left_contexts  = read_u32(sysdic)?;
        let num_sysdic_right_contexts = read_u32(sysdic)?;
        
        // 0x18
        let linkbytes = read_u32(sysdic)?; // number of bytes used to store the dual-array trie
        if linkbytes%8 != 0
        {
            return Err("dictionary broken: link table stored with number of bytes that is not a multiple of 8");
        }
        let tokenbytes = read_u32(sysdic)?; // number of bytes used to store the list of tokens
        if tokenbytes%16 != 0
        {
            return Err("dictionary broken: token table stored with number of bytes that is not a multiple of 16");
        }
        // 0x20
        let featurebytes = read_u32(sysdic)?; // number of bytes used to store the feature string pile
        seek_rel_4(sysdic)?;
        
        let encoding = read_nstr(sysdic, 0x20)?;
        if encoding != "UTF-8"
        {
            return Err("only UTF-8 dictionaries are supported. stop using legacy encodings for infrastructure!");
        }
        
        //println!("start reading link table");
        let mut links : Vec<Link> = Vec::with_capacity((linkbytes/8) as usize);
        for _i in 0..(linkbytes/8)
        {
            links.push(Link::read(sysdic)?);
        }
        //println!("end reading link table");
        
        let mut tokens : Vec<FormatToken> = Vec::with_capacity((tokenbytes/16) as usize);
        //println!("start reading tokens");
        for _i in 0..(tokenbytes/16)
        {
            tokens.push(FormatToken::read(sysdic, tokens.len() as u32)?);
        }
        //println!("end reading tokens");
        
        //println!("feature table starts at {}", seek_rel(sysdic, 0).unwrap());
        //println!("going to read {} bytes", featurebytes);
        
        let mut feature_string_bytes : Vec<u8> = Vec::with_capacity(featurebytes as usize);
        feature_string_bytes.resize(featurebytes as usize, 0);
        //println!("double checking {}", feature_string_bytes.len());
        
        //println!("start reading feature table");
        match sysdic.read_exact(&mut feature_string_bytes)
        {
            Err(_) => { return Err("IO error") }
            Ok(_) => { }
        }
        //println!("end reading feature table");
        
        //println!("ended on {}", seek_rel(sysdic, 0).unwrap());
        
        //println!("start collecting dictionary");
        let dictionary = dart::collect_links_into_hashmap(&links, &tokens);
        drop(links);
        //println!("end collecting dictionary");
        
        let mut contains_longer : HashSet<String> = HashSet::new();
        
        //println!("start building prefix set");
        for entry in dictionary.keys()
        {
            let codepoints = codepoints(entry);
            for i in 1..codepoints.len()-1
            {
                contains_longer.insert(codepoints[0..i].iter().collect());
            }
        }
        //println!("end building prefix set");
        
        //println!("start reading matrix");
        let left_edges  = read_u16(matrix)?;
        let right_edges = read_u16(matrix)?;
        
        if num_sysdic_left_contexts != left_edges as u32 || num_sysdic_right_contexts != right_edges as u32
        {
            return Err("sys.dic and matrix.bin have inconsistent left/right edge counts");
        }
        
        let connections = left_edges as u32 * right_edges as u32;
        
        let mut connection_matrix : Vec<i16> = Vec::with_capacity(connections as usize);
        connection_matrix.resize(connections as usize, 0);
        read_i16_buffer(matrix, &mut connection_matrix)?;
        //println!("end reading matrix");
        
        //println!("start preparing heuristic");
        let mut min_edge_cost_ever : i64 = 0;
        for edge_cost in &connection_matrix
        {
            min_edge_cost_ever = std::cmp::min(min_edge_cost_ever, *edge_cost as i64);
        }
        //println!("end preparing heuristic");
        
        Ok(Dict
        { dictionary,
          contains_longer,
          feature_string_bytes,
          
          min_edge_cost_ever,
          
          left_edges,
          right_edges,
          connection_matrix,
        })
    }
    fn calculate_cost(&self, left : &LexerToken, right : &LexerToken) -> i64
    {
        if left.end != right.start
        {
            return std::i64::MAX;
        }
        
        if right.kind == TokenType::UNK
        {
            return 0;
        }
        
        if left.kind == TokenType::UNK
        {
            return right.cost as i64;
        }
        
        if left.right_context > self.left_edges
        {
            panic!("bad right_context");
        }
        if right.left_context > self.right_edges
        {
            panic!("bad left_context");
        }
        
        let connection_cost =
        self.connection_matrix
        [ left.right_context as usize 
          + self.left_edges as usize
            * right.left_context as usize
        ];
        
        return right.cost as i64 + connection_cost as i64;
    }
}

fn build_lattice(dict : &Dict, pseudo_string : &Vec<char>) -> (Vec<Vec<LexerToken>>, i64)
{
    let mut lattice : Vec<Vec<LexerToken>> = Vec::with_capacity(pseudo_string.len());
    
    let mut max_covered_index = 0usize;
    
    let mut min_token_cost_ever : i64 = 0;
    
    lattice.push(vec!(LexerToken::make_bos(lattice.len(), lattice.len()+1)));
    
    for start in 0..pseudo_string.len()
    {
        let mut end = start+1;
        let mut substring : String = pseudo_string[start..end].iter().collect();
        
        let mut lattice_column : Vec<LexerToken> = Vec::new();
        
        while dict.contains_longer.contains(&substring) || dict.dictionary.contains_key(&substring)
        {
            if let Some(matching_tokens) = dict.dictionary.get(&substring)
            {
                for token in matching_tokens
                {
                    lattice_column.push(LexerToken::from(token, start+1, end+1));
                    
                    min_token_cost_ever = std::cmp::min(min_token_cost_ever, token.cost as i64);
                }
                max_covered_index = std::cmp::max(max_covered_index, end);
            }
            
            if end >= pseudo_string.len()
            {
                break;
            }
            substring.push(pseudo_string[end]);
            end += 1;
        }
        
        if start == max_covered_index && lattice_column.len() == 0
        {
            lattice_column.push(LexerToken::make_unk(start+1, start+2));
        }
        
        lattice.push(lattice_column);
    }
    
    lattice.push(vec!(LexerToken::make_bos(lattice.len(), lattice.len()+1)));
    
    (lattice, min_token_cost_ever)
}

pub fn parse_to_lexertokens(dict : &Dict, pseudo_string : &Vec<char>) -> Option<(Vec<LexerToken>, i64)>
{
    let (lattice, min_token_cost_ever) = build_lattice(dict, pseudo_string);
    
    let result = astar(
        // start
        &(0usize, 0usize),
        // successors
        |&(column, row)|
        {
            let left = &lattice[column][row];
            
            if left.end >= lattice.len()
            {
                return vec!();
            }
            else
            {
                let mut ret : Vec<((usize, usize), i64)> = Vec::new();
                for row in 0..lattice[left.end].len()
                {
                    let right = &lattice[left.end][row];
                    ret.push(((left.end, row), dict.calculate_cost(left, right)));
                }
                return ret;
            }
        },
        // heuristic
        |&(column, row)|
        {
            let left = &lattice[column][row];
            
            if left.end < lattice.len()
            {
                let distance = lattice.len() - left.end;
                return distance as i64 * (dict.min_edge_cost_ever + min_token_cost_ever);
                //return 0;
            }
            else
            {
                return std::i64::MAX;
            }
        },
        // success
        |&(column, row)|
        {
            let left = &lattice[column][row];
            
            left.end == lattice.len()
        }
    );
    // convert result into callee-usable vector of parse tokens, tupled together with cost
    if let Some(result) = result
    {
        let mut token_events : Vec<LexerToken> = Vec::new();
        
        for event in result.0
        {
            let token = &lattice[event.0][event.1];
            if token.kind != TokenType::BOS
            {
                token_events.push(token.clone());
            }
        }
        
        return Some((token_events, result.1));
    }
    else
    {
        return None;
    }
}

pub fn parse(dict : &Dict, text : &String) -> Option<(Vec<ParserToken>, i64)>
{
    let pseudo_string = codepoints(text);
    
    let result = parse_to_lexertokens(dict, &pseudo_string);
    // convert result into callee-usable vector of parse tokens, tupled together with cost
    if let Some(result) = result
    {
        let mut lexeme_events : Vec<ParserToken> = Vec::new();
        
        for token in result.0
        {
            let surface : String = pseudo_string[token.start-1..token.end-1].iter().collect();
            let feature =
            if token.kind == TokenType::Normal
            {
                read_str_buffer(&dict.feature_string_bytes[token.feature_offset as usize..]).unwrap()
            }
            else
            {
                "UNK".to_string()
            };
            lexeme_events.push(ParserToken::build(surface, feature, token.original_id, token.kind == TokenType::UNK));
        }
        
        return Some((lexeme_events, result.1));
    }
    else
    {
        return None;
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use super::*;
    
    // concatenate surface forms of parsertoken stream, with given comma between tokens
    fn tokenstream_to_string(stream : &Vec<ParserToken>, comma : &str) -> String
    {
        let mut ret = String::new();
        
        let mut first = true;
        for token in stream
        {
            if !first
            {
                ret += comma;
            }
            ret += &token.surface;
            first = false;
        }
        return ret;
    }
    
    #[test]
    fn test_general()
    {
        let sysdic_raw = File::open("data/sys.dic").unwrap();
        let mut sysdic = BufReader::new(sysdic_raw);
        
        let matrix_raw = File::open("data/matrix.bin").unwrap();
        let mut matrix = BufReader::new(matrix_raw);
        
        let dict = Dict::load(&mut sysdic, &mut matrix).unwrap();
        
        let result = parse(&dict, &"これを持っていけ".to_string());
        
        if let Some(result) = result
        {
            for token in &result.0
            {
                println!("{}", token.feature);
            }
            let split_up_string = tokenstream_to_string(&result.0, "|");
            println!("{}", split_up_string);
            assert_eq!(split_up_string, "これ|を|持っ|て|いけ");
        }
    }
}

