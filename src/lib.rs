use std::io::BufReader;
use std::io::Read;
use std::io::Seek;

use std::str;

extern crate pathfinding;

mod strings;
mod file;
mod dart;
mod unkchar;
mod userdict;

use self::file::*;
use self::strings::*;
use self::dart::*;
use self::unkchar::*;
use self::userdict::*;

#[derive(Clone)]
#[derive(Debug)]
pub (crate) struct FormatToken {
    left_context : u16,
    right_context : u16,
    
    pos  : u16,
    cost : i64,
    
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
          cost : read_i16(sysdic)? as i64,
          original_id,
          feature_offset : read_u32(sysdic)?,
        };
        
        // seek away a u32 of padding
        seek_rel_4(sysdic)?;
        
        Ok(ret)
    }
    fn blank() -> FormatToken
    {
        FormatToken{left_context:0, right_context:0, pos:0, cost:0, original_id:0, feature_offset:0}
    }
}

#[derive(Clone)]
#[derive(Debug)]
#[derive(PartialEq)]
pub enum TokenType {
    /// Token came from a mecab dictionary.
    Normal,
    /// Token came from a user dictionary.
    User,
    /// Token over section of text not covered by dictionary (unknown).
    UNK,
    /// Used internally for virtual beginning-of-string and end-of-string tokens. Not exposed to outside functions.
    BOS,
}

#[derive(Clone)]
#[derive(Debug)]
pub struct LexerToken {
    /// Used internally during lattice pathfinding.
    left_context : u16,
    /// Used internally during lattice pathfinding.
    right_context : u16,
    
    /// I don't know what this is.
    pos  : u16,
    /// Used internally during lattice pathfinding.
    pub cost : i64,
    
    /// Location, in codepoints, of the surface of this LexerToken in the string it was parsed from.
    pub start : usize, 
    /// Corresponding ending location, in codepoints. Exclusive. (i.e. when start+1 == end, the LexerToken's surface is one codepoint long)
    pub end   : usize,
    
    // Start point (inclusive) on lattice.
    lattice_start : usize,
    // End point (exclusive) on lattice.
    lattice_end : usize,
    
    /// Origin of token. BOS and UNK are virtual origins ("beginning/ending-of-string" and "unknown", respectively). Normal means it came from the mecab dictionary.
    ///
    /// The BOS (beginning/ending-of-string) tokens are stripped away in parse_to_lexertokens.
    pub kind : TokenType,
    
    /// Unique identifier of what specific lexeme realization this is, from the mecab dictionary. changes between dictionary versions.
    pub original_id : u32,
    
    feature_offset : u32,
}

impl LexerToken {
    fn from(other : & FormatToken, start : usize, end : usize, lattice_start : usize, lattice_end : usize, kind : TokenType) -> LexerToken
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
          lattice_start,
          lattice_end,
          kind
        }
    }
    fn make_bos(start : usize, end : usize, lattice_start : usize, lattice_end : usize) -> LexerToken
    {
        LexerToken
        { left_context : 0, // context ID of EOS/BOS
          right_context : 0,
          pos : 0,
          cost : 0,
          original_id : 0,
          feature_offset : 0,
          start,
          end,
          lattice_start,
          lattice_end,
          kind : TokenType::BOS
        }
    }
}

#[derive(Clone)]
#[derive(Debug)]
pub struct ParserToken {
    /// Exact sequence of characters with which this token appeared in the string that was parsed.
    pub surface : String,
    /// Description of this token's features.
    ///
    /// The feature string contains almost all useful information, including things like part of speech, spelling, pronunciation, etc.
    ///
    /// The exact format of the feature string is dictionary-specific.
    pub feature : String,
    /// Unique identifier of what specific lexeme realization this is, from the mecab dictionary. changes between dictionary versions.
    pub original_id : u32,
    /// Origin of token.
    pub kind : TokenType,
}

impl ParserToken {
    fn build(surface : String, feature : String, original_id : u32, kind : TokenType) -> ParserToken
    {
        ParserToken
        { surface,
          feature,
          original_id,
          kind
        }
    }
}

pub struct Dict {
    sys_dic : DartDict,
    unk_dic : DartDict,
    unk_data : UnkChar,
    user_dic : Option<UserDict>,
    left_edges : u16,
    right_edges : u16,
    connection_matrix : Vec<i16>, // 2d matrix encoded as 1d
}

impl Dict {
    /// Load sys.dic and matrix.bin files into memory and prepare the data that's stored in them to be used by the parser.
    ///
    /// Returns a Dict or, on error, a string describing an error that prevented the Dict from being created.
    ///
    /// Only supports UTF-8 mecab dictionaries with a version number of 0x66.
    ///
    /// Ensures that sys.dic and matrix.bin have compatible connection matrix sizes.
    pub fn load<T : Read + Seek>(
        sysdic : &mut BufReader<T>,
        matrix : &mut BufReader<T>,
        unkdic : &mut BufReader<T>,
        unkchar : &mut BufReader<T>,
    ) -> Result<Dict, &'static str>
    {
        let sys_dic = load_mecab_dart_file(0xE1_17_21_81, sysdic)?;
        let unk_dic = load_mecab_dart_file(0xEF_71_9A_03, unkdic)?;
        let unk_data = load_char_bin(unkchar)?;
        
        let left_edges  = read_u16(matrix)?;
        let right_edges = read_u16(matrix)?;
        
        if sys_dic.left_contexts != left_edges as u32 || sys_dic.right_contexts != right_edges as u32
        {
            return Err("sys.dic and matrix.bin have inconsistent left/right edge counts");
        }
        let connections = left_edges as u32 * right_edges as u32;
        
        let mut connection_matrix : Vec<i16> = Vec::with_capacity(connections as usize);
        connection_matrix.resize(connections as usize, 0);
        read_i16_buffer(matrix, &mut connection_matrix)?;
        
        Ok(Dict
        { sys_dic,
          unk_dic,
          unk_data,
          user_dic: None,
          left_edges,
          right_edges,
          connection_matrix,
        })
    }
    pub fn load_user_dictionary<T : Read>(&mut self, userdic : &mut BufReader<T>) -> Result<(), &'static str>
    {
        self.user_dic = Some(UserDict::load(userdic)?);
        Ok(())
    }
    /// Returns the feature string belonging to a LexerToken. They are stored in a large byte buffer internally as copying them on each parse may be expensive.
    pub fn read_feature_string(&self, token : &LexerToken) -> Result<String, &'static str>
    {
        match token.kind
        {
            TokenType::UNK => self.unk_dic.feature_get(token.feature_offset),
            TokenType::Normal | TokenType::BOS => self.sys_dic.feature_get(token.feature_offset),
            TokenType::User => Ok(self.user_dic.as_ref().unwrap().feature_get(token.feature_offset)),
        }
    }
    fn calculate_cost(&self, left : &LexerToken, right : &LexerToken) -> i64
    {
        if left.lattice_end != right.lattice_start
        {
            panic!("disconnected nodes");
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
        
        right.cost as i64 + connection_cost as i64
    }
    fn may_contain(&self, find : &str) -> bool
    {
        self.sys_dic.may_contain(find) || self.user_dic.as_ref().map(|x| x.may_contain(find)).unwrap_or_else(|| false)
    }
}

fn build_lattice_column(dict: &Dict, text : &str, mut start : usize, lattice_len : usize) -> (Vec<LexerToken>, usize)
{
    let mut offset = 0;
    while match text[start+offset..].chars().next() { Some(' ') | Some('\n') | Some('\r') => true, _ => false }
    {
        offset += 1;
    }
    start += offset;
    
    let mut index_iter = text[start..].char_indices();
    let mut end = start;
    let mut first_char = ' ';
    match index_iter.next()
    {
        Some((_, c)) =>
        {
            end += c.len_utf8();
            first_char = c;
        }
        None => return (vec!(LexerToken::make_bos(0, 0, lattice_len, lattice_len+1)), offset),
    };
    
    let mut substring : &str = &text[start..end];
    
    let mut lattice_column : Vec<LexerToken> = Vec::with_capacity(20);
    
    while dict.may_contain(&substring)
    {
        if let Some(matching_tokens) = dict.sys_dic.dic_get(&substring)
        {
            lattice_column.reserve(matching_tokens.len());
            for token in matching_tokens
            {
                lattice_column.push(LexerToken::from(token, start, end, lattice_len, lattice_len+end-start+offset, TokenType::Normal));
            }
        }
        if let Some(user_dic) = &dict.user_dic
        {
            if let Some(matching_tokens) = user_dic.dic_get(&substring)
            {
                lattice_column.reserve(matching_tokens.len());
                for token in matching_tokens
                {
                    lattice_column.push(LexerToken::from(token, start, end, lattice_len, lattice_len+end-start+offset, TokenType::User));
                }
            }
        }
        
        match index_iter.next()
        {
            Some((_, c)) =>
            {
                end += c.len_utf8();
                substring = &text[start..end];
            }
            None => break
        }
    }
    
    if dict.unk_data.always_process(first_char) || lattice_column.is_empty()
    {
        let start_type = &dict.unk_data.get_type(first_char);
        let mut find_unk = text[start..].char_indices();
        let mut unk_end = start;
        
        let mut unk_indices = vec!();
        while let Some((_, c)) = find_unk.next()
        {
            if dict.unk_data.has_type(c, start_type.number)
            {
                unk_end += c.len_utf8();
                unk_indices.push(unk_end);
            }
            else
            {
                break;
            }
        }
        
        if !unk_indices.is_empty()
        {
            if let Some(matching_tokens) = dict.unk_dic.dic_get(&start_type.name)
            {
                lattice_column.reserve(matching_tokens.len() * (start_type.prefix_group_len as usize + start_type.greedy_group as usize));
                for token in matching_tokens
                {
                    if start_type.greedy_group
                    {
                        let end = *unk_indices.last().unwrap();
                        lattice_column.push(LexerToken::from(token, start, end, lattice_len, lattice_len+end-start+offset, TokenType::UNK));
                    }
                    if start_type.prefix_group_len > 0
                    {
                        for i in 0..start_type.prefix_group_len
                        {
                            if let Some(end) = unk_indices.get(i as usize)
                            {
                                lattice_column.push(LexerToken::from(token, start, *end, lattice_len, lattice_len+end-start+offset, TokenType::UNK));
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    
    (lattice_column, offset)
}

fn build_lattice(dict : &Dict, text : &str) -> Vec<Vec<LexerToken>>
{
    let mut lattice : Vec<Vec<LexerToken>> = Vec::with_capacity(text.len()+2);
    
    lattice.push(vec!(LexerToken::make_bos(0, 0, lattice.len(), lattice.len()+1)));
    
    let mut skip_until_after = 0;
    
    for i in 0..text.len()
    {
        if i < skip_until_after || !text.is_char_boundary(i)
        {
            lattice.push(Vec::new());
        }
        else
        {
            let (column, skipnext) = build_lattice_column(dict, text, i, lattice.len());
            skip_until_after = i+skipnext;
            lattice.push(column);
        }
    }
    
    lattice
}

/// Tokenizes a char slice by creating a lattice of possible tokens over it and finding the lowest-cost path over that lattice. Returns a list of LexerTokens and the cost of the tokenization.
///
/// The dictionary defines what tokens exist, how they appear in the string, their costs, and the costs of their possible connections.
///
/// Returns a vector listing the LexerTokens on the chosen path and the cost the path took. Cost can be negative.
///
/// It's possible for multiple paths to tie for the lowest cost. It's not defined which path is returned in that case.
pub fn parse_to_lexertokens(dict : &Dict, text : &str) -> Option<(Vec<LexerToken>, i64)>
{
    let lattice = build_lattice(dict, text);
    
    let result = pathfinding::directed::dijkstra::dijkstra(
        // start
        &(0usize, 0usize),
        // successors
        |&(column, row)|
        {
            let left = &lattice[column][row];
            lattice[left.lattice_end].iter().enumerate().map(move |(row, right)| ((left.lattice_end, row), dict.calculate_cost(left, right)))
        },
        // success
        |&(column, row)| lattice[column][row].lattice_end == lattice.len()
    );
    
    // convert result into callee-usable vector of parse tokens, tupled together with cost
    if let Some(result) = result
    {
        let token_events : Vec<LexerToken> = result.0[1..result.0.len()].into_iter().map(|(column, row)| lattice[*column][*row].clone()).collect();
        Some((token_events, result.1))
    }
    else
    {
        None
    }
}

/// Tokenizes a string by creating a lattice of possible tokens over it and finding the lowest-cost path over that lattice. Returns a list of ParserToken and the cost of the tokenization.
///
/// The dictionary defines what tokens exist, how they appear in the string, their costs, and the costs of their possible connections.
/// 
/// Generates ParserTokens over the chosen path and returns a list of those ParserTokens and the cost the path took. Cost can be negative.
/// 
/// It's possible for multiple paths to tie for the lowest cost. It's not defined which path is returned in that case.
pub fn parse(dict : &Dict, text : &str) -> Option<(Vec<ParserToken>, i64)>
{
    let result = parse_to_lexertokens(dict, &text);
    // convert result into callee-usable vector of parse tokens, tupled together with cost
    if let Some(result) = result
    {
        let mut lexeme_events : Vec<ParserToken> = Vec::with_capacity(result.0.len());
        
        for token in result.0
        {
            let surface : String = text[token.start..token.end].to_string();
            let feature = dict.read_feature_string(&token).unwrap();
            lexeme_events.push(ParserToken::build(surface, feature, token.original_id, token.kind));
        }
        
        Some((lexeme_events, result.1))
    }
    else
    {
        None
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
        ret
    }
    
    fn assert_parse(dict : &Dict, input : &'static str, truth : &'static str)
    {
        println!("testing parse...");
        let result = parse(dict, &input.to_string()).unwrap();
        
        for token in &result.0
        {
            println!("{}", token.feature);
        }
        let split_up_string = tokenstream_to_string(&result.0, "|");
        println!("{}", split_up_string);
        
        assert_eq!(split_up_string, truth);
    }
    
    #[test]
    fn test_various()
    {
        // you need to acquire a mecab dictionary and place these files here manually
        // These tests will probably fail if you use a different dictionary than me. That's normal. Different dicionaries parse differently.
        let mut sysdic = BufReader::new(File::open("data/sys.dic").unwrap());
        let mut matrix = BufReader::new(File::open("data/matrix.bin").unwrap());
        let mut unkdic = BufReader::new(File::open("data/unk.dic").unwrap());
        let mut unkdef = BufReader::new(File::open("data/char.bin").unwrap());
        
        let mut dict = Dict::load(&mut sysdic, &mut matrix, &mut unkdic, &mut unkdef).unwrap();
        
        // general nonbrokenness
        assert_parse(&dict,
          "これを持っていけ",
          "これ|を|持っ|て|いけ"
        );
        
        // lots of text
        assert_parse(&dict,
          "メタプログラミング (metaprogramming) とはプログラミング技法の一種で、ロジックを直接コーディングするのではなく、あるパターンをもったロジックを生成する高位ロジックによってプログラミングを行う方法、またその高位ロジックを定義する方法のこと。主に対象言語に埋め込まれたマクロ言語によって行われる。",
          "メタ|プログラミング|(|metaprogramming|)|と|は|プログラミング|技法|の|一種|で|、|ロジック|を|直接|コーディング|する|の|で|は|なく|、|ある|パターン|を|もっ|た|ロジック|を|生成|する|高位|ロジック|に|よっ|て|プログラミング|を|行う|方法|、|また|その|高位|ロジック|を|定義|する|方法|の|こと|。|主に|対象|言語|に|埋め込ま|れ|た|マクロ|言語|に|よっ|て|行わ|れる|。"
        );
        
        // lorem ipsum
        // This test will CERTAINLY fail if you don't have the same mecab dictionary.
        assert_parse(&dict,
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
          "Lorem|i|p|s|u|m|d|o|l|o|r|s|i|t|a|m|e|t|,|consectetur|adipiscing|elit|,|sed|do|eiusmod|tempor|incididunt|u|t|l|a|b|o|r|e|e|t|dolore|magna|aliqua|."
        );
        
        // user dictionary
        assert_parse(&dict, "飛行機", "飛行|機");
        dict.load_user_dictionary(&mut BufReader::new(File::open("data/userdict.csv").unwrap())).unwrap();
        assert_parse(&dict, "飛行機", "飛行機");
    }
}

