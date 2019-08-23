#![allow(clippy::suspicious_else_formatting)]
use std::io::Cursor;
use std::io::Read;
use std::io::Seek;

use std::str;

extern crate pathfinding;

mod blob;
mod file;
mod dart;
mod unkchar;
mod userdict;

use self::file::*;
use self::dart::*;
use self::unkchar::*;
use self::userdict::*;

pub use self::blob::Blob;

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
    #[allow(clippy::cast_lossless)]
    fn read<T : Read + std::io::Seek>(sysdic : &mut T, original_id : u32) -> Result<FormatToken, &'static str>
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
}

#[derive(Clone)]
#[derive(Copy)]
#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Eq)]
#[derive(Hash)]
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
    /// Cost updated to include right-edge connection cost after parsing.
    pub real_cost : i64, 
    
    /// Location, in bytes, of the surface of this LexerToken in the string it was parsed from.
    pub start : usize, 
    /// Corresponding ending location, in bytes. Exclusive. (i.e. when start+1 == end, the LexerToken's surface is one byte long)
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
    
    pub feature_offset : u32,
}

impl LexerToken {
    fn from(other : & FormatToken, start : usize, end : usize, lattice_start : usize, lattice_end : usize, kind : TokenType) -> LexerToken
    {
        LexerToken
        { left_context : other.left_context,
          right_context : other.right_context,
          pos : other.pos,
          cost : other.cost,
          real_cost : 0,
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
          real_cost : 0,
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
pub struct ParserToken<'text, 'dict> {
    /// Exact sequence of characters with which this token appeared in the string that was parsed.
    pub surface : &'text str,
    /// Description of this token's features.
    ///
    /// The feature string contains almost all useful information, including things like part of speech, spelling, pronunciation, etc.
    ///
    /// The exact format of the feature string is dictionary-specific.
    pub feature : &'dict str,
    /// Unique identifier of what specific lexeme realization this is, from the mecab dictionary. changes between dictionary versions.
    pub original_id : u32,
    /// Origin of token.
    pub kind : TokenType,
}

impl<'text, 'dict> ParserToken<'text, 'dict> {
    fn build(surface : &'text str, feature : &'dict str, original_id : u32, kind : TokenType) -> Self
    {
        ParserToken
        { surface,
          feature,
          original_id,
          kind
        }
    }
}

struct EdgeInfo {
    full_cache_enabled : bool,
    
    fast_edge_enabled : bool,
    fast_edge_map_left : Vec<u16>,
    fast_edge_map_right : Vec<u16>,
    fast_edge_left_edges : usize,
    fast_matrix_cache : Vec<i16>,
    
    blob : Blob,
}

impl EdgeInfo {
    fn new(blob : Blob) -> EdgeInfo
    {
        EdgeInfo {
            full_cache_enabled : false,
            fast_edge_enabled : false,
            fast_edge_map_left : Vec::new(),
            fast_edge_map_right : Vec::new(),
            fast_edge_left_edges : 0,
            fast_matrix_cache : Vec::new(),
            blob
        }
    }
}

pub struct Dict {
    sys_dic : DartDict,
    unk_dic : DartDict,
    unk_data : UnkChar,
    user_dic : Option<UserDict>,
    
    use_space_stripping : bool,
    use_unk_forced_processing : bool,
    use_unk_greedy_grouping : bool,
    use_unk_prefix_grouping : bool,
    
    left_edges : u16,
    right_edges : u16,
    
    matrix : EdgeInfo
}

impl Dict {
    /// Load sys.dic and matrix.bin files into memory and prepare the data that's stored in them to be used by the parser.
    ///
    /// Returns a Dict or, on error, a string describing an error that prevented the Dict from being created.
    ///
    /// Only supports UTF-8 mecab dictionaries with a version number of 0x66.
    ///
    /// Ensures that sys.dic and matrix.bin have compatible connection matrix sizes.
    #[allow(clippy::cast_lossless)]
    pub fn load(
        sysdic : Blob,
        unkdic : Blob,
        matrix : Blob,
        unkchar : Blob,
    ) -> Result<Dict, &'static str>
    {
        let sys_dic = load_mecab_dart_file(sysdic)?;
        let unk_dic = load_mecab_dart_file(unkdic)?;
        let unk_data = load_char_bin(&mut Cursor::new(unkchar))?;
        
        let mut matrix_cursor = Cursor::new(matrix.as_ref());
        let left_edges  = read_u16(&mut matrix_cursor)?;
        let right_edges = read_u16(&mut matrix_cursor)?;
        
        if sys_dic.left_contexts != left_edges as u32 || sys_dic.right_contexts != right_edges as u32
        {
            return Err("sys.dic and matrix.bin have inconsistent left/right edge counts");
        }
        
        Ok(Dict {
            sys_dic,
            unk_dic,
            unk_data,
            user_dic: None,
            use_space_stripping : true,
            use_unk_forced_processing : true,
            use_unk_greedy_grouping : true,
            use_unk_prefix_grouping : true,
            left_edges,
            right_edges,
            
            matrix : EdgeInfo::new(matrix)
        })
    }
    /// Load a user dictionary, comma-separated fields.
    ///
    /// The first four fields are the surface, left context ID, right context ID, and cost of the token.
    ///
    /// Everything past the fourth comma is treated as pure text and is the token's feature string. It is itself normally a list of comma-separated fields with the same format as the feature strings of the main mecab dictionary.
    pub fn load_user_dictionary(&mut self, userdic : Blob) -> Result<(), &'static str>
    {
        let mut userdic = Cursor::new(userdic);
        self.user_dic = Some(UserDict::load(&mut userdic)?);
        Ok(())
    }
    /// Returns the feature string belonging to a LexerToken.
    pub fn read_feature_string(&self, token : &LexerToken) -> &str
    {
        self.read_feature_string_by_source(token.kind, token.feature_offset)
    }
    /// Calling this with values not taken from a real token is unsupported behavior.
    pub fn read_feature_string_by_source(&self, kind : TokenType, offset : u32) -> &str
    {
        match kind
        {
            TokenType::UNK => self.unk_dic.feature_get(offset),
            TokenType::Normal | TokenType::BOS => self.sys_dic.feature_get(offset),
            TokenType::User => self.user_dic.as_ref().unwrap().feature_get(offset),
        }
    }
    /// Optional feature for applications that need to use as little memory as possible without accessing disk constantly. "Undocumented". May be removed at any time for any reason.
    ///
    /// Does nothing if the prepare_full_matrix_cache has already been called.
    #[allow(clippy::cast_lossless)]
    pub fn prepare_fast_matrix_cache(&mut self, fast_left_edges : Vec<u16>, fast_right_edges : Vec<u16>)
    {
        let mut matrix = &mut self.matrix;
        
        if matrix.full_cache_enabled
        {
            return;
        }
        
        let mut left_map  = vec!(!0u16; self.left_edges  as usize);
        let mut right_map = vec!(!0u16; self.right_edges as usize);
        for (i, left) in fast_left_edges.iter().enumerate()
        {
            left_map[*left as usize] = i as u16;
        }
        for (i, right) in fast_right_edges.iter().enumerate()
        {
            right_map[*right as usize] = i as u16;
        }
        
        let mut submatrix = vec!(0i16; fast_left_edges.len() * fast_right_edges.len());
        for (y, right) in fast_right_edges.iter().enumerate()
        {
            let mut row = vec!(0i16; self.left_edges as usize);
            let location = self.left_edges as u64 * *right as u64;
            let mut reader = Cursor::new(&matrix.blob);
            reader.seek(std::io::SeekFrom::Start(4 + location*2)).unwrap();
            read_i16_buffer(&mut reader, &mut row).unwrap();
            for (i, left) in fast_left_edges.iter().enumerate()
            {
                submatrix[y * fast_left_edges.len() + i] = row[*left as usize];
            }
        }
        
        matrix.fast_edge_enabled = true;
        matrix.fast_edge_map_left  = left_map;
        matrix.fast_edge_map_right = right_map;
        matrix.fast_edge_left_edges = fast_left_edges.len();
        matrix.fast_matrix_cache = submatrix;
    }
    /// Load the entire connection matrix into memory. Suitable for small dictionaries, but is actually SLOWER than using prepare_fast_matrix_cache properly for extremely large dictionaries, like modern versions of unidic. "Undocumented".
    ///
    /// Overrides prepare_fast_matrix_cache if it has been called before.
    pub fn prepare_full_matrix_cache(&mut self)
    {
        let mut matrix = &mut self.matrix;
        
        matrix.full_cache_enabled = true;
        matrix.fast_edge_enabled = false;
        matrix.fast_edge_map_left  = Vec::new();
        matrix.fast_edge_map_right = Vec::new();
        matrix.fast_edge_left_edges = 0;
        matrix.fast_matrix_cache = Vec::new();
        
        let size = self.left_edges as usize * self.right_edges as usize;
        let mut new_fast_cache = vec!(0; size);
        
        let mut reader = Cursor::new(&matrix.blob);
        reader.seek(std::io::SeekFrom::Start(4)).unwrap();
        read_i16_buffer(&mut reader, &mut new_fast_cache[..]).unwrap();
        
        matrix.fast_matrix_cache = new_fast_cache;
    }
    #[allow(clippy::cast_lossless)]
    fn access_matrix(&self, left : u16, right : u16) -> i16
    {
        let matrix = &self.matrix;
        if matrix.full_cache_enabled
        {
            let loc = self.left_edges as usize * right as usize + left as usize;
            return matrix.fast_matrix_cache[loc];
        }
        
        if matrix.fast_edge_enabled
        {
            let new_left  = matrix.fast_edge_map_left [left  as usize];
            let new_right = matrix.fast_edge_map_right[right as usize];
            if new_left != !0u16 && new_right != !0u16
            {
                let loc = matrix.fast_edge_left_edges * new_right as usize + new_left as usize;
                return matrix.fast_matrix_cache[loc];
            }
        }
        
        let location = self.left_edges as u32 * right as u32 + left as u32;
        
        // the 4 is for the two u16s at the beginning that specify the shape of the matrix
        let offset = 4 + location as usize * 2;
        let cost = &matrix.blob[offset..offset + 2];
        i16::from_le_bytes([cost[0], cost[1]])
    }
    #[allow(clippy::cast_lossless)]
    fn calculate_cost(&self, left : &LexerToken, right : &LexerToken) -> i64
    {
        if left.lattice_end != right.lattice_start
        {
            panic!("disconnected nodes");
        }
        if left.right_context >= self.left_edges
        {
            panic!("bad right_context");
        }
        if right.left_context >= self.right_edges
        {
            panic!("bad left_context");
        }
        
        right.cost as i64 + self.access_matrix(left.right_context, right.left_context) as i64
    }
    fn may_contain(&self, find : &str) -> bool
    {
        self.sys_dic.may_contain(find) || self.user_dic.as_ref().map(|x| x.may_contain(find)).unwrap_or_else(|| false)
    }
    /// Set whether the 0x20 whitespace stripping behavior is enabled. Returns the previous value of the setting.
    ///
    /// Enabled by default.
    ///
    /// When enabled, spaces are virtually added to the front of the next token/tokens during lattice construction. This has the effect of turning 0x20 whitespace sequences into forced separators without affecting connection costs, but makes it slightly more difficult to reconstruct the exact original text from the output of the parser.
    pub fn set_space_stripping(&mut self, setting : bool) -> bool
    {
        let prev = self.use_space_stripping;
        self.use_space_stripping = setting;
        prev
    }
    /// Set whether support for forced unknown token processing is enabled. Returns the previous value of the setting.
    ///
    /// Enabled by default.
    ///
    /// When the parser's input string has locations where no entries can be found in the dictionary, the parser has to fill that location with unknown tokens. The unknown tokens are made by grouping up as many compatible characters as possible AND/OR grouping up every group of compatible characters from a length of 1 to a length of N. Whether either type of grouping is done (and how long the maximum prefix group is) is specified for each character in the unknown character data (usually char.bin).
    ///
    /// The unknown character data can also specify that certain character types always trigger grouping into unknown tokens, even if the given location in the input string can be found in a normal dictionary. Disabling this setting will override that data and cause the lattice builder to ONLY create unknown tokens when nothing can be found in a normal dictionary.
    ///
    /// If all unknown character processing at some problematic point in the input string fails for some reason, such as a defective unknown character data file, or one or both of the grouping modes being disabled, then that problematic point in the input string will create a single-character unknown token.
    ///
    /// When enabled, the unknown character data's flag for forcing processing is observed. When disabled, it is ignored, and processing is never forced.
    pub fn set_unk_forced_processing(&mut self, setting : bool) -> bool
    {
        let prev = self.use_unk_forced_processing;
        self.use_unk_forced_processing = setting;
        prev
    }
    /// Set whether greedy grouping behavior is enabled. Returns the previous value of the setting.
    ///
    /// Enabled by default.
    ///
    /// When enabled, problematic locations in the input string will (if specified in the unknown character data) be greedily grouped into an unknown token, covering all compatible characters.
    ///
    /// Note that this does not prevent real words inside of the grouping from being detected once the lattice constructor comes around to them, which means that greedy grouping does not necessarily override prefix grouping, and for some character types, the unknown character data will have both greedy grouping and prefix grouping enabled.
    pub fn set_unk_greedy_grouping(&mut self, setting : bool) -> bool
    {
        let prev = self.use_unk_greedy_grouping;
        self.use_unk_greedy_grouping = setting;
        prev
    }
    /// Set whether greedy grouping behavior is enabled. Returns the previous value of the setting.
    ///
    /// Enabled by default. See the documentation for the other set_unk_ functions for an explanation of what unknown token prefix grouping is.
    pub fn set_unk_prefix_grouping(&mut self, setting : bool) -> bool
    {
        let prev = self.use_unk_prefix_grouping;
        self.use_unk_prefix_grouping = setting;
        prev
    }
}

fn build_lattice_column(dict: &Dict, text : &str, mut start : usize, lattice_len : usize) -> (Vec<LexerToken>, usize)
{
    // skip spaces
    let mut offset = 0;
    while dict.use_space_stripping && start < text.len() && text[start..].starts_with(' ')
    {
        offset += 1;
        start += 1;
    }
    
    // find first character, make a BOS(EOS) column if there is none
    let mut index_iter = text[start..].char_indices();
    let mut end = start;
    let first_char = match index_iter.next()
    {
        Some((_, c)) =>
        {
            end += c.len_utf8();
            c
        }
        None => return (vec!(LexerToken::make_bos(0, 0, lattice_len, lattice_len+1+offset)), offset)
    };
    
    let mut substring : &str = &text[start..end];
    let mut lattice_column : Vec<LexerToken> = Vec::with_capacity(20);
    
    // find all tokens starting at this point in the string
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
    
    // build unknown tokens if appropriate
    let start_type = &dict.unk_data.get_type(first_char);
    
    if (dict.use_unk_greedy_grouping || dict.use_unk_prefix_grouping)
       && ((dict.use_unk_forced_processing && dict.unk_data.always_process(first_char))
           || lattice_column.is_empty())
    {
        let mut unk_end = start;
        
        let do_greedy = dict.use_unk_greedy_grouping && start_type.greedy_group;
        let do_prefix = dict.use_unk_prefix_grouping && start_type.prefix_group_len > 0;
        let mut prefix_len = if do_prefix { start_type.prefix_group_len } else { 0 } as usize;
        
        // find possible split points and furthest allowed ending in advance
        let mut unk_indices = vec!();
        for (_, c) in text[start..].char_indices()
        {
            if dict.unk_data.has_type(c, start_type.number)
            {
                unk_end += c.len_utf8();
                unk_indices.push(unk_end);
                // stop building when necessary
                if !do_greedy && unk_indices.len() >= prefix_len
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
        prefix_len = std::cmp::min(prefix_len, unk_indices.len());
        
        if let Some(matching_tokens) = dict.unk_dic.dic_get(&start_type.name)
        {
            lattice_column.reserve(matching_tokens.len() * (start_type.prefix_group_len as usize + start_type.greedy_group as usize));
            for token in matching_tokens
            {
                if do_greedy
                {
                    lattice_column.push(LexerToken::from(token, start, unk_end, lattice_len, lattice_len+unk_end-start+offset, TokenType::UNK));
                }
                for end in unk_indices[0..prefix_len].iter()
                {
                    lattice_column.push(LexerToken::from(token, start, *end, lattice_len, lattice_len+end-start+offset, TokenType::UNK));
                }
            }
        }
    }
    
    let first_char_len = first_char.len_utf8();
    let mut build_unknown_single = |name|
    {
        if lattice_column.is_empty()
        {
            if let Some(default_tokens) = dict.unk_dic.dic_get(name)
            {
                if let Some(first_token) = default_tokens.iter().next()
                {
                    lattice_column.push(LexerToken::from(first_token, start, start+first_char_len, lattice_len, lattice_len+first_char_len+offset, TokenType::UNK));
                }
            }
        }
    };
    
    // build fallback token if appropriate
    build_unknown_single(&start_type.name);
    build_unknown_single("DEFAULT");
    if lattice_column.is_empty()
    {
        panic!("unknown chars dictionary has a broken DEFAULT token");
    }
    
    (lattice_column, offset)
}

fn build_lattice(dict : &Dict, text : &str) -> Vec<Vec<LexerToken>>
{
    let mut lattice : Vec<Vec<LexerToken>> = Vec::with_capacity(text.char_indices().count()+2);
    
    lattice.push(vec!(LexerToken::make_bos(0, 0, lattice.len(), lattice.len()+1)));
    
    let mut skip_until_after = 0;
    
    for i in 0..=text.len()
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
    
    //let result : Option<(Vec<(usize, usize)>, i64)> = Some((vec!((0,0),(0,0)), 0));
    
    // pathfinding's dijkstra implementation seems to be buggy and gives non-optimal parses in some situations.
    // its astar implementation always produces an optimal parse, but it's a lot slower.
    // FIXME: write my own dijkstra or astar implementation
    
    //let result = pathfinding::directed::dijkstra::dijkstra(
    let result = pathfinding::directed::astar::astar(
        // start
        &(0usize, 0usize),
        // successors
        |&(column, row)|
        {
            let left = &lattice[column][row];
            lattice[left.lattice_end].iter().enumerate().map(move |(row, right)| ((left.lattice_end, row), dict.calculate_cost(left, right)))
        },
        // heuristic
        |&(column, _)|
        {
            // this is just as good as a heuristic derived from real minimums on real data
            let dist = lattice.len() as i64 - column as i64;
            -66000*2*dist
        },
        // success
        |&(column, row)| lattice[column][row].lattice_end == lattice.len()
    );
    
    // convert result into callee-usable vector of parse tokens, tupled together with cost
    if let Some(result) = result
    {
        let mut token_events : Vec<LexerToken> = result.0[..].iter().map(|(column, row)| lattice[*column][*row].clone()).collect();
        #[allow(clippy::cast_lossless)]
        for i in 1..result.0.len()-1
        {
            let left = &token_events[i-1];
            let right = &token_events[i];
            let edge_cost =  dict.access_matrix(left.right_context, right.left_context);
            token_events[i].real_cost = token_events[i].cost + edge_cost as i64;
        }
        token_events.pop();
        token_events.remove(0);
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
pub fn parse<'dict, 'text>(dict : &'dict Dict, text : &'text str) -> Option<(Vec<ParserToken<'text, 'dict>>, i64)>
{
    let result = parse_to_lexertokens(dict, &text);
    // convert result into callee-usable vector of parse tokens, tupled together with cost
    if let Some(result) = result
    {
        let mut lexeme_events : Vec<ParserToken> = Vec::with_capacity(result.0.len());
        
        for token in result.0
        {
            let surface = &text[token.start..token.end];
            let feature = dict.read_feature_string(&token);
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
    
    fn assert_implements_sync<T>() where T: Sync {}
    fn assert_implements_send<T>() where T: Send {}
    
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
    
    fn assert_parse(dict : &Dict, input : &str, truth : &str)
    {
        println!("testing parse...");
        let result = parse(dict, input).unwrap();
        
        for token in &result.0
        {
            println!("{}", token.feature);
        }
        let split_up_string = tokenstream_to_string(&result.0, "|");
        println!("{}", split_up_string);
        
        assert_eq!(split_up_string, truth);
    }
    
    fn file_to_string(file : &mut File) -> String
    {
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        text
    }
    
    #[test]
    fn test_various()
    {
        assert_implements_sync::<Dict>();
        assert_implements_send::<Dict>();
        
        // you need to acquire a mecab dictionary and place these files here manually
        // These tests will probably fail if you use a different dictionary than me. That's normal. Different dicionaries parse differently.
        let sysdic = Blob::open("data/sys.dic").unwrap();
        let unkdic = Blob::open("data/unk.dic").unwrap();
        let matrix = Blob::open("data/matrix.bin").unwrap();
        let unkdef = Blob::open("data/char.bin").unwrap();
        
        let mut dict = Dict::load(sysdic, unkdic, matrix, unkdef).unwrap();
        
        // general nonbrokenness
        assert_parse(&dict,
          "これ",
          "これ"
        );

        assert_parse(&dict,
          "これを",
          "これ|を"
        );

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
        /*
        // original version
        assert_parse(&dict,
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
          "Lorem|ipsum|dolor|s|i|t|a|m|e|t|,|consectetur|adipiscing|elit|,|sed|do|eiusmod|tempor|incididunt|u|t|l|a|b|o|r|e|e|t|dolore|magna|aliqua|."
        );
        */
        // version that should be agnostic w/r/t spoken language vs written language variants of unidic 2.3.0
        assert_parse(&dict,
          "Lorem sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
          "Lorem|s|i|t|a|m|e|t|,|consectetur|adipiscing|elit|,|sed|do|eiusmod|tempor|incididunt|u|t|l|a|b|o|r|e|e|t|dolore|magna|aliqua|."
        );
        
        // string that is known to trigger problems with at least one buggy pathfinding algorithm notmecab used before
        /*
        // original version
        assert_parse(&dict,
          "だっでおら、こんな、こんなにっ！飛車角のこと、好きなんだでっ！！！！！！",
          "だっ|で|おら|、|こんな|、|こんな|に|っ|！|飛車|角|の|こと|、|好き|な|ん|だ|で|っ|！|！|！|！|！|！"
        );
        */
        // version that should be agnostic w/r/t spoken language vs written language variants of unidic 2.3.0
        assert_parse(&dict,
          "だっでおら、こんな、こんなにっ！飛車角のこと、好きなんだ！！！！！！",
          "だっ|で|おら|、|こんな|、|こんな|に|っ|！|飛車|角|の|こと|、|好き|な|ん|だ|！|！|！|！|！|！"
        );
        
        // unknown character token stuff
        assert_parse(&dict, "噛", "噛");
        assert_parse(&dict, "噛 ", "噛");
        assert_parse(&dict, "噛\n", "噛|\n");
        
        // overrides
        dict.set_space_stripping(false);
        assert_parse(&dict, "a b", "a| |b");
        dict.set_space_stripping(true);
        assert_parse(&dict, "」   ", "」");
        
        assert_parse(&dict, "噛噛", "噛噛");
        dict.set_unk_prefix_grouping(false);
        dict.set_unk_greedy_grouping(false);
        assert_parse(&dict, "噛噛", "噛|噛");
        dict.set_unk_prefix_grouping(true);
        dict.set_unk_greedy_grouping(true);
        
        assert_parse(&dict, "programmprogram", "programmprogram");
        dict.set_unk_forced_processing(false);
        assert_parse(&dict, "programmprogram", "program|m|program");
        dict.set_unk_forced_processing(true);
        
        // hentaigana
        assert_parse(&dict, "𛁁", "𛁁");
        
        // user dictionary
        assert_parse(&dict, "飛行機", "飛行|機");
        dict.load_user_dictionary(Blob::open("data/userdict.csv").unwrap()).unwrap();
        assert_parse(&dict, "飛行機", "飛行機");
        
        
        if let Ok(mut common_left_edge_file) = File::open("data/common_edges_left.txt")
        {
            if let Ok(mut common_right_edge_file) = File::open("data/common_edges_right.txt")
            {
                let fast_edges_left_text  = file_to_string(&mut common_left_edge_file);
                let fast_edges_right_text = file_to_string(&mut common_right_edge_file);
                let fast_edges_left  = fast_edges_left_text .lines().map(|x| x.parse::<u16>().unwrap()).collect::<Vec<_>>();
                let fast_edges_right = fast_edges_right_text.lines().map(|x| x.parse::<u16>().unwrap()).collect::<Vec<_>>();
                dict.prepare_fast_matrix_cache(fast_edges_left, fast_edges_right);
                
                assert_parse(&dict,
                  "メタプログラミング (metaprogramming) とはプログラミング技法の一種で、ロジックを直接コーディングするのではなく、あるパターンをもったロジックを生成する高位ロジックによってプログラミングを行う方法、またその高位ロジックを定義する方法のこと。主に対象言語に埋め込まれたマクロ言語によって行われる。",
                  "メタ|プログラミング|(|metaprogramming|)|と|は|プログラミング|技法|の|一種|で|、|ロジック|を|直接|コーディング|する|の|で|は|なく|、|ある|パターン|を|もっ|た|ロジック|を|生成|する|高位|ロジック|に|よっ|て|プログラミング|を|行う|方法|、|また|その|高位|ロジック|を|定義|する|方法|の|こと|。|主に|対象|言語|に|埋め込ま|れ|た|マクロ|言語|に|よっ|て|行わ|れる|。"
                );
            }
        }
    }
}

