notmecab-rs is a very basic mecab clone, designed only to do parsing, not training.

notmecab-rs loads everything into memory, so it has higher memory requirements than mecab, which uses memory mapping for most things.

This is meant to be used as a library by other tools such as frequency analyzers. Not directly by people.
It also only works with UTF-8 dictionaries. (Stop using encodings other than UTF-8 for infrastructural software.)

Licensed under the Apache License, Version 2.0.

# Testing

Get unidic's sys.dic, matrix.bin, unk.dic, and char.bin and put them in data/. Then invoke tests from the repository root.

# Performance

notmecab performs significantly worse than mecab, but there are many cases where mecab fails to find the lowest-cost string of tokens, so I'm pretty sure that it's just cutting corners somewhere performance sensitive in its code.

There are a couple difficult-to-use caching features designed to improve performance. By default, matrix costs will be cached in a hashmap. You can upload a matrix of connections between the most common connection edge types with ```prepare_fast_matrix_cache```, which is for extremely large dictionaries like modern versions of unidic, or you can load the entire matrix connection cache into memory with ```prepare_full_matrix_cache```, which is for small dictionaries like ipadic. Note that ```prepare_full_matrix_cache``` is actually slower than ```prepare_fast_matrix_cache``` for modern versions of unidic after long periods of pumping text through notmecab, though obviously ```prepare_full_matrix_cache``` is the best option for small dictionaries.

There are no stability guarantees about the presence or behavior of ```prepare_fast_matrix_cache```, because it's very hacky and if I find a better way to do what it's doing then I'm going to remove it.

# Example (from tests)

    // you need to acquire a mecab dictionary and place these files here manually
    let sysdic = BufReader::new(File::open("data/sys.dic").unwrap());
    let unkdic = BufReader::new(File::open("data/unk.dic").unwrap());
    let matrix = BufReader::new(File::open("data/matrix.bin").unwrap());
    let unkdef = BufReader::new(File::open("data/char.bin").unwrap());
    
    let dict = Dict::load(sysdic, unkdic, matrix, unkdef).unwrap();

    let result = parse(&dict, &"これを持っていけ".to_string()).unwrap();

    for token in &result.0
    {
        println!("{}", token.feature);
    }
    let split_up_string = tokenstream_to_string(&result.0, "|");
    println!("{}", split_up_string);
    assert_eq!(split_up_string, "これ|を|持っ|て|いけ"); // this test might fail if you're not testing with unidic (i.e. the correct parse might be different)

## Output of example

    代名詞,*,*,*,*,*,コレ,此れ,これ,コレ,これ,コレ,和,*,*,*,*,*,*,体,コレ,コレ,コレ,コレ,0,*,*,3599534815060480,13095
    助詞,格助詞,*,*,*,*,ヲ,を,を,オ,を,オ,和,*,*,*,*,*,*,格助,ヲ,ヲ,ヲ,ヲ,*,"動詞%F2@0,名詞%F1,形容詞%F2@-1",*,11381878116459008,41407
    動詞,一般,*,*,五段-タ行,連用形-促音便,モツ,持つ,持っ,モッ,持つ,モツ,和,*,*,*,*,*,*,用,モッ,モツ,モッ,モツ,1,C1,*,10391493084848772,37804
    助詞,接続助詞,*,*,*,*,テ,て,て,テ,て,テ,和,*,*,*,*,*,*,接助,テ,テ,テ,テ,*,"動詞%F1,形容詞%F2@-1",*,6837321680953856,24874
    動詞,非自立可能,*,*,五段-カ行,命令形,イク,行く,いけ,イケ,いく,イク,和,*,*,*,*,*,*,用,イケ,イク,イケ,イク,0,C2,*,470874478224161,1713
    これ|を|持っ|て|いけ

You can also call parse_to_lexertoken, which does less string allocation, but you don't get the feature string as a string.

# Notes

- This software is unusably slow if optimizations are disabled.
- Cost rewriting is not performed when user dictionaries are loaded.
- There are some cases where multiple parses tie for the lowest cost. It's not defined which parse gets chosen in these cases.
- There are some cases where mecab failed to find an ideal parse, but notmecab-rs does. Notmecab-rs should never produce a parse that has a higher total cost than the parse that mecab gives. If it does, it indicates some underlying bug, and should be reported, please.
