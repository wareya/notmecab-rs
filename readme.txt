notmecab-rs is a very basic mecab clone, designed only to do parsing, not training.

This is meant to be used as a library by other tools such as frequency analyzers. Not directly by people.

It also only works with UTF-8 dictionaries. (Stop using encodings other than UTF-8 for infrastructural software.)

Licensed under the Apache License, Version 2.0.

Example (from tests, edited):

    let mut sysdic_raw = File::open("data/sys.dic").unwrap();
    let mut sysdic = BufReader::new(sysdic_raw);
    
    let mut matrix_raw = File::open("data/matrix.bin").unwrap();
    let mut matrix = BufReader::new(matrix_raw);
    
    let mut dict = Dict::load(&mut sysdic, &mut matrix).unwrap();
    
    let result = parse(&dict, &"これを持っていけ".to_string());
    
    if let Some(result) = result
    {
        for token in &result.0
        {
            println!("{}", token.feature);
        }
        let mut first = true;
        for token in &result.0
        {
            if !first
            {
                print!("｜");
            }
            print!("{}", token.surface);
            first = false;
        }
        println!();
    }

Output of example:

    代名詞,*,*,*,*,*,コレ,此れ,これ,コレ,これ,コレ,和,*,*,*,*,*,*,体,コレ,コレ,コレ,コレ,0,*,*,3599534815060480,13095
    助詞,格助詞,*,*,*,*,ヲ,を,を,オ,を,オ,和,*,*,*,*,*,*,格助,ヲ,ヲ,ヲ,ヲ,*,"動詞%F2@0,名詞%F1,形容詞%F2@-1",*,11381878116459008,41407
    動詞,一般,*,*,五段-タ行,連用形-促音便,モツ,持つ,持っ,モッ,持つ,モツ,和,*,*,*,*,*,*,用,モッ,モツ,モッ,モツ,1,C1,*,10391493084848772,37804
    助詞,接続助詞,*,*,*,*,テ,て,て,テ,て,テ,和,*,*,*,*,*,*,接助,テ,テ,テ,テ,*,"動詞%F1,形容詞%F2@-1",*,6837321680953856,24874
    動詞,非自立可能,*,*,五段-カ行,命令形,イク,行く,いけ,イケ,いく,イク,和,*,*,*,*,*,*,用,イケ,イク,イケ,イク,0,C2,*,470874478224161,1713
    これ｜を｜持っ｜て｜いけ

You can also call parse_to_lexertoken, which doesn't do any string copying or allocations, but you don't get the feature string.

Testing:

Get unidic's sys.dic and matrix.bin and put them under a new folder next to src/ called data/. Then invoke tests from the repository root.

NOTE: This software is unusably slow if optimizations are disabled.

TODO:

- implement unk.dic and its right/left context IDs
