use std::ops::Range;

pub struct Cache
{
    rank_to_range : Vec<Range<u32>>,
    cost_for_node : Vec<Cost>,
    source_node : Vec<u32>,
    path : Vec<u32>
}

impl Cache
{
    pub fn new() -> Self
    {
        Cache
        {
            rank_to_range : Vec::new(),
            cost_for_node : Vec::new(),
            source_node : Vec::new(),
            path : Vec::new()
        }
    }

    fn clear(&mut self)
    {
        self.rank_to_range.clear();
        self.cost_for_node.clear();
        self.source_node.clear();
        self.path.clear();
    }
}

pub type Cost = i64;
const COST_MAX: Cost = std::i64::MAX;

pub fn shortest_path(
    cache: &mut Cache,
    node_count: usize,
    get_rank: impl Fn(usize) -> u32,
    get_next_rank: impl Fn(usize) -> u32,
    get_cost: impl Fn(usize, usize) -> Cost,
    get_cost_for_start_node: impl Fn(usize) -> Cost,
    get_cost_for_end_node: impl Fn(usize) -> Cost
) -> (&[u32], Cost) {
    if node_count == 0 {
        return (&[], 0);
    }

    cache.clear();

    debug_assert!((0..node_count).zip(1..node_count).all(|(index, next_index)| get_rank(next_index) >= get_rank(index)));
    debug_assert!((0..node_count).all(|node| get_next_rank(node) > get_rank(node)));

    let min_rank = get_rank(0);
    let max_rank = get_rank(node_count - 1);
    let mut end_rank = 0;
    let rank_to_range = &mut cache.rank_to_range;
    rank_to_range.resize((max_rank + 1) as usize, 0..0);

    {
        let mut previous_rank = min_rank;
        for index in 0..node_count
        {
            let rank = get_rank(index);
            if rank != previous_rank
            {
                let index = index as u32;
                rank_to_range[rank as usize].start = index;

                let previous_rank_range = &mut rank_to_range[previous_rank as usize];
                previous_rank_range.end = index;
                previous_rank = rank;
            }

            end_rank = std::cmp::max(end_rank, get_next_rank(index));
        }

        rank_to_range[previous_rank as usize].end = node_count as u32;
    }

    debug_assert!(rank_to_range.iter().all(|range| range.start < node_count as u32));
    debug_assert!(rank_to_range.iter().all(|range| range.end >= range.start as u32));
    debug_assert!(rank_to_range.iter().filter(|range| range.end - range.start > 0).collect::<Vec<_>>().windows(2).all(|window| window[1].start >= window[0].start));
    debug_assert_eq!(rank_to_range.iter().map(|range| range.end - range.start).sum::<u32>(), node_count as u32);
    debug_assert_eq!(rank_to_range[min_rank as usize].start, 0);
    debug_assert_eq!(rank_to_range[max_rank as usize].end, node_count as u32);
    debug_assert!(end_rank > max_rank);

    let cost_for_node = &mut cache.cost_for_node;
    cost_for_node.resize(node_count, COST_MAX);

    for index in rank_to_range[min_rank as usize].clone()
    {
        cost_for_node[index as usize] = get_cost_for_start_node(index as usize);
        debug_assert_ne!(cost_for_node[index as usize], COST_MAX);
    }

    let source_node = &mut cache.source_node;
    source_node.resize(node_count, std::u32::MAX);

    let mut lowest_cost_index = 0;
    let mut lowest_cost = COST_MAX;

    let mut starting_index = 0;
    while (starting_index as usize) < node_count
    {
        let current_rank = get_rank(starting_index as usize);
        let range = rank_to_range[current_rank as usize].clone();
        for index in range.clone()
        {
            let current_node_cost = cost_for_node[index as usize];
            if current_node_cost == COST_MAX
            {
                // Node is not connected to the start of the graph.
                debug_assert!(range.clone().all(|index| cost_for_node[index as usize] == COST_MAX));
                break;
            }

            let next_rank = get_next_rank(index as usize);
            if next_rank > max_rank
            {
                if next_rank == end_rank
                {
                    let new_cost = get_cost_for_end_node(index as usize) + current_node_cost;
                    if new_cost < lowest_cost
                    {
                        lowest_cost = new_cost;
                        lowest_cost_index = index;
                    }
                }
                continue;
            }

            let next_range = rank_to_range[next_rank as usize].clone();
            for next_index in next_range
            {
                let new_cost = get_cost(index as usize, next_index as usize) + current_node_cost;
                let old_cost = cost_for_node[next_index as usize];
                if new_cost < old_cost
                {
                    cost_for_node[next_index as usize] = new_cost;
                    source_node[next_index as usize] = index;
                }
            }
        }

        starting_index = range.end;
    }

    let path = &mut cache.path;
    let mut index = lowest_cost_index;
    let total_cost = cost_for_node[index as usize] + get_cost_for_end_node(index as usize);
    loop
    {
        path.push(index);
        match source_node[index as usize]
        {
            std::u32::MAX => break,
            source_index => index = source_index
        }
    }

    path.reverse();
    (&cache.path, total_cost)
}

#[test]
fn test_shortest_path()
{
    let mut cache = Cache::new();
    let (path, total_cost) = shortest_path(
        &mut cache,
        0,
        |_| unreachable!(),
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_| unreachable!(),
        |_| unreachable!()
    );
    assert_eq!(path, &[]);
    assert_eq!(total_cost, 0);

    let (path, total_cost) = shortest_path(
        &mut cache,
        1,
        |_| 0,
        |_| 1,
        |_, _| unreachable!(),
        |_| 0,
        |_| 0
    );
    assert_eq!(path, &[0]);
    assert_eq!(total_cost, 0);

    let (path, total_cost) = shortest_path(
        &mut cache,
        2,
        |index| match index {
            0 => 0,
            1 => 0,
            _ => unreachable!()
        },
        |index| match index {
            0 => 10,
            1 => 5,
            _ => unreachable!()
        },
        |_, _| unreachable!(),
        |index| match index {
            0 => 1000,
            1 => 0,
            _ => unreachable!()
        },
        |_| 0
    );
    assert_eq!(path, &[0]);
    assert_eq!(total_cost, 1000);

    let (path, total_cost) = shortest_path(
        &mut cache,
        2,
        |index| match index {
            0 => 0,
            1 => 1,
            _ => unreachable!()
        },
        |index| match index {
            0 => 1,
            1 => 2,
            _ => unreachable!()
        },
        |_, _| 1,
        |_| 0,
        |_| 0
    );
    assert_eq!(path, &[0, 1]);
    assert_eq!(total_cost, 1);

    let (path, total_cost) = shortest_path(
        &mut cache,
        5,
        |index| match index {
            0 | 1 => 0,
            2 | 3 => 1,
                4 => 2,
            _ => unreachable!()
        },
        |index| match index {
            0 | 1 => 1,
            2 | 3 => 2,
                4 => 3,
            _ => unreachable!()
        },
        |a, b| match (a, b) {
            (0, 2) => 100,
            (0, 3) => 0,
            (1, 2) => 100,
            (1, 3) => 100,
            (2, 4) => 0,
            (3, 4) => 10000,
            _ => unreachable!()
        },
        |_| 0,
        |_| 0
    );
    assert_eq!(path, &[0, 2, 4]);
    assert_eq!(total_cost, 100);

    let (path, total_cost) = shortest_path(
        &mut cache,
        5,
        |index| match index {
            0 => 0,
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            _ => unreachable!()
        },
        |index| match index {
            0 => 1,
            1 => 4,
            2 => 2,
            3 => 3,
            4 => 4,
            _ => unreachable!()
        },
        |a, b| match (a, b) {
            (0, 2) => 0,
            (2, 3) => 0,
            (3, 4) => 0,
            _ => unreachable!()
        },
        |_| 0,
        |_| 0
    );
    assert_eq!(path, &[1]);
    assert_eq!(total_cost, 0);

    let (path, total_cost) = shortest_path(
        &mut cache,
        5,
        |index| match index {
            0 => 0,
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            _ => unreachable!()
        },
        |index| match index {
            0 => 1,
            1 => 4,
            2 => 2,
            3 => 3,
            4 => 4,
            _ => unreachable!()
        },
        |a, b| match (a, b) {
            (0, 2) => 0,
            (2, 3) => 0,
            (3, 4) => 0,
            _ => unreachable!()
        },
        |index| match index {
            0 => 1,
            1 => 0,
            _ => unreachable!()
        },
        |_| 0
    );
    assert_eq!(path, &[1]);
    assert_eq!(total_cost, 0);

    let (path, total_cost) = shortest_path(
        &mut cache,
        5,
        |index| match index {
            0 => 0,
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            _ => unreachable!()
        },
        |index| match index {
            0 => 1,
            1 => 4,
            2 => 2,
            3 => 3,
            4 => 4,
            _ => unreachable!()
        },
        |a, b| match (a, b) {
            (0, 2) => 0,
            (2, 3) => 0,
            (3, 4) => 0,
            _ => unreachable!()
        },
        |index| match index {
            0 => 0,
            1 => 1,
            _ => unreachable!()
        },
        |_| 0
    );
    assert_eq!(path, &[0, 2, 3, 4]);
    assert_eq!(total_cost, 0);

    let (path, total_cost) = shortest_path(
        &mut cache,
        5,
        |index| match index {
            0 => 0,
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            _ => unreachable!()
        },
        |index| match index {
            0 => 1,
            1 => 4,
            2 => 2,
            3 => 3,
            4 => 4,
            _ => unreachable!()
        },
        |a, b| match (a, b) {
            (0, 2) => 0,
            (2, 3) => 0,
            (3, 4) => 0,
            _ => unreachable!()
        },
        |index| match index {
            0 => 0,
            1 => 1,
            _ => unreachable!()
        },
        |index| match index {
            1 => 0,
            4 => 2,
            _ => unreachable!()
        }
    );
    assert_eq!(path, &[1]);
    assert_eq!(total_cost, 1);
}
