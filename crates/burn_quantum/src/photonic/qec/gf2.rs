//! Shared bounded-shape GF(2) elimination. Callers validate allocation budgets.
//! Packed row XORs preserve the deterministic column/row pivot convention.
pub(super) fn solve(
    matrix: &[Vec<bool>],
    variables: usize,
    rhs: &[bool],
) -> Option<(Vec<bool>, Vec<Vec<bool>>)> {
    if matrix.len() != rhs.len() || matrix.iter().any(|row| row.len() != variables) {
        return None;
    }
    let words = variables.checked_add(1)?.div_ceil(64);
    let bit = |row: &[u64], column: usize| row[column / 64] >> (column % 64) & 1 != 0;
    let mut rows: Vec<Vec<u64>> = matrix
        .iter()
        .zip(rhs)
        .map(|(row, &right)| {
            let mut packed = vec![0; words];
            for (column, value) in row.iter().copied().chain([right]).enumerate() {
                if value {
                    packed[column / 64] |= 1 << (column % 64);
                }
            }
            packed
        })
        .collect();
    let mut pivots = vec![];
    let mut is_pivot = vec![false; variables];
    for col in 0..variables {
        let rank = pivots.len();
        if let Some(p) = (rank..rows.len()).find(|&r| bit(&rows[r], col)) {
            rows.swap(rank, p);
            let pivot = rows[rank].clone();
            for (r, row) in rows.iter_mut().enumerate() {
                if r != rank && bit(row, col) {
                    // Earlier pivot/free coordinates in this pivot row are zero.
                    // XORing the complete containing word therefore agrees with
                    // scalar elimination beginning exactly at `col`.
                    for (value, &other) in row[col / 64..].iter_mut().zip(&pivot[col / 64..]) {
                        *value ^= other;
                    }
                }
            }
            pivots.push(col);
            is_pivot[col] = true;
        }
    }
    // Every nonpivot row has a zero left side after elimination.
    if rows[pivots.len()..].iter().any(|row| bit(row, variables)) {
        return None;
    }
    let mut particular = vec![false; variables];
    for (r, &p) in pivots.iter().enumerate() {
        particular[p] = bit(&rows[r], variables);
    }
    let mut nullspace = vec![];
    for free in (0..variables).filter(|&c| !is_pivot[c]) {
        let mut v = vec![false; variables];
        v[free] = true;
        for (r, &p) in pivots.iter().enumerate() {
            v[p] = bit(&rows[r], free);
        }
        nullspace.push(v);
    }
    Some((particular, nullspace))
}

#[cfg(test)]
mod tests {
    use super::solve;
    // Independent scalar implementation retained as an exact output oracle.
    fn scalar(
        matrix: &[Vec<bool>],
        variables: usize,
        rhs: &[bool],
    ) -> Option<(Vec<bool>, Vec<Vec<bool>>)> {
        let mut rows: Vec<Vec<bool>> = matrix
            .iter()
            .zip(rhs)
            .map(|(row, &b)| row.iter().copied().chain([b]).collect())
            .collect();
        let mut pivots = vec![];
        for col in 0..variables {
            let rank = pivots.len();
            if let Some(p) = (rank..rows.len()).find(|&r| rows[r][col]) {
                rows.swap(rank, p);
                let pivot = rows[rank].clone();
                for (r, row) in rows.iter_mut().enumerate() {
                    if r != rank && row[col] {
                        for j in col..=variables {
                            row[j] ^= pivot[j];
                        }
                    }
                }
                pivots.push(col);
            }
        }
        if rows
            .iter()
            .any(|r| !r[..variables].iter().any(|&v| v) && r[variables])
        {
            return None;
        }
        let mut particular = vec![false; variables];
        for (r, &p) in pivots.iter().enumerate() {
            particular[p] = rows[r][variables];
        }
        let mut kernel = vec![];
        for f in (0..variables).filter(|c| !pivots.contains(c)) {
            let mut v = vec![false; variables];
            v[f] = true;
            for (r, &p) in pivots.iter().enumerate() {
                v[p] = rows[r][f];
            }
            kernel.push(v);
        }
        Some((particular, kernel))
    }
    #[test]
    fn packed_elimination_matches_every_small_binary_system() {
        for columns in 0..=4 {
            for rows in 0..=3 {
                for mask in 0usize..1usize << (columns * rows) {
                    let matrix: Vec<Vec<bool>> = (0..rows)
                        .map(|r| {
                            (0..columns)
                                .map(|c| mask >> (r * columns + c) & 1 != 0)
                                .collect()
                        })
                        .collect();
                    for bits in 0usize..1usize << rows {
                        let rhs: Vec<_> = (0..rows).map(|r| bits >> r & 1 != 0).collect();
                        assert_eq!(
                            solve(&matrix, columns, &rhs),
                            scalar(&matrix, columns, &rhs)
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn packed_elimination_preserves_pivots_and_rhs_at_word_boundaries() {
        for columns in [1, 63, 64, 65, 127, 128, 129, 257] {
            for rows in [0, 1, 17, 70] {
                let mut state = 107u64;
                let mut random = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state & 1 != 0
                };
                let matrix: Vec<Vec<bool>> = (0..rows)
                    .map(|_| (0..columns).map(|_| random()).collect())
                    .collect();
                let rhs: Vec<_> = (0..rows).map(|_| random()).collect();
                assert_eq!(
                    solve(&matrix, columns, &rhs),
                    scalar(&matrix, columns, &rhs)
                );
            }
        }
        assert!(solve(&[vec![true]], 2, &[false]).is_none());
        assert!(solve(&[vec![true]], 1, &[]).is_none());
    }
}
