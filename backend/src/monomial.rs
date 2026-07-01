use std::cmp;

#[inline]
pub fn factorial(n: usize) -> f64 {
    (1..=n).map(|v| v as f64).product()
}

#[inline]
pub fn math_comb(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = cmp::min(k, n - k);
    let mut res = 1;
    for i in 1..=k {
        res = res * (n - i + 1) / i;
    }
    res
}

pub fn unrank_tuple(
    m_id: usize,
    n: usize,
    d: usize,
    offsets: &[usize],
    binom_2: &[usize],
    binom_3: &[usize],
    is_squarefree: bool,
) -> Vec<usize> {
    let mut k = 0;
    for i in 0..=d {
        if offsets[i] <= m_id && m_id < offsets[i + 1] {
            k = i;
            break;
        }
    }

    if k == 0 {
        return vec![];
    }

    let rank = m_id - offsets[k];

    if k == 1 {
        return vec![rank];
    } else if k == 2 {
        if is_squarefree {
            // Combinadic: rank = comb(c2, 2) + c1, c1 < c2
            let mut c2 = ((rank as f64) * 2.0).sqrt() as usize;
            while binom_2[c2] > rank {
                c2 -= 1;
            }
            while binom_2[c2 + 1] <= rank {
                c2 += 1;
            }
            let c1 = rank - binom_2[c2];
            return vec![c1, c2];
        } else {
            // With-repetition: rank = comb(c2+1, 2) + c1, c1 <= c2
            let mut c2 = ((rank as f64) * 2.0).sqrt() as usize;
            while binom_2[c2 + 1] > rank {
                c2 -= 1;
            }
            while binom_2[c2 + 2] <= rank {
                c2 += 1;
            }
            let c1 = rank - binom_2[c2 + 1];
            return vec![c1, c2];
        }
    } else if k == 3 {
        let mut c3 = ((rank as f64) * 6.0).powf(1.0 / 3.0) as usize;
        while binom_3[c3] > rank {
            c3 -= 1;
        }
        while binom_3[c3 + 1] <= rank {
            c3 += 1;
        }
        let rem = rank - binom_3[c3];

        let mut c2 = ((rem as f64) * 2.0).sqrt() as usize;
        while binom_2[c2] > rem {
            c2 -= 1;
        }
        while binom_2[c2 + 1] <= rem {
            c2 += 1;
        }
        let c1 = rem - binom_2[c2];

        return vec![c1, c2, c3];
    }

    let mut c = Vec::new();
    let mut rem = rank;
    for i in (1..=k).rev() {
        let mut guess = if rem == 0 {
            i - 1
        } else {
            ((rem as f64) * factorial(i)).powf(1.0 / (i as f64)) as usize
        };
        while math_comb(guess, i) > rem {
            guess -= 1;
        }
        while math_comb(guess + 1, i) <= rem {
            guess += 1;
        }
        c.push(guess);
        rem -= math_comb(guess, i);
    }

    let mut res = vec![0; k];
    for j in 0..k {
        res[j] = c[k - 1 - j];
    }
    res
}

pub fn rank_tuple(
    tup: &[usize],
    n: usize,
    d: usize,
    offsets: &[usize],
    binom_2: &[usize],
    binom_3: &[usize],
    is_squarefree: bool,
) -> usize {
    let k = tup.len();
    if k == 0 {
        return 0;
    }

    let mut rank = 0;
    if k == 1 {
        rank = tup[0];
    } else if k == 2 {
        // Both squarefree and non-squarefree use the generic formula:
        // rank = sum(comb(tup[i] + offset, i+1) for i in range(k))
        // For squarefree: rank = comb(tup[0], 1) + comb(tup[1], 2) = tup[0] + binom_2[tup[1]]
        // For non-squarefree: rank = comb(tup[0], 1) + comb(tup[1]+1, 2) = tup[0] + binom_2[tup[1]+1]
        if is_squarefree {
            rank = tup[0] + binom_2[tup[1]];
        } else {
            rank = tup[0] + binom_2[tup[1] + 1];
        }
    } else if k == 3 {
        if is_squarefree {
            rank = binom_3[tup[2]] + binom_2[tup[1]] + tup[0];
        } else {
            rank = binom_3[tup[2] + 2] + binom_2[tup[1] + 1] + tup[0];
        }
    } else {
        for i in 0..k {
            if is_squarefree {
                rank += math_comb(tup[i], i + 1);
            } else {
                rank += math_comb(tup[i] + i, i + 1);
            }
        }
    }

    offsets[k] + rank
}

pub fn apply_permutation(m_tuple: &[usize], p: &[usize], is_squarefree: bool) -> Vec<usize> {
    let mut mapped = Vec::with_capacity(m_tuple.len());
    for &val in m_tuple {
        mapped.push(p[val]);
    }
    mapped.sort_unstable();
    if !is_squarefree {
        let mut j = 0;
        for i in 1..mapped.len() {
            if mapped[i] != mapped[j] {
                j += 1;
                mapped[j] = mapped[i];
            }
        }
        mapped.truncate(j + 1);
    }
    mapped
}

pub fn apply_permutation_arr(
    m_id: usize,
    inv_data: &[usize],
    n: usize,
    d: usize,
    offsets: &[usize],
    binom_2: &[usize],
    binom_3: &[usize],
    is_squarefree: bool,
) -> usize {
    if d >= 1 && m_id < offsets[2] {
        if m_id < offsets[1] {
            return 0;
        }
        let v = m_id - offsets[1];
        return offsets[1] + inv_data[v];
    } else if d >= 2 && m_id < offsets[3] && !is_squarefree {
        let rank = m_id - offsets[2];
        // Unrank: rank = binom_2[c2+1] + c1 where c1 <= c2
        let mut c2 = ((rank as f64) * 2.0).sqrt() as usize;
        while binom_2[c2 + 1] > rank {
            c2 -= 1;
        }
        while binom_2[c2 + 2] <= rank {
            c2 += 1;
        }
        let c1 = rank - binom_2[c2 + 1];
        let mut v0 = inv_data[c1];
        let mut v1 = inv_data[c2];
        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }
        return offsets[2] + binom_2[v1 + 1] + v0;
    } else if d >= 2 && m_id < offsets[3] && is_squarefree {
        let rank = m_id - offsets[2];
        let mut c2 = ((rank as f64) * 2.0).sqrt() as usize;
        while binom_2[c2] > rank {
            c2 -= 1;
        }
        while binom_2[c2 + 1] <= rank {
            c2 += 1;
        }
        let c1 = rank - binom_2[c2];
        let mut v0 = inv_data[c1];
        let mut v1 = inv_data[c2];
        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }
        return offsets[2] + binom_2[v1] + v0;
    } else if d >= 3 && m_id < offsets[4] && is_squarefree {
        let rank = m_id - offsets[3];
        let mut c3 = ((rank as f64) * 6.0).powf(1.0 / 3.0) as usize;
        while binom_3[c3] > rank {
            c3 -= 1;
        }
        while binom_3[c3 + 1] <= rank {
            c3 += 1;
        }
        let rem = rank - binom_3[c3];

        let mut c2 = ((rem as f64) * 2.0).sqrt() as usize;
        while binom_2[c2] > rem {
            c2 -= 1;
        }
        while binom_2[c2 + 1] <= rem {
            c2 += 1;
        }
        let c1 = rem - binom_2[c2];

        let mut v0 = inv_data[c1];
        let mut v1 = inv_data[c2];
        let mut v2 = inv_data[c3];

        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }
        if v1 > v2 {
            std::mem::swap(&mut v1, &mut v2);
        }
        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }

        return offsets[3] + binom_3[v2] + binom_2[v1] + v0;
    } else {
        let tup = unrank_tuple(m_id, n, d, offsets, binom_2, binom_3, is_squarefree);
        let mapped = apply_permutation(&tup, inv_data, is_squarefree);
        return rank_tuple(&mapped, n, d, offsets, binom_2, binom_3, is_squarefree);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_unrank_tuple() {
        let n = 4;
        let d = 2;
        let is_squarefree = true;

        let mut offsets = vec![0; d + 2];
        let mut total = 0;
        for k in 0..=d {
            let count = if k == 0 {
                1
            } else {
                math_comb(n + k - 1 - (if is_squarefree { k - 1 } else { 0 }), k)
            };
            total += count;
            offsets[k + 1] = total;
        }

        let binom_size = n + 3;
        let mut binom_2 = Vec::with_capacity(binom_size);
        let mut binom_3 = Vec::with_capacity(binom_size);
        for i in 0..binom_size {
            binom_2.push(math_comb(i, 2));
            binom_3.push(math_comb(i, 3));
        }

        // Test ranking and unranking for squarefree tuples
        for i in 0..offsets[d + 1] {
            let tup = unrank_tuple(i, n, d, &offsets, &binom_2, &binom_3, is_squarefree);
            let ranked = rank_tuple(&tup, n, d, &offsets, &binom_2, &binom_3, is_squarefree);
            assert_eq!(i, ranked, "Failed on id {} -> {:?} -> {}", i, tup, ranked);
        }
    }
}
