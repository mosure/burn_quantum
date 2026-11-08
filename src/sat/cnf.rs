//! Normalized Boolean formulas, DIMACS parsing, and checked serialization.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct CnfData {
    variables: usize,
    clauses: Vec<Vec<i32>>,
}

/// Conjunction of disjunctions, using signed, one-based DIMACS variable numbers.
/// Duplicate literals/clauses and tautological clauses are removed. Empty clauses
/// are false; no clauses is true. Assignment element zero corresponds to variable 1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "CnfData", into = "CnfData")]
pub struct Cnf {
    variables: usize,
    clauses: Vec<Vec<i32>>,
}

impl Cnf {
    /// Validate and normalize a CNF. Zero is a terminator only in DIMACS text,
    /// never a literal in this constructor. Unused declared variables are retained.
    pub fn new(variables: usize, clauses: Vec<Vec<i32>>) -> Result<Self> {
        if variables > i32::MAX as usize {
            return Err(Error::InvalidValue(
                "CNF variable count exceeds DIMACS i32 range".into(),
            ));
        }
        let mut normalized = Vec::new();
        for mut clause in clauses {
            if clause
                .iter()
                .any(|&x| x == 0 || x.unsigned_abs() as usize > variables)
            {
                return Err(Error::InvalidValue(
                    "CNF literal outside declared variables".into(),
                ));
            }
            clause.sort_unstable_by_key(|x| (x.unsigned_abs(), *x));
            clause.dedup();
            if clause
                .windows(2)
                .any(|w| w[0].unsigned_abs() == w[1].unsigned_abs())
            {
                continue;
            }
            normalized.push(clause);
        }
        normalized.sort_unstable();
        normalized.dedup();
        Ok(Self {
            variables,
            clauses: normalized,
        })
    }

    /// Parse DIMACS `p cnf <variables> <clauses>`, with `c` comment lines and
    /// zero-terminated clauses that may span lines. Counts are checked before
    /// normalization. Missing headers/terminators and out-of-range literals fail.
    pub fn from_dimacs(text: &str) -> Result<Self> {
        let invalid = |reason: &str| Error::InvalidValue(format!("DIMACS: {reason}"));
        let mut header = None;
        let mut clauses = Vec::new();
        let mut clause = Vec::new();
        for line in text.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('c') {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields[0] == "p" {
                if header.is_some() || fields.len() != 4 || fields[1] != "cnf" {
                    return Err(invalid("expected exactly one p cnf header"));
                }
                let variables = fields[2]
                    .parse::<usize>()
                    .map_err(|_| invalid("invalid variable count"))?;
                let count = fields[3]
                    .parse::<usize>()
                    .map_err(|_| invalid("invalid clause count"))?;
                if variables > i32::MAX as usize {
                    return Err(invalid("too many variables"));
                }
                header = Some((variables, count));
                continue;
            }
            let (variables, count) = header.ok_or_else(|| invalid("clauses precede header"))?;
            for field in fields {
                let literal = field
                    .parse::<i32>()
                    .map_err(|_| invalid("invalid literal"))?;
                if clauses.len() >= count {
                    return Err(invalid("too many clauses"));
                }
                if literal == 0 {
                    clauses.push(std::mem::take(&mut clause));
                } else {
                    if literal.unsigned_abs() as usize > variables {
                        return Err(invalid("literal outside declared variables"));
                    }
                    clause.push(literal);
                }
            }
        }
        let (variables, count) = header.ok_or_else(|| invalid("missing header"))?;
        if !clause.is_empty() {
            return Err(invalid("missing clause terminator"));
        }
        if clauses.len() != count {
            return Err(invalid("clause count mismatch"));
        }
        Self::new(variables, clauses)
    }

    /// Number of declared Boolean variables, including unused variables.
    pub fn num_variables(&self) -> usize {
        self.variables
    }
    /// Normalized clauses; an empty clause is an immediate contradiction.
    pub fn clauses(&self) -> &[Vec<i32>] {
        &self.clauses
    }
    /// Classically verify one candidate assignment in variable-number order.
    pub fn evaluate(&self, assignment: &[bool]) -> Result<bool> {
        if assignment.len() != self.variables {
            return Err(Error::Shape(
                "assignment length differs from CNF variable count".into(),
            ));
        }
        Ok(self.clauses.iter().all(|clause| {
            clause
                .iter()
                .any(|&x| assignment[x.unsigned_abs() as usize - 1] == (x > 0))
        }))
    }
}
impl TryFrom<CnfData> for Cnf {
    type Error = Error;
    fn try_from(data: CnfData) -> Result<Self> {
        Self::new(data.variables, data.clauses)
    }
}
impl From<Cnf> for CnfData {
    fn from(cnf: Cnf) -> Self {
        Self {
            variables: cnf.variables,
            clauses: cnf.clauses,
        }
    }
}
