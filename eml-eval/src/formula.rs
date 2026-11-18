//! Pure E-formulas: straight-line programs over E(x, y) = exp(x) − ln(y).

use std::{error::Error, f64::consts::PI, fmt, str::FromStr};

use num_complex::Complex64;

/// E(x, y) = exp(x) − ln(y) on the principal branch.
///
/// `num-complex` follows the C99 special values the formulas rely on:
/// ln 0 = −∞, exp(−∞) = 0, exp(+∞) = +∞, and signed zeros
/// (ln(−0 − 0i) = −∞ − iπ), as NumPy's complex128 does.
#[inline]
pub fn eml(x: Complex64, y: Complex64) -> Complex64 {
    exp(x) - ln(y)
}

// Shortcuts for real arguments, the common case in compiled formulas: they
// give the same values as num-complex, signed zeros included, with fewer
// calls. One deliberate difference: when e^x overflows, the imaginary part
// stays ±0 (as in NumPy) instead of num-complex's ∞·0 = NaN.

/// e^z; for real z one real exp instead of exp, cos and sin.
#[inline]
fn exp(z: Complex64) -> Complex64 {
    if z.im == 0.0 {
        Complex64::new(z.re.exp(), z.im)
    } else {
        z.exp()
    }
}

/// ln z; for real z, ln|z| and an argument of ±0 (z ≥ +0) or ±π (z ≤ −0),
/// instead of hypot, ln and atan2.
#[inline]
fn ln(z: Complex64) -> Complex64 {
    if z.im == 0.0 && !z.re.is_nan() {
        let arg = if z.re.is_sign_positive() {
            z.im
        } else {
            PI.copysign(z.im)
        };
        Complex64::new(z.re.abs().ln(), arg)
    } else {
        z.ln()
    }
}

/// A pure E-formula, prepared for fast repeated evaluation.
///
/// Everything that does not depend on the inputs is computed once, when the
/// formula is read: E's built only from 1 (the weights) become constants,
/// and so does the exp or ln half of an E whose operand is such a constant.
/// What is left is a straight-line program over slots: slots 0..n_inputs
/// hold the inputs, and each remaining E appends one slot.
#[derive(Debug, Clone, PartialEq)]
pub struct Formula {
    n_inputs: usize,
    n_es: usize,
    ops: Vec<Op>,
    outputs: Vec<Value>,
}

/// One E that depends on the inputs: exp(x) − ln(y), each half either
/// computed from a slot or, for a constant operand, known in advance.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Op {
    exp_x: Value,
    ln_y: Value,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    Slot(u32),
    Known(Complex64),
}

impl Value {
    fn map(self, f: fn(Complex64) -> Complex64) -> Self {
        match self {
            Value::Known(c) => Value::Known(f(c)),
            slot => slot,
        }
    }
}

impl Formula {
    /// Number of E's in the formula as written.
    pub fn len(&self) -> usize {
        self.n_es
    }

    pub fn is_empty(&self) -> bool {
        self.n_es == 0
    }

    /// Number of E's evaluated per input; the others are constants.
    pub fn input_dependent(&self) -> usize {
        self.ops.len()
    }

    /// The highest input the formula reads (p1..pN); callers pass at least N.
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Evaluates the formula and returns the real part of every output.
    ///
    /// `slots` is scratch space: reuse one vector across calls to avoid
    /// allocating for every evaluation.
    pub fn eval_with(&self, inputs: &[f64], slots: &mut Vec<Complex64>) -> Vec<f64> {
        assert!(
            inputs.len() >= self.n_inputs,
            "the formula reads {} inputs",
            self.n_inputs
        );
        slots.clear();
        slots.reserve(self.n_inputs + self.ops.len());
        slots.extend(inputs[..self.n_inputs].iter().map(|&x| Complex64::from(x)));
        for op in &self.ops {
            let exp_x = match op.exp_x {
                Value::Slot(s) => exp(slots[s as usize]),
                Value::Known(c) => c,
            };
            let ln_y = match op.ln_y {
                Value::Slot(s) => ln(slots[s as usize]),
                Value::Known(c) => c,
            };
            slots.push(exp_x - ln_y);
        }
        let read = |v: &Value| match *v {
            Value::Slot(s) => slots[s as usize].re,
            Value::Known(c) => c.re,
        };
        self.outputs.iter().map(read).collect()
    }

    pub fn eval(&self, inputs: &[f64]) -> Vec<f64> {
        self.eval_with(inputs, &mut Vec::new())
    }
}

/// A formula file that could not be read, with the line at fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl Error for ParseError {}

/// Reads the text format written by the Python side:
///
/// ```text
/// # comments
/// e1 = E(1, p5)
/// e2 = E(e1, 1)
/// score_0 = e2
/// ```
impl FromStr for Formula {
    type Err = ParseError;

    fn from_str(text: &str) -> Result<Self, ParseError> {
        let mut ops: Vec<[Operand; 2]> = Vec::new();
        let mut outputs: Vec<Operand> = Vec::new();

        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let err = |message: String| ParseError {
                line: index + 1,
                message,
            };
            let operand = |s: &str, defined: usize| {
                Operand::parse(s.trim(), defined)
                    .ok_or_else(|| err(format!("unknown operand `{}`", s.trim())))
            };
            let (name, value) = line
                .split_once('=')
                .map(|(n, v)| (n.trim(), v.trim()))
                .ok_or_else(|| err("expected `name = value`".into()))?;

            if let Some(k) = name.strip_prefix("score_") {
                if k.parse() != Ok(outputs.len()) {
                    return Err(err(format!("expected score_{} next", outputs.len())));
                }
                outputs.push(operand(value, ops.len())?);
            } else if let Some(j) = name.strip_prefix('e') {
                if j.parse() != Ok(ops.len() + 1) {
                    return Err(err(format!("expected e{} next", ops.len() + 1)));
                }
                let (a, b) = value
                    .strip_prefix("E(")
                    .and_then(|v| v.strip_suffix(')'))
                    .and_then(|v| v.split_once(','))
                    .ok_or_else(|| err("expected `E(a, b)`".into()))?;
                ops.push([operand(a, ops.len())?, operand(b, ops.len())?]);
            } else {
                return Err(err(format!("unexpected name `{name}`")));
            }
        }

        Ok(Formula::prepare(&ops, &outputs))
    }
}

impl Formula {
    /// Computes the constant parts once and numbers the slots of the rest.
    fn prepare(written: &[[Operand; 2]], outputs: &[Operand]) -> Self {
        let n_inputs = written
            .iter()
            .flatten()
            .chain(outputs)
            .filter_map(|&r| match r {
                Operand::Input(i) => Some(i as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        let mut values: Vec<Value> = Vec::with_capacity(written.len()); // one per written E
        let mut ops = Vec::new();
        let value_of = |r: Operand, values: &[Value]| match r {
            Operand::One => Value::Known(Complex64::ONE),
            Operand::Input(i) => Value::Slot(i - 1),
            Operand::E(j) => values[j as usize - 1],
        };
        for &[a, b] in written {
            let value = match (value_of(a, &values), value_of(b, &values)) {
                (Value::Known(x), Value::Known(y)) => Value::Known(eml(x, y)),
                (x, y) => {
                    ops.push(Op {
                        exp_x: x.map(exp),
                        ln_y: y.map(ln),
                    });
                    Value::Slot((n_inputs + ops.len() - 1) as u32)
                }
            };
            values.push(value);
        }
        let outputs = outputs.iter().map(|&r| value_of(r, &values)).collect();
        Formula {
            n_inputs,
            n_es: written.len(),
            ops,
            outputs,
        }
    }
}

/// An operand as written in the file: `1`, an input `pK`, or an earlier `eK`.
#[derive(Debug, Clone, Copy)]
enum Operand {
    One,
    Input(u32),
    E(u32),
}

impl Operand {
    /// `defined` is how many E's exist so far; only those may be used.
    fn parse(s: &str, defined: usize) -> Option<Self> {
        if s == "1" {
            return Some(Operand::One);
        }
        let (tag, number) = s.split_at_checked(1)?;
        let k: u32 = number.parse().ok().filter(|&k| k >= 1)?;
        match tag {
            "p" => Some(Operand::Input(k)),
            "e" if k as usize <= defined => Some(Operand::E(k)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::E;

    use super::*;

    fn formula(text: &str) -> Formula {
        text.parse().unwrap()
    }

    #[test]
    fn e_of_one_and_one_is_e() {
        assert_eq!(formula("e1 = E(1, 1)\nscore_0 = e1").eval(&[]), [E]);
    }

    /// ln x = E(1, E(E(1, x), 1)), the identity the compiler uses everywhere.
    const LN: &str = "e1 = E(1, p1)\ne2 = E(e1, 1)\ne3 = E(1, e2)\nscore_0 = e3";

    #[test]
    fn ln_identity() {
        let ln = formula(LN);
        assert!((ln.eval(&[2.0])[0] - 2f64.ln()).abs() < 1e-15);
        assert_eq!(ln.eval(&[0.0]), [f64::NEG_INFINITY], "ln 0 must be −∞");
    }

    #[test]
    fn ln_of_a_negative_number_is_complex() {
        // exp(ln x) = x for x = −2: the iπ in ln(−2) must survive.
        let exp_ln =
            formula("e1 = E(1, p1)\ne2 = E(e1, 1)\ne3 = E(1, e2)\ne4 = E(e3, 1)\nscore_0 = e4");
        assert!((exp_ln.eval(&[-2.0])[0] + 2.0).abs() < 1e-14);
    }

    #[test]
    fn signed_zeros_follow_c99() {
        // exp(−∞ − iπ) = −0 − 0i and ln(−0 − 0i) = −∞ − iπ.
        let z = eml(Complex64::new(f64::NEG_INFINITY, -PI), Complex64::ONE);
        assert!(
            z.re == 0.0 && z.re.is_sign_negative() && z.im.is_sign_negative(),
            "{z}"
        );
        assert_eq!(eml(Complex64::ONE, z), Complex64::new(f64::INFINITY, PI));
    }

    #[test]
    fn real_shortcuts_match_complex_maths() {
        let reals = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -3.0,
            700.0,
            710.0,
            -800.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let bits = |z: Complex64| (z.re.to_bits(), z.im.to_bits());
        for re in reals {
            for im in [0.0, -0.0] {
                let z = Complex64::new(re, im);
                assert_eq!(bits(ln(z)), bits(z.ln()), "ln {z}");
                assert_eq!(exp(z).re.to_bits(), z.exp().re.to_bits(), "exp {z}");
                if re != 710.0 {
                    assert_eq!(exp(z).im.to_bits(), z.exp().im.to_bits(), "exp {z}");
                } else {
                    // the one allowed difference: e^710 overflows; ±0, not ∞·0 = NaN
                    assert!(exp(z).im == 0.0 && z.exp().im.is_nan());
                }
            }
        }
    }

    #[test]
    fn constants_are_computed_once() {
        // e1 and e2 use no input; of e3 = exp(e2) − ln(p1) only ln(p1) is left per input.
        let f = formula("e1 = E(1, 1)\ne2 = E(e1, 1)\ne3 = E(e2, p1)\nscore_0 = e3\nscore_1 = e1");
        assert_eq!((f.len(), f.input_dependent()), (3, 1));
        assert_eq!(f.eval(&[1.0]), [E.exp().exp(), E]);
    }

    #[test]
    fn inputs_and_slots() {
        let f = formula("e1 = E(p2, 1)\nscore_0 = e1\nscore_1 = p1");
        assert_eq!((f.len(), f.n_inputs(), f.n_outputs()), (1, 2, 2));
        assert_eq!(f.eval(&[7.0, 0.0, 99.0]), [1.0, 7.0]);
    }

    #[test]
    fn rejects_bad_files() {
        let line = |text: &str| text.parse::<Formula>().unwrap_err().line;
        assert_eq!(line("e1 = E(1, e1)"), 1, "an E cannot use itself");
        assert_eq!(
            line("e1 = E(1, 1)\ne3 = E(e1, 1)"),
            2,
            "E's are numbered in order"
        );
        assert_eq!(
            line("# note\n\nscore_1 = 1"),
            3,
            "scores are numbered in order"
        );
        assert_eq!(line("e1 = E(1, q2)"), 1);
        assert_eq!(line("e1 = exp(1)"), 1);
    }
}
