//! Evaluate pure E-formulas, built only from E(x, y) = exp(x) − ln(y), the
//! constant 1 and the inputs.

pub mod formula;
pub mod mnist;

pub use formula::{Formula, ParseError, eml};
