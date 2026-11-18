//! Runs a pure E-formula on the 10,000 MNIST test digits and reports its
//! accuracy, whether it gives the same scores as NumPy, and its speed on one
//! core and on all cores.

use std::{
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use eml_eval::{Formula, mnist::Digits};
use rayon::prelude::*;
use serde_json::json;

#[derive(Parser)]
#[command(about)]
struct Args {
    /// Formula file: one `eN = E(a, b)` per line, then `score_k = ...`.
    formula: PathBuf,
    /// Folder holding t10k-images-idx3-ubyte.gz and t10k-labels-idx1-ubyte.gz.
    #[arg(long, default_value = "data/mnist")]
    mnist: PathBuf,
    /// NumPy's scores for the same formula (digits x scores, little-endian f64).
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Print one JSON object instead of text.
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let formula: Formula = fs::read_to_string(&args.formula)
        .with_context(|| format!("cannot read {}", args.formula.display()))?
        .parse()
        .with_context(|| format!("cannot parse {}", args.formula.display()))?;
    let digits = Digits::test_set(&args.mnist)
        .with_context(|| format!("cannot read the MNIST test set in {}", args.mnist.display()))?;
    ensure!(
        formula.n_inputs() <= digits.image_size,
        "the formula reads more inputs than an image has"
    );

    let (scores, one_core) = per_digit(&digits, || {
        let mut slots = Vec::new();
        digits
            .images()
            .map(|image| formula.eval_with(image, &mut slots))
            .collect::<Vec<_>>()
    });
    let (_, all_cores) = per_digit(&digits, || {
        digits
            .pixels
            .par_chunks_exact(digits.image_size)
            .map_init(Vec::new, |slots, image| formula.eval_with(image, slots))
            .collect::<Vec<_>>()
    });
    let threads = rayon::current_num_threads();

    let correct = scores
        .iter()
        .zip(&digits.labels)
        .filter(|&(s, &label)| argmax(s) == label as usize)
        .count();
    let accuracy = correct as f64 / digits.len() as f64;
    let numpy = args
        .reference
        .as_deref()
        .map(|path| compare(&scores, path))
        .transpose()?;

    if args.json {
        println!(
            "{}",
            json!({
                "e_count": formula.len(),
                "e_count_per_digit": formula.input_dependent(),
                "accuracy": accuracy,
                "same_answer_as_numpy": numpy.as_ref().map(|a| a.same_answer),
                "same_scores_as_numpy": numpy.as_ref().map(|a| a.same_scores),
                "max_score_difference": numpy.as_ref().map(|a| a.max_difference),
                "us_per_digit_one_core": micros(one_core),
                "us_per_digit_all_cores": micros(all_cores),
                "threads": threads,
            })
        );
    } else {
        let n = digits.len();
        println!(
            "formula   {} E's ({} depend on the digit, the rest are computed once), {} inputs, {} scores",
            formula.len(),
            formula.input_dependent(),
            formula.n_inputs(),
            formula.n_outputs()
        );
        println!(
            "accuracy  {:.2}% ({correct} of {n} test digits)",
            100.0 * accuracy
        );
        if let Some(a) = numpy {
            println!(
                "vs NumPy  same answer on {} of {n}, same scores on {} of {n} (largest difference {:.1e})",
                a.same_answer, a.same_scores, a.max_difference
            );
        }
        println!(
            "speed     {:.1} µs per digit on one core, {:.1} µs per digit on {threads} cores",
            micros(one_core),
            micros(all_cores)
        );
    }
    Ok(())
}

/// Runs `work` (one pass over every digit) and returns its result and the time per digit.
fn per_digit<T>(digits: &Digits, work: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let out = black_box(work());
    (out, start.elapsed() / digits.len() as u32)
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Index of the first largest score, like NumPy's argmax.
fn argmax(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .min_by(|a, b| b.1.total_cmp(a.1))
        .map_or(0, |(i, _)| i)
}

/// How closely our scores match NumPy's.
#[derive(Default)]
struct Agreement {
    same_answer: usize,
    same_scores: usize,
    max_difference: f64,
}

fn compare(scores: &[Vec<f64>], path: &Path) -> Result<Agreement> {
    let bytes = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let (values, rest) = bytes.as_chunks::<8>();
    let width = scores.first().map_or(0, Vec::len);
    ensure!(
        rest.is_empty() && values.len() == scores.len() * width,
        "{} does not hold {} x {width} scores",
        path.display(),
        scores.len()
    );
    let reference: Vec<f64> = values.iter().map(|&b| f64::from_le_bytes(b)).collect();

    let mut agreement = Agreement::default();
    for (ours, theirs) in scores.iter().zip(reference.chunks_exact(width)) {
        agreement.same_answer += usize::from(argmax(ours) == argmax(theirs));
        agreement.same_scores += usize::from(ours.iter().zip(theirs).all(|(&a, &b)| same(a, b)));
        for (a, b) in ours
            .iter()
            .zip(theirs)
            .filter(|(a, b)| a.is_finite() && b.is_finite())
        {
            agreement.max_difference = agreement.max_difference.max((a - b).abs());
        }
    }
    Ok(agreement)
}

/// Equal up to rounding; infinities and NaN must match exactly.
fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}
