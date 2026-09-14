use crate::tacer::types::RetrievalSignals;

pub fn retrieval_signals(scores: &[f64]) -> RetrievalSignals {
    let scores: Vec<f64> = scores.iter().map(|score| score.max(0.0)).collect();

    if scores.is_empty() {
        return RetrievalSignals {
            top_score: 0.0,
            mean_score: 0.0,
            score_std: 0.0,
            gap_1_2: 0.0,
            gap_3_4: 0.0,
            gap_5_6: 0.0,
            gap_1_5: 0.0,
            top1_mass: 0.0,
            top3_mass: 0.0,
            top5_mass: 0.0,
            top1_to_top5: 0.0,
            top3_to_top8: 0.0,
            entropy: 0.0,
            normalized_entropy: 0.0,
        };
    }

    let top_score = scores[0];

    let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

    let variance = scores
        .iter()
        .map(|score| (score - mean_score).powi(2))
        .sum::<f64>()
        / scores.len() as f64;

    let score_std = variance.sqrt();

    let score_sum: f64 = scores.iter().sum();

    let gap = |left: usize, right: usize| -> f64 {
        if scores.len() <= right || top_score == 0.0 {
            0.0
        } else {
            (scores[left] - scores[right]) / top_score
        }
    };

    let top1_mass = if score_sum > 0.0 {
        scores[0] / score_sum
    } else {
        0.0
    };

    let top3_mass = if score_sum > 0.0 {
        scores.iter().take(3).sum::<f64>() / score_sum
    } else {
        0.0
    };

    let top5_mass = if score_sum > 0.0 {
        scores.iter().take(5).sum::<f64>() / score_sum
    } else {
        0.0
    };

    let top1_to_top5 = if scores.len() >= 5 && scores[4] > 0.0 {
        scores[0] / scores[4]
    } else {
        0.0
    };

    let top8_sum: f64 = scores.iter().take(8).sum();

    let top3_to_top8 = if top8_sum > 0.0 {
        scores.iter().take(3).sum::<f64>() / top8_sum
    } else {
        0.0
    };

    let entropy = if score_sum > 0.0 {
        scores
            .iter()
            .filter(|&&score| score > 0.0)
            .map(|&score| {
                let p = score / score_sum;
                -p * p.ln()
            })
            .sum()
    } else {
        0.0
    };

    let normalized_entropy = if scores.len() > 1 {
        entropy / (scores.len() as f64).ln()
    } else {
        0.0
    };

    RetrievalSignals {
        top_score,
        mean_score,
        score_std,
        gap_1_2: gap(0, 1),
        gap_3_4: gap(2, 3),
        gap_5_6: gap(4, 5),
        gap_1_5: gap(0, 4),
        top1_mass,
        top3_mass,
        top5_mass,
        top1_to_top5,
        top3_to_top8,
        entropy,
        normalized_entropy,
    }
}
