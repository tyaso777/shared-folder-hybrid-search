use serde::{Deserialize, Serialize};

use crate::schema::PreparedRecord;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkMode {
    None,
    Smart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkOptions {
    pub mode: ChunkMode,
    pub size: usize,
    pub overlap: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedChunk {
    pub chunk_id: String,
    pub record_id: String,
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub text: String,
}

struct ChunkSpan {
    start_char: usize,
    end_char: usize,
    text: String,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            mode: ChunkMode::None,
            size: 1200,
            overlap: 200,
        }
    }
}

impl ChunkOptions {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.mode == ChunkMode::None {
            return Ok(());
        }
        if self.size == 0 {
            anyhow::bail!("chunk size must be greater than zero");
        }
        if self.overlap >= self.size {
            anyhow::bail!("chunk overlap must be smaller than chunk size");
        }
        Ok(())
    }
}

pub fn build_chunks(
    records: &[PreparedRecord],
    options: &ChunkOptions,
) -> anyhow::Result<Vec<PreparedChunk>> {
    options.validate()?;
    let mut chunks = Vec::new();
    for record in records {
        let spans = match options.mode {
            ChunkMode::None => vec![ChunkSpan {
                start_char: 0,
                end_char: record.searchable_text.chars().count(),
                text: record.searchable_text.clone(),
            }],
            ChunkMode::Smart => {
                smart_chunks(&record.searchable_text, options.size, options.overlap)
            }
        };
        for (chunk_index, span) in spans.into_iter().enumerate() {
            if span.text.trim().is_empty() {
                continue;
            }
            chunks.push(PreparedChunk {
                chunk_id: format!("{}#chunk-{chunk_index}", record.record_id),
                record_id: record.record_id.clone(),
                chunk_index,
                start_char: span.start_char,
                end_char: span.end_char,
                text: span.text,
            });
        }
    }
    Ok(chunks)
}

fn smart_chunks(text: &str, size: usize, overlap: usize) -> Vec<ChunkSpan> {
    let text = text.trim();
    if text.chars().count() <= size {
        return vec![ChunkSpan {
            start_char: 0,
            end_char: text.chars().count(),
            text: text.to_string(),
        }];
    }

    let total_chars = text.chars().count();
    let min_size = (size * 6 / 10).max(1);
    let mut chunks = Vec::new();
    let mut start_char = 0usize;

    while start_char < total_chars {
        let remaining = total_chars - start_char;
        if remaining <= size {
            if let Some(chunk) = make_span(text, start_char, total_chars) {
                chunks.push(chunk);
            }
            break;
        }

        let target_char = start_char + size;
        let max_char = (start_char + size + size / 4).min(total_chars);
        let cut_char = choose_cut_char(text, start_char, target_char, max_char, min_size)
            .filter(|cut| *cut > start_char)
            .unwrap_or(target_char.min(total_chars));

        if let Some(chunk) = make_span(text, start_char, cut_char) {
            chunks.push(chunk);
        }

        if cut_char >= total_chars {
            break;
        }
        start_char = next_start_char(text, start_char, cut_char, overlap);
        if start_char >= cut_char {
            start_char = cut_char;
        }
    }

    chunks
}

fn make_span(text: &str, start_char: usize, end_char: usize) -> Option<ChunkSpan> {
    let raw = slice_chars(text, start_char, end_char);
    let leading_trim = raw.chars().take_while(|ch| ch.is_whitespace()).count();
    let trailing_trim = raw
        .chars()
        .rev()
        .take_while(|ch| ch.is_whitespace())
        .count();
    let trimmed_start = start_char + leading_trim;
    let trimmed_end = end_char.saturating_sub(trailing_trim);
    if trimmed_start >= trimmed_end {
        return None;
    }
    Some(ChunkSpan {
        start_char: trimmed_start,
        end_char: trimmed_end,
        text: slice_chars(text, trimmed_start, trimmed_end).to_string(),
    })
}

fn choose_cut_char(
    text: &str,
    start_char: usize,
    target_char: usize,
    max_char: usize,
    min_size: usize,
) -> Option<usize> {
    let min_char = start_char + min_size;
    let mut best: Option<(usize, i32, usize)> = None;
    for (idx, ch) in text.chars().enumerate() {
        let boundary_char = idx + 1;
        if boundary_char < min_char || boundary_char > max_char {
            continue;
        }
        let score = boundary_score(ch);
        if score == 0 {
            continue;
        }
        let distance = boundary_char.abs_diff(target_char);
        match best {
            None => best = Some((boundary_char, score, distance)),
            Some((_, best_score, best_distance)) => {
                if score > best_score || (score == best_score && distance < best_distance) {
                    best = Some((boundary_char, score, distance));
                }
            }
        }
    }
    best.map(|(boundary_char, _, _)| boundary_char)
}

fn next_start_char(text: &str, chunk_start: usize, cut_char: usize, overlap: usize) -> usize {
    if overlap == 0 || cut_char <= overlap {
        return cut_char;
    }
    let min_start = (cut_char - overlap).max(chunk_start + 1);
    let mut best = min_start;
    for (idx, ch) in text.chars().enumerate() {
        let boundary_char = idx + 1;
        if boundary_char < min_start || boundary_char >= cut_char {
            continue;
        }
        if boundary_score(ch) >= 3 {
            best = boundary_char;
        }
    }
    best
}

fn boundary_score(ch: char) -> i32 {
    match ch {
        '\n' => 5,
        '。' | '！' | '？' | '.' | '!' | '?' => 4,
        '、' | ',' | ';' | '；' | ':' | '：' => 2,
        c if c.is_whitespace() => 1,
        _ => 0,
    }
}

fn slice_chars(text: &str, start: usize, end: usize) -> &str {
    let start_byte = char_to_byte(text, start);
    let end_byte = char_to_byte(text, end);
    &text[start_byte..end_byte]
}

fn char_to_byte(text: &str, char_pos: usize) -> usize {
    if char_pos == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_pos)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}
