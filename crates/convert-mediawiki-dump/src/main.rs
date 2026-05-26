use std::fs;
use std::io::{BufReader, Write};
use std::path::PathBuf;

use anyhow::Context;
use bzip2::read::BzDecoder;
use clap::Parser;
use quick_xml::events::Event;
use quick_xml::Reader;
use regex::Regex;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Convert a local MediaWiki XML dump into flat JSONL"
)]
struct Args {
    #[arg(long, default_value = "jawikibooks")]
    dataset: String,
    #[arg(long)]
    input: PathBuf,
    #[arg(long, default_value = "examples/jawikibooks")]
    output_dir: PathBuf,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long, default_value_t = 100)]
    min_chars: usize,
}

#[derive(Debug, Default)]
struct Page {
    title: String,
    id: String,
    ns: String,
    text: String,
    redirect: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    None,
    Title,
    Id,
    Ns,
    Text,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)?;
    write_schema(&args)?;

    let input = fs::File::open(&args.input)
        .with_context(|| format!("open dump {}", args.input.display()))?;
    let reader: Box<dyn std::io::BufRead> = if args
        .input
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bz2"))
    {
        Box::new(BufReader::new(BzDecoder::new(input)))
    } else {
        Box::new(BufReader::new(input))
    };

    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut current = Page::default();
    let mut in_page = false;
    let mut in_revision = false;
    let mut field = Field::None;
    let mut records = Vec::new();
    let cleaner = WikiTextCleaner::new()?;

    loop {
        match xml.read_event_into(&mut buf)? {
            Event::Start(e) => match e.name().as_ref() {
                b"page" => {
                    in_page = true;
                    current = Page::default();
                }
                b"revision" if in_page => in_revision = true,
                b"title" if in_page => field = Field::Title,
                b"ns" if in_page => field = Field::Ns,
                b"id" if in_page && !in_revision && current.id.is_empty() => field = Field::Id,
                b"text" if in_page && in_revision => field = Field::Text,
                b"redirect" if in_page => current.redirect = true,
                _ => {}
            },
            Event::Text(e) => {
                let text = e.decode()?.into_owned();
                match field {
                    Field::Title => current.title.push_str(&text),
                    Field::Id => current.id.push_str(&text),
                    Field::Ns => current.ns.push_str(&text),
                    Field::Text => current.text.push_str(&text),
                    Field::None => {}
                }
            }
            Event::CData(e) => {
                let text = e.decode()?.into_owned();
                if field == Field::Text {
                    current.text.push_str(&text);
                }
            }
            Event::End(e) => match e.name().as_ref() {
                b"revision" => in_revision = false,
                b"title" | b"id" | b"ns" | b"text" => field = Field::None,
                b"page" => {
                    in_page = false;
                    if let Some(record) = page_to_record(&args, &cleaner, &current) {
                        records.push(record);
                        if records.len() % 100 == 0 {
                            eprintln!("converted {} pages", records.len());
                        }
                        if args.limit.is_some_and(|limit| records.len() >= limit) {
                            break;
                        }
                    }
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    let output_path = args.output_dir.join("input.jsonl");
    let mut output = fs::File::create(&output_path)?;
    for record in records {
        writeln!(output, "{}", serde_json::to_string(&record)?)?;
    }
    println!("wrote {}", output_path.display());
    Ok(())
}

fn page_to_record(
    args: &Args,
    cleaner: &WikiTextCleaner,
    page: &Page,
) -> Option<serde_json::Value> {
    if page.redirect || page.ns != "0" || page.id.is_empty() || page.title.trim().is_empty() {
        return None;
    }
    let text = cleaner.clean(&page.text);
    if text.chars().count() < args.min_chars {
        return None;
    }
    let encoded_title = page.title.replace(' ', "_");
    Some(json!({
        "page_id": page.id,
        "title": page.title,
        "text": text,
        "source": args.dataset,
        "url": format!("https://ja.wikibooks.org/wiki/{encoded_title}"),
        "converted_at": chrono::Utc::now().to_rfc3339()
    }))
}

fn write_schema(args: &Args) -> anyhow::Result<()> {
    let schema = json!({
        "dataset_id": args.dataset,
        "primary_key": "page_id",
        "text_fields": ["title", "text"],
        "full_text_fields": ["title", "text"],
        "source_uri_field": "url",
        "source_label_field": "source",
        "display_fields": ["title", "url", "source", "converted_at"],
        "filter_fields": {
            "source": {
                "type": "keyword",
                "label": "Source",
                "ui": "select"
            }
        }
    });
    fs::write(
        args.output_dir.join("schema.json"),
        serde_json::to_vec_pretty(&schema)?,
    )?;
    Ok(())
}

struct WikiTextCleaner {
    comment: Regex,
    ref_tag: Regex,
    template: Regex,
    file_link: Regex,
    category_link: Regex,
    external_link: Regex,
    internal_link_with_label: Regex,
    internal_link: Regex,
    markup: Regex,
    whitespace: Regex,
}

impl WikiTextCleaner {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            comment: Regex::new(r"(?s)<!--.*?-->")?,
            ref_tag: Regex::new(r"(?is)<ref[^>]*>.*?</ref>|<ref[^>]*/>")?,
            template: Regex::new(r"(?s)\{\{.*?\}\}")?,
            file_link: Regex::new(r"\[\[(?:ファイル|画像|File|Image):[^\]]+\]\]")?,
            category_link: Regex::new(r"\[\[(?:Category|カテゴリ):[^\]]+\]\]")?,
            external_link: Regex::new(r"\[(https?://[^\s\]]+)\s+([^\]]+)\]")?,
            internal_link_with_label: Regex::new(r"\[\[[^|\]]+\|([^\]]+)\]\]")?,
            internal_link: Regex::new(r"\[\[([^\]]+)\]\]")?,
            markup: Regex::new(r"'{2,}|={2,}|<[^>]+>|&nbsp;")?,
            whitespace: Regex::new(r"[ \t\r\n]+")?,
        })
    }

    fn clean(&self, input: &str) -> String {
        let mut text = input.to_string();
        for _ in 0..4 {
            text = self.template.replace_all(&text, " ").into_owned();
        }
        text = self.comment.replace_all(&text, " ").into_owned();
        text = self.ref_tag.replace_all(&text, " ").into_owned();
        text = self.file_link.replace_all(&text, " ").into_owned();
        text = self.category_link.replace_all(&text, " ").into_owned();
        text = self.external_link.replace_all(&text, "$2").into_owned();
        text = self
            .internal_link_with_label
            .replace_all(&text, "$1")
            .into_owned();
        text = self.internal_link.replace_all(&text, "$1").into_owned();
        text = self.markup.replace_all(&text, " ").into_owned();
        self.whitespace.replace_all(&text, " ").trim().to_string()
    }
}
