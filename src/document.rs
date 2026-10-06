use crate::extraction::SourceUnit;

pub enum DocumentFormat {
    Text,
    Markdown,
    Pdf,
    Docx,
}

pub struct ParsedDocument {
    pub name: String,
    pub format: DocumentFormat,
    pub sources: Vec<SourceUnit>,
}


pub fn parse_text(name: &str, text: &str) -> ParsedDocument {
    let sources = text
        .split("\n\n")
        .map(str::trim)
        .filter(|paragraph| !paragraph.is_empty())
        .enumerate()
        .map(|(id, paragraph)| SourceUnit {
            id,
            text: paragraph.to_string(),
            page: None,
            paragraph: Some(id),
            section: None,
        })
        .collect();

    ParsedDocument {
        name: name.to_string(),
        format: DocumentFormat::Text,
        sources,
    }
}

pub fn parse_markdown(name: &str, text: &str) -> ParsedDocument {
    let mut sources = Vec::new();
    let mut current_section: Option<String> = None;
    let mut paragraph_lines = Vec::new();

    let flush_paragraph = |
        sources: &mut Vec<SourceUnit>,
        paragraph_lines: &mut Vec<&str>,
        section: &Option<String>,
    | {
        if paragraph_lines.is_empty() {
            return;
        }

        let paragraph = paragraph_lines.join(" ");
        let id = sources.len();

        sources.push(SourceUnit {
            id,
            text: paragraph,
            page: None,
            paragraph: Some(id),
            section: section.clone(),
        });

        paragraph_lines.clear();
    };

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with('#') {
            flush_paragraph(
                &mut sources,
                &mut paragraph_lines,
                &current_section,
            );

            current_section = Some(
                trimmed
                    .trim_start_matches('#')
                    .trim()
                    .to_string(),
            );
        } else if trimmed.is_empty() {
            flush_paragraph(
                &mut sources,
                &mut paragraph_lines,
                &current_section,
            );
        } else {
            paragraph_lines.push(trimmed);
        }
    }

    flush_paragraph(
        &mut sources,
        &mut paragraph_lines,
        &current_section,
    );

    ParsedDocument {
        name: name.to_string(),
        format: DocumentFormat::Markdown,
        sources,
    }
}

pub fn parse_pdf(name: &str, path: &str) -> Result<ParsedDocument, String> {
    let pages = pdf_extract::extract_text_by_pages(path)
        .map_err(|error| format!("Could not extract PDF text: {}", error))?;

    let mut sources = Vec::new();

    for (page_index, page_text) in pages.iter().enumerate() {
        for paragraph in page_text
            .split("\n\n")
            .map(str::trim)
            .filter(|paragraph| !paragraph.is_empty())
        {
            let id = sources.len();

            sources.push(SourceUnit {
                id,
                text: paragraph.to_string(),
                page: Some(page_index + 1),
                paragraph: Some(id),
                section: None,
            });
        }
    }

    Ok(ParsedDocument {
        name: name.to_string(),
        format: DocumentFormat::Pdf,
        sources,
    })
}

pub fn parse_docx(name: &str, path: &str) -> Result<ParsedDocument, String> {
    use quick_xml::events::Event;
    use quick_xml::Reader;
    use std::fs::File;
    use std::io::Read;
    use zip::ZipArchive;

    let file = File::open(path)
        .map_err(|error| format!("Could not open DOCX: {}", error))?;

    let mut archive = ZipArchive::new(file)
        .map_err(|error| format!("Could not read DOCX archive: {}", error))?;

    let mut document_xml = String::new();

    archive
        .by_name("word/document.xml")
        .map_err(|error| format!("DOCX has no word/document.xml: {}", error))?
        .read_to_string(&mut document_xml)
        .map_err(|error| format!("Could not read DOCX document XML: {}", error))?;

    let mut reader = Reader::from_str(&document_xml);
    reader.config_mut().trim_text(false);

    let mut sources = Vec::new();
    let mut current_section: Option<String> = None;

    let mut in_paragraph = false;
    let mut in_text = false;
    let mut paragraph_text = String::new();
    let mut paragraph_style: Option<String> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Start(event)) => match event.local_name().as_ref() {
                b"p" => {
                    in_paragraph = true;
                    paragraph_text.clear();
                    paragraph_style = None;
                }
                b"t" if in_paragraph => {
                    in_text = true;
                }
                b"pStyle" if in_paragraph => {
                    for attribute in event.attributes().flatten() {
                        if attribute.key.local_name().as_ref() == b"val" {
                            paragraph_style =
                                Some(String::from_utf8_lossy(&attribute.value).into_owned());
                        }
                    }
                }
                _ => {}
            },

            Ok(Event::Empty(event)) => {
                if in_paragraph && event.local_name().as_ref() == b"pStyle" {
                    for attribute in event.attributes().flatten() {
                        if attribute.key.local_name().as_ref() == b"val" {
                            paragraph_style =
                                Some(String::from_utf8_lossy(&attribute.value).into_owned());
                        }
                    }
                }
            }

            Ok(Event::Text(event)) if in_paragraph && in_text => {
                let text = event
                    .decode()
                    .map_err(|error| format!("Could not decode DOCX text: {}", error))?;

                paragraph_text.push_str(&text);
            }

            Ok(Event::End(event)) => match event.local_name().as_ref() {
                b"t" => {
                    in_text = false;
                }
                b"p" if in_paragraph => {
                    in_paragraph = false;

                    let text = paragraph_text.trim().to_string();

                    if text.is_empty() {
                        continue;
                    }

                    let is_heading = paragraph_style
                        .as_deref()
                        .map(|style| style.to_ascii_lowercase().starts_with("heading"))
                        .unwrap_or(false);

                    if is_heading {
                        current_section = Some(text);
                        continue;
                    }

                    let id = sources.len();

                    sources.push(SourceUnit {
                        id,
                        text,
                        page: None,
                        paragraph: Some(id),
                        section: current_section.clone(),
                    });
                }
                _ => {}
            },

            Ok(Event::Eof) => break,

            Err(error) => {
                return Err(format!("Could not parse DOCX XML: {}", error));
            }

            _ => {}
        }
    }

    Ok(ParsedDocument {
        name: name.to_string(),
        format: DocumentFormat::Docx,
        sources,
    })
}