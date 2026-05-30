# Journal of Earthquake Engineering submission guideline check

Date checked: 2026-05-30

Target journal: Journal of Earthquake Engineering, Taylor & Francis, journal code `ueqe20`.

## Live-source findings

- The Taylor & Francis journal page confirms that the journal publishes research and development in analytical, experimental and field studies of earthquakes, including soil dynamics and foundations, site effects, dynamic soil-structure interaction, foundation design for earthquake loading, and seismic response of buildings.
- The official Taylor & Francis author-instructions endpoint for `ueqe20` returned HTTP 403 from this machine during the check, so the live IFA page could not be fully scraped.
- A search-indexed copy of the journal instructions states that accepted article types include original articles and notes, and that the manuscript structure should be title page, abstract, keywords, main text, acknowledgments, declaration of interest statement, references, appendices as appropriate, tables with captions, figures, and figure captions.
- No journal-specific strict word limit was found in the accessible official/search-indexed materials. Taylor & Francis general author guidance says word limits are journal-specific; if no word limit is specified, authors can assume the editors do not have a strong preference.
- Taylor & Francis provides a general Word template for journal articles. The 2025-03-12 Microsoft Word 2016 instructions and template zip were downloaded locally:
  - `TF_Template_Word_Windows_2016_instructions.pdf`
  - `TF_Template_Word_Windows_2016.zip`
  - extracted template: `TF_Template_Word_Windows_2016/TF_Template_Word_Windows.dotx`

## Adaptation decision

- The manuscript was not split, because no strict word limit was found for Journal of Earthquake Engineering in the accessible current materials.
- The article was adapted for journal fit by sharpening the title, abstract novelty statement, introduction scope fit, and conclusion contribution framing.
- The general Taylor & Francis Word template was downloaded and archived, but direct template attachment was not forced after Word became unstable during that operation. The manuscript was instead formatted conservatively with Word-native styles, equation-editor objects preserved, readable tables, declarations, data/code/supplementary statements, and exported PDF.
- Supplementary material with text and tables was created as a separate file with independent supplementary numbering: `Table S1`, `Table S2`.

## Current local counts

- Main manuscript Word count by Microsoft Word: 11,378 words.
- Main manuscript pages by Microsoft Word: 58 pages.
- Microsoft Word equation objects: 40.
- Main manuscript tables: 19.
- Supplementary material tables: 2, numbered `Table S1` and `Table S2`.

## Remaining caveat

Before final portal upload, manually inspect the Taylor & Francis submission portal fields because portal-specific requirements can differ from the public journal page and may include separate upload slots for manuscript, figures, supplementary files, data availability, graphical abstract, or declarations.
