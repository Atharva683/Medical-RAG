import fitz  # PyMuPDF
import re

# Load PDF
doc = fitz.open("dsa509.pdf")
full_text = "\n".join([page.get_text() for page in doc])

# Define medical section headings (regex)
section_pattern = re.compile(r"(?:^|\n)([A-Z][A-Z \-]{5,})\n")

# Find all headings
matches = list(section_pattern.finditer(full_text))

# Extract sections and their content
chunks = []
for i, match in enumerate(matches):
    title = match.group(1).strip()
    start = match.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
    content = full_text[start:end].strip()

    if len(content.split()) > 1200:
        # Split long sections into subchunks
        words = content.split()
        for j in range(0, len(words), 1000):
            subchunk = " ".join(words[j:j + 1000])
            chunks.append({
                "title": f"{title} (Part {j // 1000 + 1})",
                "content": subchunk
            })
    else:
        chunks.append({
            "title": title,
            "content": content
        })

# Save to file
with open("section_chunks.txt", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(f"### {c['title']}\n{c['content']}\n{'='*80}\n")

print(f"Saved {len(chunks)} section-aware chunks.")
