"""
Fetch papers citing AESTETIK and update README.md.

Combines results from Semantic Scholar and OpenCitations APIs to maximize
coverage. Uses Semantic Scholar for metadata and OpenCitations to discover
additional citing DOIs.

AESTETIK DOI: 10.1101/2024.06.04.24308256
"""

import json
import re
import urllib.request
import urllib.error
from pathlib import Path

AESTETIK_DOI = "10.1101/2024.06.04.24308256"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
OPENCITATIONS_API = "https://opencitations.net/index/api/v2"
README_PATH = Path(__file__).parent.parent / "README.md"

SECTION_START = "<!-- CITATIONS:START -->"
SECTION_END = "<!-- CITATIONS:END -->"

SS_FIELDS = "title,authors,year,externalIds,url,venue,journal,publicationDate"


def _api_get(url):
    """Make a GET request and return parsed JSON, or None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "AESTETIK-Citation-Bot"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"API error for {url}: {e}")
        return None


def fetch_ss_citations():
    """Fetch citing papers from Semantic Scholar."""
    url = (
        f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{AESTETIK_DOI}/citations"
        f"?fields={SS_FIELDS}&limit=500"
    )
    data = _api_get(url)
    if not data:
        return {}

    papers = {}
    for item in data.get("data", []):
        paper = item.get("citingPaper", {})
        doi = paper.get("externalIds", {}).get("DOI", "")
        if not paper.get("title"):
            continue
        key = doi.lower() if doi else paper.get("paperId", "")
        papers[key] = _parse_ss_paper(paper)
    return papers


def fetch_opencitations_dois():
    """Fetch citing DOIs from OpenCitations (may find papers Semantic Scholar misses)."""
    url = f"{OPENCITATIONS_API}/citations/doi:{AESTETIK_DOI}"
    data = _api_get(url)
    if not data:
        return set()

    dois = set()
    for item in data:
        citing = item.get("citing", "")
        for part in citing.split():
            if part.startswith("doi:"):
                dois.add(part[4:])
    return dois


def fetch_ss_paper_by_doi(doi):
    """Fetch a single paper's metadata from Semantic Scholar by DOI."""
    url = f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{doi}?fields={SS_FIELDS}"
    data = _api_get(url)
    if not data or not data.get("title"):
        return None
    return _parse_ss_paper(data)


def _parse_ss_paper(paper):
    """Parse a Semantic Scholar paper object into our format."""
    authors = [a["name"] for a in paper.get("authors", []) if a.get("name")]
    doi = paper.get("externalIds", {}).get("DOI", "")
    link = f"https://doi.org/{doi}" if doi else paper.get("url", "")
    venue = paper.get("venue", "")
    journal = paper.get("journal", {})
    if journal and journal.get("name") and not venue:
        venue = journal["name"]

    return {
        "title": paper["title"],
        "authors": authors,
        "year": paper.get("year", ""),
        "venue": venue,
        "link": link,
        "date": paper.get("publicationDate", ""),
    }


def fetch_all_citing_papers():
    """Combine Semantic Scholar and OpenCitations to find all citing papers."""
    # Fetch from Semantic Scholar (primary source with full metadata)
    papers = fetch_ss_citations()
    print(f"Semantic Scholar: {len(papers)} citing paper(s)")

    # Fetch citing DOIs from OpenCitations and fill in any missing ones
    oc_dois = fetch_opencitations_dois()
    print(f"OpenCitations: {len(oc_dois)} citing DOI(s)")

    missing_dois = [doi for doi in oc_dois if doi.lower() not in papers]
    for doi in missing_dois:
        paper = fetch_ss_paper_by_doi(doi)
        if paper:
            papers[doi.lower()] = paper
            print(f"  Added from OpenCitations: {paper['title']}")

    all_papers = list(papers.values())
    # Deterministic order: newest first, with title as a stable tiebreaker so
    # papers sharing a date don't reorder between runs.
    all_papers.sort(
        key=lambda p: (p.get("date") or str(p.get("year", "")), p.get("title", "")),
        reverse=True,
    )
    print(f"Total: {len(all_papers)} unique citing paper(s)")
    return all_papers


DOI_LINK_RE = re.compile(r"https?://doi\.org/([^)\s]+)", re.IGNORECASE)
ENTRY_LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")
ENTRY_AUTHORS_RE = re.compile(r'^\d+\.\s+(.+?)\s+"')


def _citation_key(link):
    """Stable key for a citation: bare DOI for doi.org links, else the raw URL."""
    if not link:
        return ""
    match = DOI_LINK_RE.search(link)
    return match.group(1).lower() if match else link.strip()


def parse_existing_authors(readme):
    """Map citation key -> author string already rendered in the README.

    Semantic Scholar returns author names inconsistently across runs (e.g. full
    name one week, initials the next). Reusing the author string already committed
    for a known paper keeps the list stable and preserves manual name fixes; only
    genuinely new papers pick up freshly fetched names.
    """
    existing = {}
    if SECTION_START not in readme or SECTION_END not in readme:
        return existing
    section = readme.split(SECTION_START, 1)[1].split(SECTION_END, 1)[0]
    for line in section.splitlines():
        authors = ENTRY_AUTHORS_RE.match(line.strip())
        links = ENTRY_LINK_RE.findall(line)
        if authors and links:
            existing[_citation_key(links[-1])] = authors.group(1).strip()
    return existing


def format_authors(authors):
    """Format author list: 'First Author, Second Author, ..., and Last Author'."""
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + ", and " + authors[-1]


def format_citations(papers, existing_authors=None):
    """Format papers as a numbered markdown list in academic citation style."""
    existing_authors = existing_authors or {}
    if not papers:
        return "No citations found yet. Check back soon!\n"

    lines = []
    for i, p in enumerate(papers, 1):
        # Prefer the author string already committed for this paper so flaky
        # name variants from the API don't churn the list on every run.
        author_str = existing_authors.get(_citation_key(p["link"])) or format_authors(p["authors"])
        parts = [f"{i}. {author_str}"]
        parts.append(f'"{p["title"].rstrip(". ")}."')
        if p["venue"]:
            parts.append(f'*{p["venue"]}*')
        if p["year"]:
            parts.append(f'({p["year"]}).')
        else:
            parts[-1] += "."
        entry = " ".join(parts)
        if p["link"]:
            entry += f" [DOI]({p['link']})"
        lines.append(entry)

    return "\n".join(lines) + "\n"


def update_readme(papers):
    """Update the citations section in README.md."""
    readme = README_PATH.read_text()

    existing_authors = parse_existing_authors(readme)
    citations_md = format_citations(papers, existing_authors)
    new_section = f"{SECTION_START}\n{citations_md}{SECTION_END}"

    if SECTION_START in readme and SECTION_END in readme:
        pattern = re.escape(SECTION_START) + r".*?" + re.escape(SECTION_END)
        readme = re.sub(pattern, new_section, readme, flags=re.DOTALL)
    else:
        print("Citation markers not found in README.md. Please add them manually.")
        return False

    README_PATH.write_text(readme)
    print(f"Updated README with {len(papers)} citing paper(s).")
    return True


if __name__ == "__main__":
    papers = fetch_all_citing_papers()
    update_readme(papers)
