"""Build docs/bib/refs.{bib,ris,json,txt} and docs/bibliography.md from DOIs via doi.org content negotiation.

    python scripts/make_bibliography.py

Add a reference by adding (key, doi) to REFS; the formatted entries come from the publisher
metadata, so author lists and volumes are not typed by hand.
"""
import json
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = [  # (key, DOI), in the order they appear on the bibliography page
    ("grech2026", "10.1088/2058-9565/ae2c16"),
    ("mckay2016", "10.1103/PhysRevApplied.6.064007"),
    ("mckay2017", "10.1103/PhysRevA.96.022330"),
    ("childs2003", "10.1103/PhysRevA.68.052311"),
    ("pedersen2007", "10.1016/j.physleta.2007.02.069"),
    ("watts2015", "10.1103/PhysRevA.91.062306"),
    ("yan2018", "10.1103/PhysRevApplied.10.054062"),
    ("sung2021", "10.1103/PhysRevX.11.021058"),
    ("ding2023", "10.1103/PhysRevX.13.031035"),
    ("li2024", "10.1103/PhysRevX.14.041050"),
    ("sarma2025", "10.1103/PhysRevApplied.23.014015"),
    ("burnett2019", "10.1038/s41534-019-0168-5"),
    ("schlor2019", "10.1103/PhysRevLett.123.190502"),
    ("zhang2022", "10.1126/sciadv.abi6690"),
    ("dhieb2025", "10.48550/arXiv.2512.18037"),
    ("groszkowski2021", "10.22331/q-2021-11-17-583"),
    ("egger2014", "10.1103/PhysRevLett.112.240503"),
    ("wu2018", "10.1103/PhysRevA.97.042122"),
    ("schuld2019", "10.1103/PhysRevA.99.032331"),
    ("kelly2018", "10.48550/arXiv.1803.03226"),
    ("proctor2020", "10.1038/s41467-020-19074-4"),
    ("klimov2024", "10.1038/s41467-024-46623-y"),
    ("nesterov2017", "10.1007/s10208-015-9296-2"),
    ("amos2023", "10.1561/2200000102"),
]


def fetch(doi, accept):
    req = urllib.request.Request(f"https://doi.org/{doi}", headers={"Accept": accept, "User-Agent": "rlquantopt-docs"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8").strip()


def main():
    bib, ris, csl, txt, md = [], [], [], [], []
    for key, doi in REFS:
        try:
            b = fetch(doi, "application/x-bibtex")
        except Exception as e:
            print("skip", key, doi, e)
            continue
        bib.append(re.sub(r"^@(\w+)\{[^,]*,", lambda m: f"@{m.group(1)}{{{key},", b, count=1))
        ris.append(fetch(doi, "application/x-research-info-systems"))
        item = json.loads(fetch(doi, "application/vnd.citationstyles.csl+json"))
        item["id"] = key
        csl.append(item)
        ref = fetch(doi, "text/x-bibliography; style=apa")
        txt.append(ref)
        md.append(f"<span id=\"{key}\"></span>**[{key}]** {ref}\n")
        print("ok", key)
    out = os.path.join(ROOT, "docs", "bib")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "refs.bib"), "w").write("\n\n".join(bib) + "\n")
    open(os.path.join(out, "refs.ris"), "w").write("\n\n".join(ris) + "\n")
    json.dump(csl, open(os.path.join(out, "refs.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(out, "refs.txt"), "w").write("\n\n".join(txt) + "\n")
    page = """# Bibliography

Every work cited on this site. Download the whole list:

[BibTeX :material-download:](bib/refs.bib){ .md-button } [RIS (Zotero, Mendeley, EndNote) :material-download:](bib/refs.ris){ .md-button } [CSL-JSON :material-download:](bib/refs.json){ .md-button } [Plain text :material-download:](bib/refs.txt){ .md-button }

Entries are generated from the publishers' metadata at doi.org (`scripts/make_bibliography.py`).
Keys in brackets are the BibTeX keys.

"""
    open(os.path.join(ROOT, "docs", "bibliography.md"), "w").write(page + "\n".join(md))


if __name__ == "__main__":
    main()
