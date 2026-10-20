# CLI reference

Generated from `attestation.cli.build_parser()` by `scripts/render_cli_reference.py`; `tests/test_docs_site.py` asserts this file is a fresh render. Do not hand-edit.

## attest ingest

```
usage: attest ingest [-h] [--db DB] [--feeds FEEDS] [--no-research]
                     [--fulltext-limit FULLTEXT_LIMIT]

options:
  -h, --help            show this help message and exit
  --db DB               DB path. Resolution order if omitted: ATTEST_DB (or
                        RSS_DB) env var > ~/.hermes/skills/science-
                        recommendations/data/hermes.db (if it exists) >
                        ./hermes.db
  --feeds FEEDS
  --no-research         skip research: feeds this run
  --fulltext-limit FULLTEXT_LIMIT
                        bodies to fetch after ingest (0 = none)
```

## attest research

```
usage: attest research [-h] [--db DB] [--sources SOURCES] [--journal JOURNAL]
                       [--since-days SINCE_DAYS] [--limit LIMIT] [--no-store]
                       query

positional arguments:
  query

options:
  -h, --help            show this help message and exit
  --db DB               DB path. Resolution order if omitted: ATTEST_DB (or
                        RSS_DB) env var > ~/.hermes/skills/science-
                        recommendations/data/hermes.db (if it exists) >
                        ./hermes.db
  --sources SOURCES     comma-separated: arxiv,pubmed,crossref,openalex
  --journal JOURNAL     pubmed/crossref only
  --since-days SINCE_DAYS
  --limit LIMIT
  --no-store            preview; do not write the library
```

## attest sources add

```
usage: attest sources add [-h] [--title TITLE] [--user USER] url

positional arguments:
  url            an RSS/Atom URL or research:<clients>?q=<query>[&journal=...]

options:
  -h, --help     show this help message and exit
  --title TITLE
  --user USER    persona to record as the one who added it
```

## attest tag

```
usage: attest tag [-h] [--db DB] [--limit LIMIT]

options:
  -h, --help     show this help message and exit
  --db DB        DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB)
                 env var > ~/.hermes/skills/science-
                 recommendations/data/hermes.db (if it exists) > ./hermes.db
  --limit LIMIT  max items to tag this run
```

## attest serve

```
usage: attest serve [-h] [--db DB] [--port PORT]

options:
  -h, --help   show this help message and exit
  --db DB      DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB) env
               var > ~/.hermes/skills/science-recommendations/data/hermes.db
               (if it exists) > ./hermes.db
  --port PORT
```

## attest eval

```
usage: attest eval [-h] [--db DB] --user USER

options:
  -h, --help   show this help message and exit
  --db DB      DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB) env
               var > ~/.hermes/skills/science-recommendations/data/hermes.db
               (if it exists) > ./hermes.db
  --user USER
```

## attest warmup

```
usage: attest warmup [-h]

options:
  -h, --help  show this help message and exit
```

## attest reload

```
usage: attest reload [-h]

options:
  -h, --help  show this help message and exit
```

## attest backup

```
usage: attest backup [-h] [--db DB] dest

positional arguments:
  dest        path to write; must not already exist

options:
  -h, --help  show this help message and exit
  --db DB     DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB) env
              var > ~/.hermes/skills/science-recommendations/data/hermes.db
              (if it exists) > ./hermes.db
```

## attest reembed

```
usage: attest reembed [-h] [--db DB] [--force]

options:
  -h, --help  show this help message and exit
  --db DB     DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB) env
              var > ~/.hermes/skills/science-recommendations/data/hermes.db
              (if it exists) > ./hermes.db
  --force     drop and rebuild even when the stored model and width already
              match
```

## attest emit

```
usage: attest emit [-h] [--write]

options:
  -h, --help  show this help message and exit
  --write     write the Claude agent files (default: report only)
```

## attest manifest

```
usage: attest manifest [-h]

options:
  -h, --help  show this help message and exit
```

## attest kg-report

```
usage: attest kg-report [-h] [--db DB] [--min-size MIN_SIZE]

options:
  -h, --help           show this help message and exit
  --db DB              DB path. Resolution order if omitted: ATTEST_DB (or
                       RSS_DB) env var > ~/.hermes/skills/science-
                       recommendations/data/hermes.db (if it exists) >
                       ./hermes.db
  --min-size MIN_SIZE  smallest cluster to list
```

## attest claims

```
usage: attest claims [-h] [--db DB] [--verdict VERDICT] [--coverage]
                     [--record OUT.json] [--force-record] [--citations]
                     [path]

positional arguments:
  path               file or directory (default: $RESEARCH_ROOT)

options:
  -h, --help         show this help message and exit
  --db DB            DB path. Resolution order if omitted: ATTEST_DB (or
                     RSS_DB) env var > ~/.hermes/skills/science-
                     recommendations/data/hermes.db (if it exists) >
                     ./hermes.db
  --verdict VERDICT  show only this verdict
  --coverage         instead: list numbers in prose that no claim covers
  --record OUT.json  also write a signed run record
  --force-record     replace an existing record
  --citations        instead: list citation keys (cite= and LaTeX \cite) the
                     bibliography lacks
```

## attest browse

```
usage: attest browse [-h] [--db DB] [--port PORT] [--open]

options:
  -h, --help   show this help message and exit
  --db DB      DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB) env
               var > ~/.hermes/skills/science-recommendations/data/hermes.db
               (if it exists) > ./hermes.db
  --port PORT
  --open       open a browser window
```

## attest support

```
usage: attest support [-h] [--db DB] [--sentence SENTENCE] [--cite CITE]
                      [--judge {auto,llm,none}] [--verdict VERDICT]
                      [--fail-on FAIL_ON] [--limit LIMIT] [--json]
                      [path]

positional arguments:
  path                  a Markdown or .tex draft, or a directory of them

options:
  -h, --help            show this help message and exit
  --db DB               DB path. Resolution order if omitted: ATTEST_DB (or
                        RSS_DB) env var > ~/.hermes/skills/science-
                        recommendations/data/hermes.db (if it exists) >
                        ./hermes.db
  --sentence SENTENCE   check one sentence instead of a draft (needs --cite)
  --cite CITE           the citation key, DOI or arXiv id --sentence cites
  --judge {auto,llm,none}
                        auto: the local model, else the deterministic check;
                        none: deterministic only
  --verdict VERDICT     show only this verdict
  --fail-on FAIL_ON     comma-separated verdicts that make the exit status 1
                        (default: contradicted)
  --limit LIMIT         check at most this many citations
  --json                print the result as JSON
```

## attest desk build

```
usage: attest desk build [-h] --user USER --out OUT [--limit LIMIT]

options:
  -h, --help     show this help message and exit
  --user USER
  --out OUT
  --limit LIMIT
```

## attest desk import

```
usage: attest desk import [-h] --user USER

options:
  -h, --help   show this help message and exit
  --user USER
```

## attest desk refresh

```
usage: attest desk refresh [-h]

options:
  -h, --help  show this help message and exit
```

## attest runs scan

```
usage: attest runs scan [-h] [--root ROOT] [--project PROJECT]

options:
  -h, --help         show this help message and exit
  --root ROOT        workspace dir (default: $RESEARCH_ROOT)
  --project PROJECT  scan only this project
```

## attest runs list

```
usage: attest runs list [-h] [--project PROJECT] [--family FAMILY]
                        [--limit LIMIT]

options:
  -h, --help         show this help message and exit
  --project PROJECT
  --family FAMILY
  --limit LIMIT
```

## attest runs compare

```
usage: attest runs compare [-h] [--metric METRIC] [--project PROJECT] family

positional arguments:
  family

options:
  -h, --help         show this help message and exit
  --metric METRIC    default: the metric most arms share
  --project PROJECT  required when the family exists in more than one project
```

## attest runs show

```
usage: attest runs show [-h] project name

positional arguments:
  project
  name

options:
  -h, --help  show this help message and exit
```

## attest runs record

```
usage: attest runs record [-h] --arm NAME [METRIC=VALUE ...] [--corpus CORPUS]
                          [--direction METRIC=lower_is_better|higher_is_better]
                          [--config KEY=VALUE] [--root ROOT] [--dry-run]
                          [--force] [--scan]
                          family

positional arguments:
  family

options:
  -h, --help            show this help message and exit
  --arm NAME [METRIC=VALUE ...]
                        one arm: its name, then one or more METRIC=VALUE pairs
  --corpus CORPUS       corpus name; declares it in corpora.toml
  --direction METRIC=lower_is_better|higher_is_better
                        declare a metric not already in
                        ledger.METRIC_DIRECTION
  --config KEY=VALUE    extra provenance pair written into each arm's config
                        file
  --root ROOT           where to write files (default: cwd)
  --dry-run             print the manifest, write nothing
  --force               overwrite existing files/entries
  --scan                also run `runs scan` and print `runs compare`
```

## attest library sync

```
usage: attest library sync [-h] [--sources SOURCES] [--limit LIMIT]

options:
  -h, --help         show this help message and exit
  --sources SOURCES  comma-separated subset:
                     bibtex,zotero,feed,arxiv,crossref,s2
  --limit LIMIT      max rows per enricher and per embed pass
```

## attest library search

```
usage: attest library search [-h] [--author AUTHOR] [--year YEAR] [--tag TAG]
                             [--limit LIMIT]
                             [query]

positional arguments:
  query

options:
  -h, --help       show this help message and exit
  --author AUTHOR  surname filter
  --year YEAR
  --tag TAG
  --limit LIMIT
```

## attest library tag

```
usage: attest library tag [-h] [--limit LIMIT]

options:
  -h, --help     show this help message and exit
  --limit LIMIT
```

## attest library embed

```
usage: attest library embed [-h] [--limit LIMIT]

options:
  -h, --help     show this help message and exit
  --limit LIMIT
```

## attest library status

```
usage: attest library status [-h]

options:
  -h, --help  show this help message and exit
```

## attest library related

```
usage: attest library related [-h] key

positional arguments:
  key         citation key, DOI, arXiv id, or library identity

options:
  -h, --help  show this help message and exit
```

## attest library fulltext

```
usage: attest library fulltext [-h] [--limit LIMIT]

options:
  -h, --help     show this help message and exit
  --limit LIMIT
```

## attest library bib

```
usage: attest library bib [-h] [--user USER]

options:
  -h, --help   show this help message and exit
  --user USER  one persona (default: every persona with signal)
```

## attest library dashboard

```
usage: attest library dashboard [-h] [--out OUT] [--persona PERSONA]
                                [--limit LIMIT] [--style STYLE]

options:
  -h, --help         show this help message and exit
  --out OUT          output path (default: $ATTEST_LIBRARY_DASHBOARD, else the
                     Research Desk's workspace/research-
                     desk/library/library.json)
  --persona PERSONA  whose saved papers to mark (default: $ATTEST_DESK_USER,
                     else owner)
  --limit LIMIT      max references listed (default and maximum: 5000)
  --style STYLE      the citation style precomputed into each row (default:
                     $ATTEST_CITATION_STYLE, else apa)
```

## attest library export

```
usage: attest library export [-h] --bib BIB [--author AUTHOR] [--year YEAR]
                             [--tag TAG] [--source SOURCE] [--force]

options:
  -h, --help       show this help message and exit
  --bib BIB        output path; refuses to overwrite without --force
  --author AUTHOR
  --year YEAR
  --tag TAG
  --source SOURCE  reference_sources.source prefix, e.g. zotero,
                   research:arxiv
  --force
```

## attest bib search

```
usage: attest bib search [-h] [--author AUTHOR] [--year YEAR] [--tag TAG]
                         [--limit LIMIT]
                         [query]

positional arguments:
  query

options:
  -h, --help       show this help message and exit
  --author AUTHOR  surname filter
  --year YEAR
  --tag TAG
  --limit LIMIT
```

## attest bib show

```
usage: attest bib show [-h] [--style STYLE] [--json] key

positional arguments:
  key            citation key, DOI, arXiv id, or library identity

options:
  -h, --help     show this help message and exit
  --style STYLE  citation style (default: $ATTEST_CITATION_STYLE, else apa)
  --json         print the record as JSON
```

## attest bib format

```
usage: attest bib format [-h] [--style STYLE] [--bibtex] keys [keys ...]

positional arguments:
  keys           citation keys, DOIs or arXiv ids

options:
  -h, --help     show this help message and exit
  --style STYLE  apa, ieee, chicago-author-date, nature or vancouver
  --bibtex       print BibTeX instead
```

## attest bib import

```
usage: attest bib import [-h] [--zotero] [path]

positional arguments:
  path        a .bib file (or zotero.sqlite with --zotero)

options:
  -h, --help  show this help message and exit
  --zotero    read a local zotero.sqlite (default: ~/Zotero)
```

## attest bib edit

```
usage: attest bib edit [-h] [--set FIELD=VALUE] [--note NOTE] [--tag TAG]
                       [--untag UNTAG] [--user USER]
                       key

positional arguments:
  key                citation key, DOI or arXiv id

options:
  -h, --help         show this help message and exit
  --set FIELD=VALUE  title, authors (A and B), year, venue, doi, arxiv, url,
                     abstract, type, volume, pages, or any other BibTeX field;
                     an empty value clears it
  --note NOTE        the owner's note (empty clears it)
  --tag TAG          add a tag
  --untag UNTAG      remove a tag
  --user USER        whose .bib it goes to (default: $ATTEST_DESK_USER, else
                     owner)
```

## attest bib merge

```
usage: attest bib merge [-h] [--all] [keep] [drop]

positional arguments:
  keep        the reference to keep
  drop        the duplicate folded into it

options:
  -h, --help  show this help message and exit
  --all       merge every listed duplicate pair
```

## attest bib export

```
usage: attest bib export [-h] [--out OUT] [--style STYLE] [--force] [keys ...]

positional arguments:
  keys           references to export (default: the whole library)

options:
  -h, --help     show this help message and exit
  --out OUT      write to this file (refuses to overwrite without --force)
  --style STYLE  formatted text in this style instead of BibTeX
  --force
```

## attest bootstrap-persona

```
usage: attest bootstrap-persona [-h] [--db DB] [-k K] name

positional arguments:
  name

options:
  -h, --help  show this help message and exit
  --db DB     DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB) env
              var > ~/.hermes/skills/science-recommendations/data/hermes.db
              (if it exists) > ./hermes.db
  -k K
```

## attest persona-merge

```
usage: attest persona-merge [-h] [--db DB] into drop [drop ...]

positional arguments:
  into        the persona to keep, e.g. matt
  drop        duplicates to fold in, e.g. "Matthew Goldey"

options:
  -h, --help  show this help message and exit
  --db DB     DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB) env
              var > ~/.hermes/skills/science-recommendations/data/hermes.db
              (if it exists) > ./hermes.db
```

## attest record create

```
usage: attest record create [-h] --spec SPEC --out OUT [--root ROOT] [--force]

options:
  -h, --help   show this help message and exit
  --spec SPEC  the run, as JSON (see the design spec)
  --out OUT    where to write the record
  --root ROOT  where the spec's paths resolve (default: cwd)
  --force      replace an existing record
```

## attest record verify

```
usage: attest record verify [-h] [--root ROOT] [--key KEY | --keyid KEYID]
                            [--json]
                            record

positional arguments:
  record         the run record (a DSSE envelope .json)

options:
  -h, --help     show this help message and exit
  --root ROOT    where the record's file names resolve (default: cwd)
  --key KEY      require this ed25519 public key (PEM)
  --keyid KEYID  require this key id (hex sha256)
  --json         print the report as JSON
```

## attest record key

```
usage: attest record key [-h] [--export PUB.pem]

options:
  -h, --help        show this help message and exit
  --export PUB.pem  also write the public key here
```

## attest install

```
usage: attest install [-h] [--check] [--yes] [--now]

options:
  -h, --help  show this help message and exit
  --check     detect only, change nothing
  --yes       non-interactive consent
  --now       also run the tag backfill inline
```
