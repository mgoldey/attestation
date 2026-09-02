# CLI reference

Generated from `attestation.cli.build_parser()` by `scripts/render_cli_reference.py`; `tests/test_docs_site.py` asserts this file is a fresh render. Do not hand-edit.

## attest ingest

```
usage: attest ingest [-h] [--db DB] [--feeds FEEDS]

options:
  -h, --help     show this help message and exit
  --db DB        DB path. Resolution order if omitted: ATTEST_DB (or RSS_DB)
                 env var > ~/.hermes/skills/science-
                 recommendations/data/hermes.db (if it exists) > ./hermes.db
  --feeds FEEDS
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

## attest emit

```
usage: attest emit [-h] [--write]

options:
  -h, --help  show this help message and exit
  --write     write the Claude agent files (default: report only)
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
usage: attest claims [-h] [--db DB] [--verdict VERDICT] [--coverage] [path]

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

## attest install

```
usage: attest install [-h] [--check] [--yes] [--now]

options:
  -h, --help  show this help message and exit
  --check     detect only, change nothing
  --yes       non-interactive consent
  --now       also run the tag backfill inline
```
