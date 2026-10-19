# Citation Style Language styles

The five `.csl` files in this folder are copied unmodified from the
Citation Style Language styles repository,
<https://github.com/citation-style-language/styles>, at commit
`0e5503bd9a75668c402567a93e20b7327784e4fa` (2026-10-16).

They are licensed under the Creative Commons Attribution-ShareAlike 3.0
Unported license, <https://creativecommons.org/licenses/by-sa/3.0/>, as each
file's own `<rights>` element states. The authors and contributors of each
style are named in its `<info>` block. Attestation's code is MIT licensed;
these data files keep their own license, and any change to one must be shared
under CC BY-SA 3.0.

| Attestation id        | File                          | Style title (from the file)                                         |
|-----------------------|-------------------------------|---------------------------------------------------------------------|
| `apa`                 | `apa.csl`                     | APA Style 7th edition                                               |
| `ieee`                | `ieee.csl`                    | IEEE Reference Guide version 11.29.2023                             |
| `chicago-author-date` | `chicago-author-date.csl`     | Chicago Manual of Style 18th edition (author-date)                  |
| `nature`              | `nature.csl`                  | Nature                                                              |
| `vancouver`           | `nlm-citation-sequence.csl`   | NLM/Vancouver: Citing Medicine 2nd edition (citation-sequence)      |

The repository renamed its Vancouver style to `nlm-citation-sequence` (pull
request 7928, "NLM/Vancouver: Use source titles and official terminology");
Attestation still calls it `vancouver` because that is the name researchers
ask for.

The locale files the styles need (`en-US`) ship inside citeproc-py, so
formatting never fetches anything.
