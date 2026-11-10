# UCI Wine benchmark fixture

Unmodified `wine.data` and `wine.names` from
[UCI Wine](https://archive.ics.uci.edu/dataset/109/winedataset), downloaded from
`https://archive.ics.uci.edu/static/public/109/wine.zip` on 2026-11-10.

Attribution: Aeberhard, S. & Forina, M. (1992). *Wine* [Dataset]. UCI Machine
Learning Repository. [DOI: 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J).
The data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
separately from the surrounding MIT/Apache-2.0 Rust code.

There are 178 chemical-analysis observations, 13 features and three cultivars.
Original class labels are 1–3; the Rust loader converts them to zero-based labels
without changing the vendored files. No data are downloaded at application runtime.

SHA-256:

```text
wine.data   6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659
wine.names  f1b84f2ef845e0bdebf13e14fa7a213e56de4f1baa40c5974dbd1ee51c5ae710
```
