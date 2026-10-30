# Data provenance

`raw/` contains byte-for-byte copies of the supplied `wine+quality` files.
Keep these immutable. `cleaned/` is generated, comma-delimited, and includes a
1-based `source_record_id` (data-record number, excluding the header).
`results/<color>/duplicate_map.csv` maps every raw record to its retained record.
Exact duplicates are removed before splitting; no imputation or outlier removal.

Source: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality),
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
[Dataset DOI](https://doi.org/10.24432/C56S3T).
UCI lists the dataset as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes: deduplication and added source record identifiers in derived files only.

Original study: Cortez et al. (2009), *Modeling wine preferences by data mining
from physicochemical properties*, Decision Support Systems 47(4), 547–553.
[Article DOI](https://doi.org/10.1016/j.dss.2009.05.016).
The supplied `raw/winequality.names` contains the original documentation and
citation request. Quality is an ordinal 0–10 expert score; actual red range 3–8,
white 3–9. The files lack wine, batch and producer identifiers.
