# Data provenance

`sources.json` pins the three user-requested Excel files to immutable GitHub commit URLs, exact byte sizes, Git blob SHA-1 identities and SHA-256 hashes. Every local or downloaded file is verified before parsing. Copies found on the author's machine were independently matched to the GitHub tree blob hashes before use. Runtime downloads use only the standard library.

The `raw/` cache is ignored by Git. To populate it, run `python datasets.py`, or simply run the experiment driver. The optional offline ZIP includes it. Rebuilding loaders requires no alternative CSV exports: the specified workbooks are the actual input sources.

| Dataset | Included source file | Attribution and terms |
|---|---|---|
| ERA5 | `ERA5_Annual_Temperature_1940_2025.xlsx` | Copernicus/ECMWF ERA5 t2m, through the author's regional-temperature repository. Consult its provenance and data attribution. The repository MIT notice is preserved in `notices/ERA5-repository-LICENSE.txt`; it does not relicense ECMWF data. |
| HoloAssist | `hand_tracking.xlsx` | HoloAssist, recording R0027-12-GoPro. Source repository data terms are preserved verbatim in `notices/HoloAssist-LICENSE-DATA.txt` (CDLA-Permissive-2.0). |
| NASA C-MAPSS | `NASA_FD001_sensor_by_cycle.xlsx` | Saxena and Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA Ames Prognostics Data Repository. The source's provenance/published-use notice is preserved in `notices/CMAPSS-LICENSE-DATA.txt`. It is not a new grant of rights or a software license. |

Original source pages and data details:

- https://github.com/ApoorvaGOR/ERA5-Regional-Temperature
- https://github.com/ApoorvaGOR/HoloAssist-Hand-Sample
- https://github.com/ApoorvaGOR/C-MAPSS-Turbofan-Data

The MIT license at the repository root covers this package's software. It does not override third-party data terms. No manuscript PDF is redistributed by the release builder.
