# Input data

The analysis scripts expect the workbook:

`Used Datas ML (24 Variables).xlsx`

by default in this directory. Alternatively set the environment variable `MACRO_DATA_FILE` to the workbook location.

Expected annual sheet names:
- `Turkiye Annually`
- `USA Annually`
- `UK Annually`
- `Germany Annualy`
- `France Annualy`

The analysis uses the ten variables `CDS, EXC, INF, TPD, INT, STC, CAB, EID, GRI, INV` over 1994--2024. The exact data sources, original frequencies, and annualization rules should be documented in Appendix A of the manuscript.

The raw workbook is not included in this code/results release because redistribution conditions should be checked for each underlying data provider. If redistribution is permitted, place the exact analysis workbook here so the repository can be reproduced without additional setup.
