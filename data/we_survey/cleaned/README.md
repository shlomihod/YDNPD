
### Description
Data from the Workplace Equity Survey 2023 https://www.openicpsr.org/openicpsr/project/202701/version/V1/view

#### Cleaning
Lightly cleaned, columns dropped with fewer than 3% of respondents answering. Additionally, answers discretized using simple categorical ecnoding:
```
column_mapping = {}
for col in df.select_dtypes(include=['category']).columns:
    column_mapping[col] = dict(enumerate(df[col].cat.categories))

columns_with_a_single_value = list(df.columns[df.nunique() == 1])
df = df.drop(columns=["respondent_id"] + columns_with_a_single_value)
df = df.fillna(-1).astype("int64")
```

#### Citation
Lemieux, Camille, Taylor, Simone, Stone, Anne, Wooden, Paige, and Chauhan, Chhavi. Workplace Equity Survey 2023. Ann Arbor, MI: Inter-university Consortium for Political and Social Research, 2024-05-30. https://doi.org/10.3886/E202701V1