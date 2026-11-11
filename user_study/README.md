# Amazon User Study Web App

This is a lightweight Web system for collecting real-user interaction data.
It uses only the Python standard library for the server.

## Run

```powershell
python user_study/app.py
```

Then open:

```text
http://127.0.0.1:8765
```

## Methods

The interface supports:

- `SUR-HDPI`
- `SUR-RH`
- `SUR-UH-Simplex`
- `SUR-UH-Random`
- `SharedQ`

The original algorithm code is simulation-oriented and expects a hidden true
utility vector. For real users, this app uses interactive study-mode selectors:
after each A/B answer, the backend updates a sampled utility range and chooses
the next pair according to the selected method's question-selection strategy.

## Dataset

Supply the study dataset separately. The default path is:

```text
after_skyline_datasets/_category_gap_search/Amazon_single_category_4d_10
```

The four displayed attributes are normalized larger-is-better values:

- Cheap price
- Rating
- Review count
- Bought count

## Output

Interaction logs:

```text
user_study/results/interactions.csv
```

Survey logs:

```text
user_study/results/surveys.csv
```

The survey records:

- question count
- elapsed time
- annoyance
- satisfaction
