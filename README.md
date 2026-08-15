# NBA Salary Analysis — 2025

A Python data-analysis project examining how NBA player performance, age, and position relate to 2025 player salary. The project cleans and merges salary and player-statistics data, fits nine OLS regression models, compares model performance, and produces exploratory and diagnostic visualizations.

## Project Questions

- How strongly are individual box-score statistics associated with player salary?
- Does combining scoring, rebounding, playmaking, defense, age, and position improve salary prediction?
- Does log-transforming salary produce a better-behaved regression model?
- Which model provides the strongest fit among the specifications tested?

## Features

- Cleans and merges NBA salary and player-statistics datasets
- Converts salary strings into numeric values for modeling
- Fits nine regression models using `statsmodels`
- Compares R², adjusted R², AIC, and BIC
- Generates salary distributions, correlation analysis, regression plots, and residual diagnostics
- Saves processed data and model output automatically
- Uses repository-relative paths, so the project works after cloning in VS Code or another IDE

## Project Structure

```text
nba-salary-analysis/
├── main.py
├── index.html
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── nba_salary_2025.csv
│   │   └── nba_stats_2025.csv
│   └── processed/
│       ├── working_data.csv
│       └── working_data_clean.xlsx
└── outputs/
    ├── model_comparison_table.csv
    ├── model_summaries.txt
    └── figures/
        └── *.png
```

## Models Tested

1. Points only
2. Assists only
3. Rebounds only
4. Basic box-score production
5. Offensive production
6. Defense and rebounding
7. Shooting efficiency
8. Box score + age + position
9. Log salary + box score + age + position

## Run Locally in VS Code

1. Clone or download this repository.
2. Open the repository folder in VS Code.
3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate it.

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the analysis:

```bash
python main.py
```

The processed datasets, model comparison table, model summaries, and figures will be written to the `data/processed/` and `outputs/` folders.

## GitHub Pages

`index.html` is included at the repository root, so the repository can also be published as a simple GitHub Pages project site. In GitHub, choose **Settings → Pages → Deploy from a branch → main → / (root)**.

## Tools

Python, pandas, NumPy, statsmodels, Matplotlib, Seaborn, and openpyxl.

## Interpretation Note

Model 8 has the highest adjusted R² among the models that predict salary on its original dollar scale. Model 9 predicts log salary, so its AIC/BIC values should not be directly compared with the raw-salary models because the response variable is on a different scale.

## Data Note

The repository contains the two raw CSV files used in the submitted analysis. If you publish this publicly, add the original data-source URLs/citations to this README if you have them.
