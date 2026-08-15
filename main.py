"""NBA Salary Analysis

Reproduces the data cleaning, regression-model comparison, and visualizations
for the 2025 NBA salary/statistics analysis.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in (PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)



def load_and_clean_data() -> pd.DataFrame:
    """Load raw salary/stat files, merge them, and return analysis-ready data."""
    nba_salary = pd.read_csv(RAW_DIR / "nba_salary_2025.csv")
    nba_stats = pd.read_csv(RAW_DIR / "nba_stats_2025.csv")

    nba_salary = nba_salary.rename(columns={"-additional": "player_id"})
    nba_stats = nba_stats.rename(columns={"Player-additional": "player_id"})

    # The first salary row is a source-table header, not a player observation.
    nba_salary = nba_salary[nba_salary["player_id"] != "-9999"].copy()

    working_data = pd.merge(nba_salary, nba_stats, on="player_id", how="inner")

    working_data = working_data.rename(
        columns={
            "Unnamed: 0": "Player_Rank_Salary",
            "Unnamed: 1": "Player_Name",
            "Unnamed: 2": "Team_Salary",
            "Unnamed: 9": "Guaranteed_Salary",
            "Salary": "Salary_2025",
            "Rk": "Player_Rank_Stats",
            "FG%": "FG_Percent",
            "3P": "ThreeP",
            "3P%": "ThreeP_Percent",
            "2P": "TwoP",
            "2P%": "TwoP_Percent",
            "FT%": "FT_Percent",
            "eFG%": "eFG_Percent",
        }
    )

    columns_to_drop = [
        "Salary.1", "Salary.2", "Salary.3", "Salary.4", "Salary.5",
        "Player", "G", "GS", "MP", "FGA", "3PA", "2PA", "FTA", "TOV", "PF",
    ]
    working_data = working_data.drop(columns=columns_to_drop)

    working_data["Salary_2025"] = (
        working_data["Salary_2025"].replace(r"[$,]", "", regex=True).astype(float)
    )
    working_data["Log_Salary_2025"] = np.log(working_data["Salary_2025"])

    working_data.to_csv(PROCESSED_DIR / "working_data.csv", index=False)
    working_data.to_excel(PROCESSED_DIR / "working_data_clean.xlsx", index=False)
    return working_data


def fit_models(data: pd.DataFrame):
    """Fit the nine regression models from the original analysis."""
    models = {
        "Model 1: Points Only": smf.ols("Salary_2025 ~ PTS", data=data).fit(),
        "Model 2: Assists Only": smf.ols("Salary_2025 ~ AST", data=data).fit(),
        "Model 3: Rebounds Only": smf.ols("Salary_2025 ~ TRB", data=data).fit(),
        "Model 4: Box Score": smf.ols(
            "Salary_2025 ~ PTS + TRB + AST + STL + BLK", data=data
        ).fit(),
        "Model 5: Offensive Production": smf.ols(
            "Salary_2025 ~ PTS + AST + FG + ThreeP + TwoP + FT + ORB", data=data
        ).fit(),
        "Model 6: Defense/Rebounding": smf.ols(
            "Salary_2025 ~ DRB + STL + BLK", data=data
        ).fit(),
        "Model 7: Efficiency Only": smf.ols(
            "Salary_2025 ~ FG_Percent + ThreeP_Percent + FT_Percent + eFG_Percent",
            data=data,
        ).fit(),
        "Model 8: Box Score + Age + Position": smf.ols(
            "Salary_2025 ~ PTS + TRB + AST + STL + BLK + Age + C(Pos)", data=data
        ).fit(),
        "Model 9: Log Salary + Box Score + Age + Position": smf.ols(
            "Log_Salary_2025 ~ PTS + TRB + AST + STL + BLK + Age + C(Pos)", data=data
        ).fit(),
    }
    return models


def build_model_comparison(models) -> pd.DataFrame:
    """Create and save a compact model-comparison table."""
    response_variables = ["Salary_2025"] * 8 + ["Log_Salary_2025"]
    comparison = pd.DataFrame(
        {
            "Model": list(models.keys()),
            "Response_Variable": response_variables,
            "R_squared": [model.rsquared for model in models.values()],
            "Adjusted_R_squared": [model.rsquared_adj for model in models.values()],
            "AIC": [model.aic for model in models.values()],
            "BIC": [model.bic for model in models.values()],
        }
    ).sort_values("Adjusted_R_squared", ascending=False)

    comparison.to_csv(OUTPUT_DIR / "model_comparison_table.csv", index=False)

    with open(OUTPUT_DIR / "model_summaries.txt", "w", encoding="utf-8") as f:
        for name, model in models.items():
            f.write(f"\n{'=' * 80}\n{name}\n{'=' * 80}\n")
            f.write(model.summary().as_text())
            f.write("\n")

    return comparison


def save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_figures(data: pd.DataFrame, models, comparison: pd.DataFrame) -> None:
    """Generate the exploratory and regression-diagnostic figures."""
    missing_counts = data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    if not missing_counts.empty:
        plt.bar(missing_counts.index, missing_counts.values)
        plt.xlabel("Variable")
        plt.ylabel("Number of Missing Values")
        plt.xticks(rotation=45, ha="right")
    else:
        plt.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=14)
        plt.axis("off")
    plt.title("Missing Values by Variable")
    save_current_figure("missing_values.png")

    plt.figure(figsize=(8, 5))
    plt.hist(data["Salary_2025"].dropna(), bins=30)
    plt.title("Distribution of NBA Player Salaries")
    plt.xlabel("Salary 2025")
    plt.ylabel("Count")
    save_current_figure("salary_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.hist(data["Log_Salary_2025"].dropna(), bins=30)
    plt.title("Distribution of Log NBA Player Salaries")
    plt.xlabel("Log Salary 2025")
    plt.ylabel("Count")
    save_current_figure("log_salary_distribution.png")

    numeric_vars = [
        "Salary_2025", "Log_Salary_2025", "Age", "PTS", "TRB", "AST", "STL", "BLK",
        "FG", "ThreeP", "TwoP", "FT", "FG_Percent", "ThreeP_Percent", "FT_Percent",
        "eFG_Percent", "ORB", "DRB",
    ]
    corr_data = data[[var for var in numeric_vars if var in data.columns]].corr()
    plt.figure(figsize=(12, 8))
    matrix = corr_data.to_numpy()
    image = plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image, label="Correlation")
    labels = corr_data.columns.tolist()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.title("Correlation Heatmap of Salary and Player Statistics")
    save_current_figure("correlation_heatmap.png")

    for stat in ["PTS", "AST", "TRB"]:
        plot_data = data[[stat, "Salary_2025"]].dropna()
        x = plot_data[stat].to_numpy()
        y = plot_data["Salary_2025"].to_numpy()
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.6)
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_line, slope * x_line + intercept)
        plt.title(f"Salary vs {stat}")
        plt.xlabel(stat)
        plt.ylabel("Salary 2025")
        save_current_figure(f"salary_vs_{stat}.png")

    positions = sorted(data["Pos"].dropna().unique())
    position_values = [data.loc[data["Pos"] == pos, "Salary_2025"].dropna() for pos in positions]
    plt.figure(figsize=(10, 6))
    plt.boxplot(position_values, tick_labels=positions)
    plt.title("Salary Distribution by Position")
    plt.xlabel("Position")
    plt.ylabel("Salary 2025")
    plt.xticks(rotation=45)
    save_current_figure("salary_by_position.png")

    comparison_plot = comparison.sort_values("Adjusted_R_squared", ascending=True)
    plt.figure(figsize=(11, 6))
    plt.barh(comparison_plot["Model"], comparison_plot["Adjusted_R_squared"])
    plt.title("Model Comparison by Adjusted R-Squared")
    plt.xlabel("Adjusted R-Squared")
    plt.ylabel("Model")
    save_current_figure("model_comparison_adjusted_r_squared.png")

    comparison_aic = comparison.sort_values("AIC", ascending=False)
    plt.figure(figsize=(11, 6))
    plt.barh(comparison_aic["Model"], comparison_aic["AIC"])
    plt.title("Model Comparison by AIC")
    plt.xlabel("AIC")
    plt.ylabel("Model")
    save_current_figure("model_comparison_aic.png")

    context_model = models["Model 8: Box Score + Age + Position"]
    data = data.copy()
    data["Predicted_Salary_Context"] = context_model.fittedvalues
    data["Residuals_Context"] = context_model.resid

    plt.figure(figsize=(8, 5))
    plt.scatter(data["Salary_2025"], data["Predicted_Salary_Context"], alpha=0.6)
    low = min(data["Salary_2025"].min(), data["Predicted_Salary_Context"].min())
    high = max(data["Salary_2025"].max(), data["Predicted_Salary_Context"].max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title("Observed vs Predicted Salary: Context Model")
    plt.xlabel("Observed Salary 2025")
    plt.ylabel("Predicted Salary 2025")
    save_current_figure("observed_vs_predicted_context_model.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(context_model.fittedvalues, context_model.resid, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals vs Fitted Values: Context Model")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    save_current_figure("residuals_vs_fitted_context_model.png")

    qqplot(context_model.resid, line="45", fit=True)
    plt.title("Q-Q Plot of Residuals: Context Model")
    save_current_figure("qq_plot_context_model.png")

    log_model = models["Model 9: Log Salary + Box Score + Age + Position"]
    data["Predicted_Log_Salary"] = log_model.fittedvalues
    data["Residuals_Log_Context"] = log_model.resid

    plt.figure(figsize=(8, 5))
    plt.scatter(data["Log_Salary_2025"], data["Predicted_Log_Salary"], alpha=0.6)
    low = min(data["Log_Salary_2025"].min(), data["Predicted_Log_Salary"].min())
    high = max(data["Log_Salary_2025"].max(), data["Predicted_Log_Salary"].max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title("Observed vs Predicted Log Salary: Log Context Model")
    plt.xlabel("Observed Log Salary 2025")
    plt.ylabel("Predicted Log Salary 2025")
    save_current_figure("observed_vs_predicted_log_context_model.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(log_model.fittedvalues, log_model.resid, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals vs Fitted Values: Log Context Model")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    save_current_figure("residuals_vs_fitted_log_context_model.png")

    qqplot(log_model.resid, line="45", fit=True)
    plt.title("Q-Q Plot of Residuals: Log Context Model")
    save_current_figure("qq_plot_log_context_model.png")


def main() -> None:
    data = load_and_clean_data()
    models = fit_models(data)
    comparison = build_model_comparison(models)
    create_figures(data, models, comparison)

    print(f"Analysis complete: {len(data):,} merged player-season rows")
    print("\nModel comparison (sorted by adjusted R-squared):")
    print(comparison.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
