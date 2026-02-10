from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from models import Student


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "static" / "plots"


def _query_to_dataframe() -> pd.DataFrame:
    """Load student records into a pandas DataFrame."""
    students = Student.query.all()
    if not students:
        return pd.DataFrame(
            columns=["name", "age", "study_hours", "attendance", "exam_score"]
        )

    data = [
        {
            "name": s.name,
            "age": s.age,
            "study_hours": s.study_hours,
            "attendance": s.attendance,
            "exam_score": s.exam_score,
        }
        for s in students
    ]
    return pd.DataFrame(data)


def get_summary_statistics() -> Dict[str, float]:
    """Return simple summary statistics for study hours, attendance, and scores."""
    df = _query_to_dataframe()
    if df.empty:
        return {}

    stats = {}
    for col in ["study_hours", "attendance", "exam_score"]:
        stats[f"{col}_mean"] = float(df[col].mean())
        stats[f"{col}_min"] = float(df[col].min())
        stats[f"{col}_max"] = float(df[col].max())
    return stats


def generate_all_plots() -> List[str]:
    """
    Generate basic visualizations and return relative paths to the plot images.

    Returns a list of paths relative to the `static` folder,
    e.g. ["plots/study_hours_vs_score.png", ...].
    """
    df = _query_to_dataframe()
    if df.empty:
        return []

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_paths: List[str] = []

    # 1. Bar plot: exam scores per student
    bar_path = PLOTS_DIR / "exam_scores_bar.png"
    plt.figure(figsize=(6, 4))
    plt.bar(df["name"], df["exam_score"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Exam Score (%)")
    plt.title("Exam Scores by Student")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    plot_paths.append(f"plots/{bar_path.name}")

    # 2. Line plot: study hours vs exam score (sorted by study hours)
    line_path = PLOTS_DIR / "study_hours_line.png"
    df_sorted = df.sort_values(by="study_hours")
    plt.figure(figsize=(6, 4))
    plt.plot(df_sorted["study_hours"], df_sorted["exam_score"], marker="o")
    plt.xlabel("Study Hours")
    plt.ylabel("Exam Score (%)")
    plt.title("Study Hours vs Exam Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(line_path)
    plt.close()
    plot_paths.append(f"plots/{line_path.name}")

    # 3. Scatter plot: attendance vs exam score
    scatter_path = PLOTS_DIR / "attendance_scatter.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(df["attendance"], df["exam_score"], c=df["study_hours"], cmap="viridis")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Exam Score (%)")
    plt.title("Attendance vs Exam Score (color = study hours)")
    cbar = plt.colorbar()
    cbar.set_label("Study Hours")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()
    plot_paths.append(f"plots/{scatter_path.name}")

    return plot_paths

