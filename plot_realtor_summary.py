"""Plot the server Realtor summary using the Google Playstore figure style."""
import argparse
import csv
from pathlib import Path

from plot_experiment_summaries import BASE_DIR, METHOD_ORDER, paired_plot


def normalize(method):
    if method == "SharedQ":
        return "sharedq"
    if method.startswith("Baseline_"):
        return "baseline_" + method.removeprefix("Baseline_")
    if method.startswith("SUR_"):
        return "shared_utility_range_" + method.removeprefix("SUR_")
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=BASE_DIR / "results" /
                        "Realtor_server_seed2026102610_full9" / "summary.csv")
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    indexed = {}
    for row in rows:
        key = (int(row["dimension"]), normalize(row["method"]))
        if key in indexed:
            raise ValueError(f"Duplicate summary row: {key}")
        if int(row["successful"]) != int(row["attempted"]):
            raise ValueError(f"Incomplete runs: {key}")
        indexed[key] = row
    summary_dir = BASE_DIR / "results" / "real_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for metric, field in [("time", "elapsed"), ("questions", "questions")]:
        path = summary_dir / f"Realtor_{metric}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dimension", *METHOD_ORDER])
            for dim in sorted({key[0] for key in indexed}):
                writer.writerow([f"{dim}d", *[indexed[dim, method][field] for method in METHOD_ORDER]])
        print(f"wrote {path}")
    paired_plot(summary_dir / "Realtor_time.csv", summary_dir / "Realtor_questions.csv",
                BASE_DIR / "results" / "figures" / "Realtor.pdf", "Realtor by dimension", "line")


if __name__ == "__main__":
    main()
