import argparse
from pathlib import Path


SQL_NAMES = [
    "q10", "q11", "q12", "q13", "q14a", "q14b", "q15", "q16", "q17", "q18",
    "q19", "q1", "q20", "q21", "q22", "q23a", "q23b", "q24a", "q24b", "q25",
    "q26", "q27", "q28", "q29", "q2", "q30", "q31", "q32", "q33", "q34",
    "q35", "q36", "q37", "q38", "q39a", "q39b", "q3", "q40", "q41", "q42",
    "q43", "q44", "q45", "q46", "q47", "q48", "q49", "q4", "q50", "q51",
    "q52", "q53", "q54", "q55", "q56", "q57", "q58", "q59", "q5", "q60",
    "q61", "q62", "q63", "q64", "q65", "q66", "q67", "q68", "q69", "q6",
    "q70", "q71", "q72", "q73", "q74", "q75", "q76", "q77", "q78", "q79",
    "q7", "q80", "q81", "q82", "q83", "q84", "q85", "q86", "q87", "q88",
    "q89", "q8", "q90", "q91", "q92", "q93", "q94", "q95", "q96", "q97",
    "q98", "q99", "q9",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mock_data"),
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_placeholder_sql(path: Path) -> None:
    if path.exists():
        return
    path.write_text("-- placeholder SQL\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    ensure_directory(output_dir)

    for sql_name in SQL_NAMES:
        sql_path = output_dir / f"{sql_name}.sql"
        write_placeholder_sql(sql_path)
    print(f"Created/verified {len(SQL_NAMES)} SQL files under {output_dir}")

if __name__ == "__main__":
    main()

