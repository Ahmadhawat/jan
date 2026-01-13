from pathlib import Path
import sys
import shutil
import pandas as pd
import yaml
from openpyxl import load_workbook

# Assumed to exist in your project (as in your screenshots)
# - load_batch_eval_config()
# - load_config(cfg_path)
# - create_run_folder(run_base_dir)
# - create_config_folder(run_dir, cfg_name)
# - get_logger(name)
# - run_query(question: str, logs, search_cfg) -> str
# - clean_for_excel(raw_answer: str) -> str
# - write_pdf_from_markdown(md_path: Path, pdf_path: Path)
# - build_config_snapshot(search_cfg, snapshot_paths) -> dict

def main():
    be_cfg = load_batch_eval_config()

    # read base run directory directly from config
    run_base_dir = Path(be_cfg["run_base_dir"])

    input_xlsx = Path(be_cfg["input_xlsx"])
    question_col = be_cfg["question_col"]   # pandas column name (e.g. "Frage")
    answer_col = be_cfg["answer_col"]       # pandas column name (optional, used for df output)

    filenames = be_cfg["filenames"]
    excel_name = filenames["excel"]                 # e.g. "FragenSupport1.xlsx"
    md_name = filenames["markdown"]                 # e.g. "answers.md"
    pdf_name = filenames["pdf"]                     # e.g. "answers.pdf"
    cfg_snapshot_name = filenames["config_snapshot"]

    snapshot_paths = be_cfg.get("snapshot_paths", [])

    # Folder that contains **all** search-configuration YAML files
    search_cfg_folder = Path(be_cfg["search_config_folder"])
    cfg_files = sorted(search_cfg_folder.glob("*.yaml"))

    logs = get_logger("Ledger")
    run_dir = create_run_folder(run_base_dir)

    if not cfg_files:
        raise FileNotFoundError(
            f"No *.yaml files found in {search_cfg_folder!s}. "
            "Please provide at least one configuration file."
        )

    # ---------------------------------------------------------------------
    # 2) Read the questions ONCE - they are the same for every config
    # ---------------------------------------------------------------------
    df = pd.read_excel(input_xlsx)

    if question_col not in df.columns:
        raise ValueError(
            f"Column '{question_col}' not found in {input_xlsx!s}. "
            f"Columns: {list(df.columns)!r}"
        )

    # Keep a mapping to the original Excel row numbers:
    # Excel rows are 1-based and row 1 is header. Data starts at row 2.
    df = df.reset_index(drop=False).rename(columns={"index": "__df_index"})
    df["excel_row"] = df["__df_index"] + 2

    # Drop empty question rows (NaN or only whitespace)
    mask = df[question_col].notna() & df[question_col].astype(str).str.strip().astype(bool)
    df = df[mask].copy()

    # Pre-extract questions in the exact order they appear in Excel
    questions = df[question_col].astype(str).tolist()
    excel_rows = df["excel_row"].tolist()

    # ---------------------------------------------------------------------
    # 3) Process each configuration file
    # ---------------------------------------------------------------------
    for cfg_path in cfg_files:
        cfg_name = cfg_path.stem  # e.g. "search_config_1"

        # Load the specific configuration for this iteration
        search_cfg = load_config(cfg_path)

        # Create a dedicated run-folder for this config
        run_dir2 = create_config_folder(run_dir, cfg_name)
        logs.info(f"Run folder: {run_dir2}")

        # Copy the original Excel file into this config folder
        try:
            shutil.copy2(input_xlsx, run_dir2)  # preserves metadata
        except FileNotFoundError:
            logs.error(f"❌ Original file not found: {input_xlsx}")
            sys.exit(1)

        logs.info(f"✅ Copied '{input_xlsx.name}' -> '{run_dir2}'")

        # Paths for outputs inside this config folder
        excel_path = run_dir2 / input_xlsx.name   # copied workbook path
        md_path = run_dir2 / md_name
        pdf_path = run_dir2 / pdf_name
        cfg_snapshot_path = run_dir2 / cfg_snapshot_name

        # Open the COPIED workbook and locate columns
        wb = load_workbook(excel_path, data_only=False, keep_vba=False)
        ws = wb.worksheets[0]
        print(f"✅ Using sheet: {ws.title}")

        # Locate the columns "Frage" and "Antwort" by header row (row 1)
        frage_col = None
        antwort_col = None
        for cell in ws[1]:  # headers are in row 1
            if cell.value is None:
                continue
            header = str(cell.value).strip().lower()
            if header == "frage":
                frage_col = cell.column
            elif header == "antwort":
                antwort_col = cell.column

        if frage_col is None or antwort_col is None:
            print("❌ Could not find both 'Frage' and 'Antwort' headers.")
            sys.exit(1)

        # -------------------------------------------------------------
        # Run the query for every question (same question set)
        # -------------------------------------------------------------
        answers_raw = []
        answers_clean = []

        total = len(questions)
        for i, q in enumerate(questions, start=1):
            logs.info(f"[{i}/{total}] ({cfg_name}) {q!r}")
            raw_answer = run_query(q, logs, search_cfg)
            answers_raw.append(raw_answer)
            answers_clean.append(clean_for_excel(raw_answer))

        # Optional: store results in df (useful if you later export df)
        # (Don’t overwrite your original Excel via pandas; we write via openpyxl)
        if answer_col:
            df.loc[df["excel_row"].isin(excel_rows), answer_col] = answers_clean

        # -------------------------------------------------------------
        # ✅ FIXED: Write answers into the copied Excel file
        # We write answer i into the SAME original Excel row it came from.
        # -------------------------------------------------------------
        rows_written = 0
        for excel_row, ans in zip(excel_rows, answers_clean):
            # Optional safety check: only write if the "Frage" cell is non-empty
            frage_value = ws.cell(row=excel_row, column=frage_col).value
            if frage_value is None or str(frage_value).strip() == "":
                continue

            ws.cell(row=excel_row, column=antwort_col).value = ans
            rows_written += 1

        print(f"✅ Wrote answers to {rows_written} row(s).")

        # Save the modified copied workbook (in-place)
        wb.save(excel_path)
        print(f"✅ Changes saved to '{excel_path}'.")

        # -------------------------------------------------------------
        # Markdown (for humans + PDF)
        # -------------------------------------------------------------
        with md_path.open("w", encoding="utf-8") as f:
            for i, (q, ans) in enumerate(zip(questions, answers_raw), start=1):
                f.write(f"## {i}. Frage\n\n")
                f.write(f"{q}\n\n")
                f.write("**Antwort**\n\n")
                f.write(str(ans).strip())
                f.write("\n\n---\n\n")

        logs.info(f"Markdown saved: {md_path}")

        # Render PDF from Markdown
        write_pdf_from_markdown(md_path, pdf_path)
        logs.info(f"PDF saved: {pdf_path}")

        # Snapshot the requested config variables
        snapshot = build_config_snapshot(search_cfg, snapshot_paths)
        with cfg_snapshot_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)

        logs.info(f"Config snapshot saved: {cfg_snapshot_path}")
        logs.info(f"Finished processing config '{cfg_name}'.")


if __name__ == "__main__":
    main()