"""Add accuracy and MCC (raw and failed-run-imputed) to figures/_shared/tidy_kappa.csv.
Values come from the same P4/P5 result CSVs as tool_interaction_analysis.py. Usage:
python make_tidy_metrics.py <THESIS_BASE> <OUT_DIR> <shared_tidy_kappa.csv>"""
import sys
SHARED = sys.argv[3]
exec(open(__file__.replace("make_tidy_metrics.py","_base.py")).read())
both = pd.concat([synth, real], ignore_index=True)[["arm","scenario","seed","tool","accuracy","mcc"]]
tk = pd.read_csv(SHARED)
m = tk.merge(both, on=["arm","scenario","seed","tool"], how="left", validate="one_to_one")
fail = m.run_status == "failed_zero"
m["accuracy_imp0"] = np.where(fail, 0.0, m.accuracy)
m["mcc_imp0"]      = np.where(fail, 0.0, m.mcc)
assert m.loc[~fail, ["accuracy","mcc"]].notna().all().all(), "missing metrics for ok runs"
out = SHARED.replace("tidy_kappa.csv", "tidy_kappa_metrics.csv")
m.to_csv(out, index=False); print("wrote", out, m.shape)
