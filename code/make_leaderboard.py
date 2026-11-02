"""Build the static AEB leaderboard from results/reference.json.

Main table: frozen EEG foundation-model encoders (released BIOT/LaBraM/CBraMod
checkpoints plus the six-dataset encoder). Supervised from-scratch models are a
separate reference table. Hand-crafted band-power is Appendix A, not a learned
encoder and not part of the main comparison.
"""
import argparse
import html
import json
from pathlib import Path

PRETTY = {
    "encoder_6datasets": "Encoder (6 datasets)",
    "biot": "BIOT",
    "labram_200": "LaBraM-200",
    "cbramod": "CBraMod",
    "bandpower_82x22": "Band-power 82×22",
    "bandpower_19x5": "Band-power 19×5",
    "eegnet": "EEGNet",
    "shallow": "ShallowConvNet",
    "deep4": "Deep4Net",
    "conformer": "EEG-Conformer",
}

FAMILY = {
    "domain_encoder": "frozen encoder",
    "pretrained_fm": "pretrained FM",
    "feature": "hand-crafted feature",
}

esc = html.escape


def fmt_params(v):
    if not isinstance(v, (int, float)) or v <= 0:
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return str(int(v))


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/reference.json")
    ap.add_argument("--out", default="leaderboard/index.html")
    args = ap.parse_args()
    d = json.load(open(args.results, encoding="utf-8"))

    frozen = d["frozen"]
    main_rows = sorted(
        [m for m in frozen if m.get("family") != "feature"],
        key=lambda m: -float(m["logistic"]),
    )
    feature_rows = sorted(
        [m for m in frozen if m.get("family") == "feature"],
        key=lambda m: -float(m["logistic"]),
    )

    main_trs = "\n".join(
        f"<tr><td>{esc(PRETTY.get(m['model'], m['model']))}</td>"
        f"<td>{esc(FAMILY.get(m.get('family', ''), m.get('family', '—')))}</td>"
        f"<td>{fmt_params(m.get('params'))}</td>"
        f"<td><b>{fmt(m.get('logistic'))}</b></td>"
        f"<td>{fmt(m.get('centroid'))}</td><td>frozen probe</td></tr>"
        for m in main_rows
    )

    sup_trs = "\n".join(
        f"<tr><td>{esc(PRETTY.get(m['model'], m['model']))}</td>"
        f"<td>{fmt_params(m.get('params'))}</td>"
        f"<td><b>{fmt(m.get('per_subject'))}</b></td>"
        f"<td>{fmt(m.get('pooled'))}</td><td>per-subject fine-tune</td></tr>"
        for m in d["finetune"]
    )

    feature_trs = "\n".join(
        f"<tr><td>{esc(PRETTY.get(m['model'], m['model']))}</td>"
        f"<td>{int(m['feature_dim']):,}</td>"
        f"<td><b>{fmt(m.get('logistic'))}</b></td>"
        f"<td>{fmt(m.get('centroid'))}</td><td>hand-crafted feature baseline</td></tr>"
        for m in feature_rows
    )

    excluded = "<br>".join(
        f"{esc(k.upper())} excluded: {esc(v)}" for k, v in d.get("excluded", {}).items()
    )

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Auditory EEG Bench — leaderboard v0.1</title>
<style>body{{font:16px/1.6 Georgia,serif;max-width:980px;margin:40px auto;padding:0 20px}}
h2{{font-size:20px;margin-top:30px}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:8px 10px;border-bottom:1px solid #ddd;text-align:left}}
th{{background:#f3f5f7}}code{{background:#eee;padding:1px 5px}}.warn{{color:#a33}}
.note{{color:#5a6478;font-size:14px}}</style></head>
<body><h1>Auditory EEG Bench — leaderboard v0.1</h1>
<p>NMED-T 10-way, per-subject trial-wise 50/50; chance 0.10. Frozen probes and
fine-tuning are separate protocols. <span class="warn">Reference results; unified 3-seed
grid and full LODO are pending.</span></p>

<h2>Frozen EEG foundation-model encoders</h2>
<p class="note">Our six-dataset encoder compared with released BIOT, LaBraM and CBraMod
checkpoints under the same per-subject logistic probe.</p>
<table><thead><tr><th>model</th><th>family</th><th>params</th><th>10-way</th><th>centroid</th><th>protocol</th></tr></thead>
<tbody>{main_trs}</tbody></table>
<p class="note">{excluded}</p>

<h2>Supervised from-scratch reference</h2>
<p class="note">Not foundation-model encoders and not part of the main comparison;
fixed-budget reference points, not tuned upper bounds.</p>
<table><thead><tr><th>model</th><th>params</th><th>per-subject</th><th>pooled</th><th>protocol</th></tr></thead>
<tbody>{sup_trs}</tbody></table>

<h2>Appendix A — Hand-crafted feature baseline</h2>
<p class="note">Not a learned encoder and not part of the main comparison. Same splits;
reported for completeness because it is a strong closed-set label feature.</p>
<table><thead><tr><th>feature</th><th>dim</th><th>10-way</th><th>centroid</th><th>note</th></tr></thead>
<tbody>{feature_trs}</tbody></table>
<p class="note">Split ladder (band-power feature control): random rows .896 /
blocks .873 / subjects .106.</p>

<p class="note">Other controls: answer-conditioned zero/shuffled EEG still 1.000 pair
accuracy. See <code>README.md</code> and <code>manifests/split_audit.json</code>.
No raw EEG or checkpoints are released.</p>
</body></html>"""
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page, encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
