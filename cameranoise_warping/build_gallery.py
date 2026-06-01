import argparse
import random
from html import escape
from pathlib import Path


def read_yaml_root(path):
    if not path:
        return None
    try:
        import yaml
    except ImportError:
        raise SystemExit("Using --config requires PyYAML: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("data_saved_root")


def resolve(path, base=None):
    p = Path(path).expanduser()
    return (p if p.is_absolute() or base is None else Path(base) / p).resolve()


def href(path, html_path):
    path = Path(path)
    if not path.exists():
        return ""
    path = path.resolve()
    html_dir = html_path.parent.resolve()
    try:
        return path.relative_to(html_dir).as_posix()
    except ValueError:
        return path.as_uri()


def npy_shape(path):
    try:
        import numpy as np
        return " × ".join(map(str, np.load(path, mmap_mode="r").shape))
    except Exception:
        return "unknown"


def collect(data_root, html_path):
    data_root = Path(data_root)
    noises_dir = data_root / "noises"
    items = []
    for idx, noise_path in enumerate(sorted(noises_dir.glob("*_noises.npy"))):
        name = noise_path.name.removesuffix("_noises.npy")
        video_path = noises_dir / f"{name}_visualization.mp4"
        flow_path = data_root / "flows" / f"{name}_flows.npy"
        items.append({
            "idx": idx,
            "id": name,
            "shape": npy_shape(noise_path),
            "video": href(video_path, html_path),
            "noise": href(noise_path, html_path),
            "flow": href(flow_path, html_path),
            "noise_path": str(noise_path),
            "flow_path": str(flow_path) if flow_path.exists() else "not found",
            "intrinsic": str(data_root / "camerapose" / name / "intrinsic.pt"),
            "extrinsic": str(data_root / "camerapose" / name / "extrinsic.pt"),
        })
    return items


def pick(items, max_items, mode, seed):
    if max_items <= 0 or len(items) <= max_items:
        return items
    if mode == "first":
        return items[:max_items]
    if mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(items, max_items), key=lambda x: x["idx"])
    step = (len(items) - 1) / max(max_items - 1, 1)
    return [items[round(i * step)] for i in range(max_items)]


def video_html(src):
    if not src:
        return '<div class="missing">No video</div>'
    return f'<video controls muted loop preload="metadata"><source src="{escape(src)}" type="video/mp4"></video>'


def link(src, text):
    return f'<a href="{escape(src)}" download>{escape(text)}</a>' if src else ""


def card(item):
    links = "".join([link(item["video"], "MP4"), link(item["noise"], "Noise"), link(item["flow"], "Flow")])
    links = links or '<span class="empty-link">No files</span>'
    return f'''
    <article class="card" data-id="{escape(item['id']).lower()}">
      <div class="media">{video_html(item['video'])}</div>
      <div class="content">
        <p class="index">Sample {item['idx']:03d}</p>
        <h2>{escape(item['id'])}</h2>
        <p class="shape">Noise shape: <b>{escape(item['shape'])}</b></p>
        <div class="links">{links}</div>
        <details>
          <summary>Paths</summary>
          <pre>Noise: {escape(item['noise_path'])}\nFlow: {escape(item['flow_path'])}\nIntrinsic: {escape(item['intrinsic'])}\nExtrinsic: {escape(item['extrinsic'])}</pre>
        </details>
      </div>
    </article>'''


def page(items, total, data_root, sample):
    cards = "\n".join(card(x) for x in items) or '<p class="empty-page">No *_noises.npy files found.</p>'
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CameraNoise Gallery</title>
<style>
:root {{
  --bg:#0b1020; --card:rgba(255,255,255,.08); --card2:rgba(255,255,255,.12);
  --text:#f8fafc; --muted:#94a3b8; --line:rgba(255,255,255,.14); --accent:#7dd3fc;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; color:var(--text); font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  background:radial-gradient(circle at 10% 0%,rgba(125,211,252,.25),transparent 28rem),
             radial-gradient(circle at 90% 0%,rgba(192,132,252,.20),transparent 32rem),var(--bg);
}}
main {{ width:min(1440px,calc(100vw - 36px)); margin:auto; padding:46px 0 64px; }}
.hero {{ display:flex; justify-content:space-between; gap:28px; align-items:end; margin-bottom:22px; }}
.kicker {{ margin:0 0 10px; color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }}
h1 {{ margin:0; font-size:clamp(36px,5vw,66px); line-height:.95; letter-spacing:-.055em; }}
.sub {{ max-width:820px; margin:16px 0 0; color:var(--muted); line-height:1.65; }}
.stats {{ display:grid; grid-template-columns:repeat(3,96px); border:1px solid var(--line); border-radius:24px; overflow:hidden; background:var(--card); backdrop-filter:blur(18px); }}
.stat {{ padding:18px; border-left:1px solid var(--line); }} .stat:first-child {{ border-left:0; }}
.stat b {{ display:block; font-size:28px; }} .stat span {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
.toolbar {{ position:sticky; top:0; z-index:2; display:flex; gap:12px; align-items:center; padding:14px 0 20px; backdrop-filter:blur(18px); }}
#search {{ width:min(520px,100%); height:46px; padding:0 16px; color:var(--text); background:rgba(15,23,42,.78); border:1px solid var(--line); border-radius:999px; outline:none; }}
.hint {{ color:var(--muted); font-size:13px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:18px; }}
.card {{ overflow:hidden; border:1px solid var(--line); border-radius:26px; background:var(--card); box-shadow:0 22px 60px rgba(0,0,0,.28); backdrop-filter:blur(18px); transition:.18s; }}
.card:hover {{ transform:translateY(-3px); background:var(--card2); border-color:rgba(125,211,252,.45); }}
.media {{ background:#020617; border-bottom:1px solid var(--line); }}
video,.missing {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#020617; }}
.missing {{ display:grid; place-items:center; color:var(--muted); }}
.content {{ padding:18px; }} .index {{ margin:0 0 5px; color:var(--accent); font-size:11px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }}
h2 {{ margin:0; font-size:18px; line-height:1.3; overflow-wrap:anywhere; }}
.shape {{ color:var(--muted); font-size:13px; }}
.links {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
.links a {{ color:var(--text); text-decoration:none; border:1px solid var(--line); border-radius:999px; padding:8px 12px; font-size:12px; font-weight:700; background:rgba(255,255,255,.08); }}
details {{ margin-top:14px; color:var(--muted); font-size:12px; }} summary {{ cursor:pointer; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:#cbd5e1; background:rgba(2,6,23,.45); border-radius:14px; padding:12px; }}
.empty-page {{ color:var(--muted); }}
@media (max-width:820px) {{ main{{width:min(100vw - 24px,1440px);padding-top:30px}} .hero{{display:block}} .stats{{margin-top:22px;grid-template-columns:repeat(3,1fr)}} .toolbar{{position:static;display:grid}} .grid{{grid-template-columns:1fr}} }}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <h1>CameraNoise</h1>
      <p class="sub">Generated from <strong>{escape(str(data_root))}</strong>. Search by sample id, preview local videos, and open related files.</p>
    </div>
    <div class="stats">
      <div class="stat"><b id="visible">{len(items)}</b><span>visible</span></div>
      <div class="stat"><b>{total}</b><span>total</span></div>
      <div class="stat"><b>{escape(sample)}</b><span>sample</span></div>
    </div>
  </section>
  <section class="toolbar">
    <input id="search" type="search" placeholder="Search sample id..." autocomplete="off">
  </section>
  <section class="grid">{cards}</section>
</main>
<script>
const q=document.querySelector('#search'), cards=[...document.querySelectorAll('.card')], visible=document.querySelector('#visible');
q.addEventListener('input',()=>{{let n=0,s=q.value.trim().toLowerCase();cards.forEach(c=>{{let ok=c.dataset.id.includes(s);c.hidden=!ok;n+=ok}});visible.textContent=n;}});
</script>
</body>
</html>'''


def build_gallery(data_root, output=None, max_items=24, sample="even", seed=0):
    data_root = resolve(data_root)
    html_path = resolve(output) if output else data_root / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = collect(data_root, html_path)
    shown_items = pick(all_items, max_items, sample, seed)
    html_path.write_text(page(shown_items, len(all_items), data_root, sample), encoding="utf-8")
    return {
        "html_path": html_path,
        "shown": len(shown_items),
        "total": len(all_items),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a clean static HTML gallery for CameraNoise results.")
    parser.add_argument("--data-root", type=Path, help="CameraNoise output root. Overrides data_saved_root in config.")
    parser.add_argument("--config", type=Path, help="Optional yaml with data_saved_root.")
    parser.add_argument("--output", type=Path, help="Output HTML. Default: data_root/index.html")
    parser.add_argument("--max-items", type=int, default=24, help="0 means show all.")
    parser.add_argument("--sample", choices=["first", "even", "random"], default="even")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_root = read_yaml_root(args.config)
    root = args.data_root or config_root
    if not root:
        raise SystemExit("Please provide --data-root or set data_saved_root in --config.")

    data_root = resolve(root, args.config.parent if args.config else None)
    result = build_gallery(
        data_root,
        output=args.output,
        max_items=args.max_items,
        sample=args.sample,
        seed=args.seed,
    )
    print(f"Gallery saved: {result['html_path']}")
    print(f"Shown: {result['shown']} / {result['total']}")


if __name__ == "__main__":
    main()
