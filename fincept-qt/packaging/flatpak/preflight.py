"""Offline pre-flight for the Flathub submission.

Checks the subset of flatpak-builder-lint / AppStream rules that can be
verified WITHOUT a Linux box, so the real lint run passes first time:

  * every git source is pinned (Flathub: a tag without a commit is a hard
    error for new submissions -- tags are mutable)
  * the ref the manifest builds actually EXISTS on the remote, and its commit
    matches the tag (the manifest once pointed at an untagged v4.5.0, so the
    build could never have succeeded)
  * screenshots are pinned to a tag/commit, not a mutable branch
  * the newest <release> matches the version the manifest builds, so Flathub
    cannot advertise a version whose binary it does not have
  * the desktop entry is internally consistent with the manifest and declares
    no MIME type or field code the app cannot honour

Run:  python preflight.py       (needs pyyaml; exits non-zero on any failure)

It does NOT replace flatpak-builder-lint -- run that on Linux too. See
FLATHUB.md.
"""
import io, os, re, sys, subprocess, xml.dom.minidom
import yaml

D = os.path.dirname(os.path.abspath(__file__))
# <repo>/fincept-qt/packaging/flatpak -> <repo>
REPO = os.path.abspath(os.path.join(D, "..", "..", ".."))
APPID = "in.fincept.FinceptTerminal"
fails, warns = [], []

def ck(ok, label, hard=True):
    print(f"  [{'PASS' if ok else ('FAIL' if hard else 'WARN')}] {label}")
    if not ok:
        (fails if hard else warns).append(label)

print("[1] manifest YAML")
man = yaml.safe_load(io.open(os.path.join(D, f"{APPID}.yml"), encoding="utf-8"))
ck(man["id"] == APPID, f"id matches filename ({APPID})")
ck(bool(man.get("command")), "command declared")
ck(man.get("runtime") and man.get("sdk"), "runtime + sdk declared")

print("[2] git sources pinned (Flathub: tag without commit = ERROR for new submissions)")
srcs = [(m.get("name", "?"), s) for m in man["modules"] if isinstance(m, dict)
        for s in m.get("sources", []) if isinstance(s, dict) and s.get("type") == "git"]
ck(len(srcs) > 0, f"found {len(srcs)} git sources")
for name, s in srcs:
    url = s["url"].rsplit("/", 1)[-1]
    if "tag" in s:
        ck("commit" in s, f"{name}:{url} has tag -> must also have commit")
    else:
        ck("commit" in s, f"{name}:{url} pinned by commit")
    if "commit" in s:
        ck(bool(re.fullmatch(r"[0-9a-f]{40}", s["commit"])), f"{name}:{url} commit is full 40-char SHA")

print("[3] built ref actually exists on the remote")
for name, s in srcs:
    if "FinceptTerminal.git" in s["url"]:
        tag = s.get("tag")
        r = subprocess.run(["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
                           cwd=REPO, capture_output=True, text=True)
        ck(bool(r.stdout.strip()), f"tag {tag} exists on origin")
        if s.get("commit"):
            r2 = subprocess.run(["git", "rev-parse", f"{tag}^{{commit}}"], cwd=REPO,
                                capture_output=True, text=True)
            ck(r2.stdout.strip() == s["commit"], f"commit matches {tag} ({r2.stdout.strip()[:8]})")

print("[4] metainfo XML")
mp = os.path.join(D, f"{APPID}.metainfo.xml")
dom = xml.dom.minidom.parse(mp)
xt = io.open(mp, encoding="utf-8").read()
ck(True, "well-formed XML")
def one(tag):
    n = dom.getElementsByTagName(tag)
    return n[0].firstChild.nodeValue.strip() if n and n[0].firstChild else None
ck(one("id") == APPID, "metainfo id matches app id")
for t in ("name", "summary", "metadata_license", "project_license"):
    ck(bool(one(t)), f"<{t}> present")
ck(len(dom.getElementsByTagName("screenshot")) > 0, "has screenshots")
ck(bool(dom.getElementsByTagName("content_rating")), "has content_rating (OARS)")
launch = dom.getElementsByTagName("launchable")
ck(bool(launch) and launch[0].firstChild.nodeValue.strip() == f"{APPID}.desktop",
   "launchable points at the desktop id")

print("[5] screenshots not on a mutable branch")
imgs = [n.firstChild.nodeValue.strip() for n in dom.getElementsByTagName("image") if n.firstChild]
ck(len(imgs) > 0, f"{len(imgs)} screenshot URLs")
for u in imgs:
    ck("/main/" not in u and "/master/" not in u, f"pinned (not a branch): .../{u.rsplit('/',1)[-1]}")

print("[6] newest listed release matches what the manifest builds")
rels = [r.getAttribute("version") for r in dom.getElementsByTagName("release")]
built = [s.get("tag") for _, s in srcs if "FinceptTerminal.git" in s["url"]][0].lstrip("v")
ck(bool(rels), f"releases listed: {rels[:3]}")
ck(rels and rels[0] == built, f"newest release {rels[0] if rels else None} == built version {built}")
for r in dom.getElementsByTagName("release"):
    ck(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", r.getAttribute("date"))),
       f"release {r.getAttribute('version')} date is ISO-8601")

print("[7] desktop entry")
dp = os.path.join(D, f"{APPID}.desktop")
kv = dict(l.split("=", 1) for l in io.open(dp, encoding="utf-8").read().splitlines()
          if "=" in l and not l.startswith("["))
ck(kv.get("Type") == "Application", "Type=Application")
ck(bool(kv.get("Name")), "Name present")
ck(kv.get("Icon") == APPID, f"Icon == app id ({APPID})")
ck(kv.get("Exec", "").split()[0] == man["command"], "Exec binary == manifest command")
cats = [c for c in kv.get("Categories", "").split(";") if c]
MAIN = {"AudioVideo","Audio","Video","Development","Education","Game","Graphics",
        "Network","Office","Science","Settings","System","Utility"}
ck(bool(set(cats) & MAIN), f"has a main category ({sorted(set(cats) & MAIN)})")
ck("MimeType" not in kv or bool(kv.get("MimeType")), "no empty MimeType")
if "MimeType" in kv:
    ck(False, "declares MimeType with no shared-mime-info XML installed", hard=False)
ck("%" not in kv.get("Exec", ""), "Exec has no field code the app cannot honour")

print("[8] installed assets exist")
ck(os.path.isfile(os.path.join(REPO, "fincept-qt", "resources", f"{APPID}.png")), "256x256 icon present")
for u in imgs:
    rel = u.split("/images/")[-1]
    ck(os.path.isfile(os.path.join(REPO, "images", rel)), f"screenshot source images/{rel}")

print(f"\n{len(fails)} failure(s), {len(warns)} warning(s)")
if fails:
    for f in fails: print("  FAIL:", f)
if warns:
    for w in warns: print("  WARN:", w)
sys.exit(1 if fails else 0)
