"""
IR System setup script.

Installs all required packages, downloads NLTK corpora and the spaCy
English model, verifies every import, writes a pinned requirements.txt,
and runs a final core-package smoke test.

Usage:
    python setup.py
"""

import importlib
import subprocess
import sys
import os

# ---------------------------------------------------------------------------
# Bootstrap colorama first — everything else uses it for coloured output.
# ---------------------------------------------------------------------------

def _pip(*args: str) -> bool:
    """Run a pip command. Returns True on success, False on failure."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False


print("Bootstrapping colorama…")
_pip("colorama")

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


def ok(msg: str)  -> None: print(f"{GREEN}{BOLD}  ✔  {RESET}{msg}")
def err(msg: str) -> None: print(f"{RED}{BOLD}  ✘  {RESET}{RED}{msg}{RESET}")
def info(msg: str)-> None: print(f"{YELLOW}  →  {RESET}{msg}")
def hdr(msg: str) -> None: print(f"\n{CYAN}{BOLD}{msg}{RESET}")


# ---------------------------------------------------------------------------
# Package definitions
# ---------------------------------------------------------------------------
# Each entry is either:
#   (install_arg, import_name)          — simple case
#   (install_arg, import_name, version_attr)  — custom __version__ attribute
# install_arg is what gets passed to pip; import_name is what Python imports.

PACKAGES = [
    # ── Core ──────────────────────────────────────────────────────────────────
    ("numpy",                   "numpy"),
    ("pandas",                  "pandas"),
    ("scipy",                   "scipy"),
    ("scikit-learn",            "sklearn"),
    ("tqdm",                    "tqdm"),
    ("python-dotenv",           "dotenv"),
    ("pydantic",                "pydantic"),
    ("httpx",                   "httpx"),
    ("requests",                "requests"),

    # ── IR & NLP ───────────────────────────────────────────────────────────────
    ("ir-datasets",             "ir_datasets"),
    ("rank-bm25",               "rank_bm25"),
    ("nltk",                    "nltk"),
    ("spacy",                   "spacy"),

    # ── Embeddings & Deep Learning ─────────────────────────────────────────────
    ("sentence-transformers",   "sentence_transformers"),
    ("transformers",            "transformers"),
    # torch CPU wheel — no version pin so pip accepts any installed CPU build
    (
        "torch",
        "torch",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ),
    ("faiss-cpu",               "faiss"),

    # ── FastAPI & Server ────────────────────────────────────────────────────────
    ("fastapi",                 "fastapi"),
    ("uvicorn[standard]",       "uvicorn"),
    ("python-multipart",        "multipart"),

    # ── Spell Check ────────────────────────────────────────────────────────────
    ("pyspellchecker",          "spellchecker"),

    # ── Evaluation & Reporting ─────────────────────────────────────────────────
    ("pytrec-eval-terrier",     "pytrec_eval"),
    ("matplotlib",              "matplotlib"),
    ("seaborn",                 "seaborn"),
    ("tabulate",                "tabulate"),

    # ── Utilities ──────────────────────────────────────────────────────────────
    ("psutil",                  "psutil"),
    ("joblib",                  "joblib"),
    ("aiofiles",                "aiofiles"),
    ("loguru",                  "loguru"),
]

NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "omw-1.4",
]

TOTAL = len(PACKAGES)

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

install_failures: list[tuple[str, str]] = []   # (package, reason)
import_failures:  list[tuple[str, str]] = []
install_ok:  list[str] = []
import_ok:   list[str] = []


def _get_version(module) -> str:
    """Try common version attribute names; return '?' if none found."""
    for attr in ("__version__", "version", "VERSION", "__VERSION__"):
        v = getattr(module, attr, None)
        if isinstance(v, str):
            return v
    try:
        import importlib.metadata as meta
        return meta.version(module.__name__.split(".")[0])
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# 1. Install packages
# ---------------------------------------------------------------------------

hdr("=" * 60)
hdr("  IR SYSTEM — DEPENDENCY INSTALLER")
hdr("=" * 60)

hdr("STEP 1/4  Installing packages")
print()

for idx, entry in enumerate(PACKAGES, start=1):
    # Unpack entry: may have extra pip flags after the import name.
    pip_pkg    = entry[0]
    import_name = entry[1]
    extra_flags = list(entry[2:])    # e.g. ["--index-url", "..."]

    display_name = pip_pkg.split("==")[0]   # strip version for display
    info(f"[{idx:2d}/{TOTAL}] Installing {display_name}…")

    pip_args = [pip_pkg, *extra_flags]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *pip_args],
            stderr=subprocess.PIPE,
        )
        ok(f"{display_name} installed")
        install_ok.append(display_name)
    except subprocess.CalledProcessError as exc:
        reason = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
        reason_short = reason[-1] if reason else "pip returned non-zero exit status"
        err(f"{display_name}: {reason_short}")
        install_failures.append((display_name, reason_short))


# ---------------------------------------------------------------------------
# 2. Download NLTK data
# ---------------------------------------------------------------------------

hdr("STEP 2/4  Downloading NLTK data")
print()

try:
    import nltk  # noqa: E402
    for resource in NLTK_RESOURCES:
        info(f"Downloading NLTK corpus: {resource}…")
        try:
            nltk.download(resource, quiet=True)
            ok(f"nltk:{resource}")
        except Exception as exc:
            err(f"nltk:{resource} — {exc}")
except ImportError:
    err("nltk not available; skipping NLTK data download.")


# ---------------------------------------------------------------------------
# 3. Download spaCy model
# ---------------------------------------------------------------------------

hdr("STEP 3/4  Downloading spaCy model")
print()

info("Downloading en_core_web_sm…")
try:
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--quiet"],
        stderr=subprocess.PIPE,
    )
    ok("spacy model en_core_web_sm downloaded")
except subprocess.CalledProcessError as exc:
    reason = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
    err(f"spacy download failed: {reason[-1] if reason else 'unknown error'}")


# ---------------------------------------------------------------------------
# 4. Verify imports + collect versions
# ---------------------------------------------------------------------------

hdr("STEP 4/4  Verifying package imports")
print()

for entry in PACKAGES:
    pip_pkg     = entry[0]
    import_name = entry[1]
    display_name = pip_pkg.split("==")[0]

    try:
        module = importlib.import_module(import_name)
        version = _get_version(module)
        ok(f"{display_name:<30}  {CYAN}v{version}{RESET}")
        import_ok.append((display_name, version))
    except ImportError as exc:
        err(f"{display_name:<30}  import failed: {exc}")
        import_failures.append((display_name, str(exc)))


# ---------------------------------------------------------------------------
# 5. Write requirements.txt (pinned)
# ---------------------------------------------------------------------------

hdr("Writing requirements.txt")
print()

req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
info(f"Running pip freeze → {req_path}…")

try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    with open(req_path, "w", encoding="utf-8") as fh:
        fh.write(result.stdout)
    line_count = result.stdout.count("\n")
    ok(f"requirements.txt written ({line_count} pinned packages)")
except Exception as exc:
    err(f"Could not write requirements.txt: {exc}")


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

hdr("=" * 60)
hdr("  SUMMARY")
hdr("=" * 60)
print()

n_install_ok  = len(install_ok)
n_install_fail = len(install_failures)
n_import_ok   = len(import_ok)
n_import_fail  = len(import_failures)

print(f"  Packages installed:  {GREEN}{BOLD}{n_install_ok}{RESET} / {TOTAL}")
print(f"  Import checks:       {GREEN}{BOLD}{n_import_ok}{RESET} / {TOTAL}")

if install_failures:
    print(f"\n  {RED}{BOLD}Install failures ({n_install_fail}):{RESET}")
    for pkg, reason in install_failures:
        print(f"    {RED}• {pkg}: {reason}{RESET}")

if import_failures:
    print(f"\n  {RED}{BOLD}Import failures ({n_import_fail}):{RESET}")
    for pkg, reason in import_failures:
        print(f"    {RED}• {pkg}: {reason}{RESET}")

if not install_failures and not import_failures:
    print(f"\n  {GREEN}{BOLD}All packages installed and verified successfully!{RESET}")

print()

# ---------------------------------------------------------------------------
# 7. Core verification smoke test
# ---------------------------------------------------------------------------

hdr("=" * 60)
hdr("  CORE PACKAGE VERIFICATION")
hdr("=" * 60)
print()

CORE_CHECKS = [
    "ir_datasets",
    "rank_bm25",
    "sentence_transformers",
    "faiss",
    "fastapi",
    "nltk",
    "spacy",
]

smoke_failures = []

for mod_name in CORE_CHECKS:
    try:
        mod = importlib.import_module(mod_name)
        version = _get_version(mod)
        ok(f"{mod_name:<30}  {CYAN}v{version}{RESET}")
    except ImportError as exc:
        err(f"{mod_name}: {exc}")
        smoke_failures.append(mod_name)

print()

if smoke_failures:
    print(f"{RED}{BOLD}VERIFICATION FAILED — missing core packages:{RESET}")
    for m in smoke_failures:
        print(f"  {RED}• {m}{RESET}")
    sys.exit(1)
else:
    print(f"{GREEN}{BOLD}ALL CORE PACKAGES VERIFIED SUCCESSFULLY{RESET}")
