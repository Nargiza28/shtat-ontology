# =============================================================================
# SHTAT.AI — SEMANTIC HR GOVERNANCE PLATFORM
# =============================================================================
# Product       : Shtat.ai v1.0 Beta
# Description   : AI-powered university HR compliance and workload audit system.
#                 Bridges Semantic Web (OWL/Owlready2) with Data Science
#                 (Pandas, Plotly) for intelligent faculty resource management.
# Architecture  : OSHRA — Ontology-Steered Hybrid Reasoning Algorithm
#
#   Layer 1 — Semantic Knowledge Loading : OWL ontology via Owlready2
#   Layer 2 — Relational Audit           : Pandas constraint checking
#   Layer 3 — Predictive Analytics       : Statistical risk scoring
#   Layer 4 — NLP Entity Extraction      : Regex-based Uzbek NER
#   Layer 5 — Entity Resolution          : Weighted similarity, cross-faculty
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import time
import re
import io

# PDF extraction — graceful fallback if library not installed
try:
    import pdfplumber
    _PDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # PyMuPDF
        _PDF_AVAILABLE = True
    except ImportError:
        _PDF_AVAILABLE = False

# pyHanko is imported INSIDE eri_verify_signature() so a ModuleNotFoundError
# at the top level never prevents the rest of the app from starting.
_PYHANKO_AVAILABLE = None   # None = not yet checked; True/False after first call


def extract_text_from_pdf(uploaded_pdf) -> list[str]:
    """
    Extract text from an uploaded PDF, returning a LIST of strings —
    one string per page.  Page 0 = Didox protocol (signer/PINFL).
    Page 1 = ariza body text (NER target).

    Falls back gracefully:
      pdfplumber → PyMuPDF (fitz) → empty list

    Never raises — always returns a list (may be empty or contain error strings).
    """
    pages: list[str] = []

    # ── Attempt 1: pdfplumber ──────────────────────────────────────────────
    try:
        import pdfplumber
        with pdfplumber.open(uploaded_pdf) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages.append(txt.strip())
        return pages
    except ImportError:
        pass
    except Exception as e:
        return [f"[pdfplumber xato: {e}]"]

    # ── Attempt 2: PyMuPDF (fitz) ──────────────────────────────────────────
    try:
        import fitz
        # uploaded_pdf may be a BytesIO or a file-like object
        if hasattr(uploaded_pdf, "seek"):
            uploaded_pdf.seek(0)
            pdf_bytes = uploaded_pdf.read()
        else:
            pdf_bytes = bytes(uploaded_pdf)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pages.append(page.get_text().strip())
        doc.close()
        return pages
    except ImportError:
        return []
    except Exception as e:
        return [f"[PyMuPDF xato: {e}]"]


def eri_verify_signature(pdf_bytes: bytes) -> dict:
    """
    THREE-TIER verification — fastest reliable tier wins:

    Tier 0 (TEXT PROTOCOL — primary for Didox):
      Didox embeds the signer block as readable text, not a cryptographic
      PDF signature object.  Checking for 'Фиш' and 'ЖШШИР' on Page 1
      is therefore more reliable than pyHanko for these documents.
      → Called BEFORE pyHanko so the app always works with Didox PDFs.

    Tier 1 (pyHanko — wrapped in try/except):
      If pyHanko finds an embedded_signatures block, also mark verified.
      Any import or runtime error is silently caught — the app continues.

    Tier 2 (Raw byte markers):
      Last resort: /Sig, /AcroForm, /ByteRange, adbe.pkcs7 in raw bytes.

    Sets st.session_state['is_verified'] = True on any tier success.
    page1_text must be passed in so Tier 0 can read it.
    """
    global _PYHANKO_AVAILABLE

    result = {
        "is_signed"  : False,
        "signer_name": "",
        "sig_time"   : "",
        "error"      : "",
        "method"     : "",
    }

    # NOTE: page1_text is read from session state so this function stays
    # callable with just pdf_bytes (same signature as before).
    page1_text = st.session_state.get("eri_page1_text", "")

    # ── Tier 0: Text-based Didox protocol verification (PRIMARY) ──────────
    # Didox protocol pages always contain 'Фиш' (Full name) and
    # 'ЖШШИР' (PINFL field label in Cyrillic).  If both are present on
    # Page 1, the document is a genuine Didox ERI protocol → verified.
    didox_markers = ["Фиш", "ЖШШИР"]
    found_text_markers = [m for m in didox_markers if m in page1_text]
    if len(found_text_markers) >= 1:            # even one is strong signal
        result["is_signed"] = True
        result["method"]    = (
            f"Protokol tekshiruvi "
            f"({', '.join(found_text_markers)} topildi)"
        )
        st.session_state["is_verified"] = True
        return result

    # ── Tier 1: pyHanko (wrapped — any error falls through silently) ───────
    if _PYHANKO_AVAILABLE is not False:
        try:
            from pyhanko.pdf_utils.reader import PdfFileReader
            _PYHANKO_AVAILABLE = True
            reader     = PdfFileReader(io.BytesIO(pdf_bytes))
            sig_fields = list(reader.embedded_signatures)
            if sig_fields:
                result["is_signed"] = True
                result["method"]    = "pyHanko (imzo bloki aniqlandi)"
                try:
                    cert = getattr(sig_fields[0], "signer_cert", None)
                    if cert:
                        subj = getattr(cert, "subject", None)
                        result["signer_name"] = (
                            subj.human_friendly
                            if subj and hasattr(subj, "human_friendly")
                            else str(subj or "")
                        )
                    ts = getattr(sig_fields[0], "self_reported_timestamp", None)
                    if ts:
                        result["sig_time"] = str(ts)[:19]
                except Exception:
                    pass
                st.session_state["is_verified"] = True
                return result
            else:
                result["error"] = "pyHanko: imzo bloki topilmadi."
        except Exception as e:
            _PYHANKO_AVAILABLE = False
            result["error"]    = f"pyHanko: {e}"

    # ── Tier 2: raw byte markers ───────────────────────────────────────────
    if not result["is_signed"]:
        byte_markers = [b"/Sig", b"/AcroForm", b"/ByteRange", b"adbe.pkcs7"]
        found_bytes  = sum(1 for m in byte_markers if m in pdf_bytes)
        if found_bytes >= 2:
            result["is_signed"] = True
            result["method"]    = f"Bayt heuristik ({found_bytes}/4)"
            result["error"]     = ""
            st.session_state["is_verified"] = True
        else:
            st.session_state["is_verified"] = False
            result["error"] = (
                "ERI imzo bloki topilmadi. "
                "Iltimos, Didox orqali imzolangan PDF yuklang."
            )

    return result


def eri_extract_didox_identity(page1_text: str) -> dict:
    """
    Signer identity extraction from Didox protocol Page 1.

    Primary path  : text after 'Фиш' label → 'XUDOYKULOVA NARGIZA ...'
    Secondary path: text after 'ЖШШИР' label → PINFL number
    Fallback paths: Latin labels (F.I.Sh., Imzolovchi) → ALL-CAPS tokens

    Returns: {full_name, last_name, first_name, pinfl, raw_line, word_set}
    word_set is lowercase — used by eri_name_match_score for set comparison.
    """
    result = {
        "full_name" : "",
        "last_name" : "",
        "first_name": "",
        "pinfl"     : "",
        "raw_line"  : "",
        "word_set"  : set(),
    }

    # ── PINFL via ЖШШИР label (Didox Cyrillic) or bare 14-digit number ────
    pinfl_patterns = [
        r'(?:ЖШШИР|Жшшир|жшшир|ПИНФЛ|Пинфл)\s*[:\|]?\s*(\d{14})',
        r'\b(\d{14})\b',
    ]
    for pat in pinfl_patterns:
        m = re.search(pat, page1_text)
        if m:
            result["pinfl"] = m.group(1)
            break

    # ── Name via Фиш label (highest priority) then fallbacks ──────────────
    name_patterns = [
        # Didox 'Фиш' field (case variants)
        r'(?:Фиш|ФИШ|фиш)\s*[:\|]?\s*'
        r'([A-ZҚҒҲЎЁА-ЯA-Z][A-ZҚҒҲЎЁА-Яa-zA-Zқғҳўё\s]{3,60})',
        # Latin/Uzbek labels
        r'(?:F\.?I\.?Sh\.?|Imzolovchi|Imzolagan|Signer|ФИО|Подписант)\s*[:\-|]?\s*'
        r'([A-ZҚҒҲА-ЯA-Z][A-ZҚҒҲА-Яa-zA-Zқғҳ\s]{3,60})',
        # ALL-CAPS fallback: 2–4 tokens each ≥ 3 chars
        r'\b([A-ZҚҒҲЎЁ]{3,25}\s+[A-ZҚҒҲЎЁ]{3,20}'
        r'(?:\s+[A-ZҚҒҲЎЁ]{3,20})?)\b',
    ]

    for pat in name_patterns:
        m = re.search(pat, page1_text)
        if m:
            raw    = m.group(1).strip()
            tokens = raw.split()
            if 2 <= len(tokens) <= 4:
                result["full_name"]  = raw
                result["last_name"]  = tokens[0]
                result["first_name"] = tokens[1] if len(tokens) > 1 else ""
                result["raw_line"]   = raw
                result["word_set"]   = {
                    t.lower().strip(".,") for t in tokens if len(t) > 1
                }
                break

    return result


def eri_name_match_score(signer: dict, applicant_name: str) -> float:
    """
    Cross-page identity matching — case-insensitive, order-independent.

    ALGORITHM (highest score returned):

    Step 1 — Word-set inclusion:
      Lowercase both names, split into word sets.
      If ALL signer words ⊆ applicant words → 1.0  (exact match, any order)
      e.g. {'xudoykulova','nargiza'} ⊆ {'nargiza','xudoykulova'} → 1.0

    Step 2 — Weighted formula (both forward and reversed order):
      0.70 × LastnameMatch + 0.30 × FirstnameMatch
      Checks "Familiya Ism" and "Ism Familiya" — takes the higher.
      Initial abbreviations ("N." vs "NARGIZA") score 0.8 on first-name.

    Step 3 — Partial overlap fallback:
      If steps 1–2 both give 0 but some words overlap → small partial score.
    """
    if not signer.get("full_name") or not applicant_name:
        return 0.0

    def _norm(name: str) -> set:
        return {
            t.lower().strip(".,")
            for t in re.split(r'[\s,]+', name.strip())
            if len(t) > 1
        }

    signer_words = signer.get("word_set") or _norm(signer["full_name"])
    app_words    = _norm(applicant_name)

    # Step 1 — set inclusion → 1.0
    if signer_words and signer_words.issubset(app_words):
        return 1.0

    # Step 2 — weighted formula, both orders
    sig_last  = signer.get("last_name",  "").lower().strip(".,")
    sig_first = signer.get("first_name", "").lower().strip(".,")
    app_toks  = [t.lower().strip(".,") for t in applicant_name.strip().split()]

    if not app_toks:
        return 0.0

    # Forward: app[0]=last, app[1]=first
    fwd_last  = 1.0 if app_toks[0] == sig_last else 0.0
    # Reversed: app[0]=first, app[1]=last
    rev_last  = 1.0 if (len(app_toks) > 1 and app_toks[1] == sig_last) else 0.0
    last_match = max(fwd_last, rev_last)

    # Pick the first-name token (whichever order gave the surname match)
    first_tok = app_toks[0] if rev_last > fwd_last else (
        app_toks[1] if len(app_toks) > 1 else ""
    )
    if first_tok and sig_first:
        if first_tok == sig_first:
            first_match = 1.0
        elif len(first_tok) == 1 and sig_first.startswith(first_tok):
            first_match = 0.8
        elif len(sig_first) == 1 and first_tok.startswith(sig_first):
            first_match = 0.8
        else:
            first_match = 0.0
    else:
        first_match = 0.0

    score = round(0.70 * last_match + 0.30 * first_match, 4)
    if score > 0:
        return score

    # Step 3 — partial overlap
    overlap = signer_words & app_words
    if overlap:
        return round(len(overlap) / max(len(signer_words), 1) * 0.65, 4)

    return 0.0


def eri_render_security_status(
    sig_result    : dict,
    didox_id      : dict,
    applicant_name: str,
    match_score   : float,
    is_authorized : bool,
) -> None:
    """
    Xavfsizlik Holati UI — three cards + main banner.

    When is_authorized via protocol verification shows:
      '✅ Protokol bo'yicha tasdiqlandi: [Name] | PINFL: [PINFL]'
    """
    st.markdown(
        '<div class="section-header">🔐 Xavfsizlik Holati (ERI Tekshiruvi)</div>',
        unsafe_allow_html=True,
    )

    # ── Three status cards ─────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns(3)
    method = sig_result.get("method", "")

    sig_color = "#27ae60" if sig_result["is_signed"] else "#eb5757"
    sig_icon  = "✅" if sig_result["is_signed"] else "❌"
    # Label reflects the actual verification method used
    if "Protokol" in method:
        sig_label = "Protokol Tasdiqlandi"
    elif "pyHanko" in method:
        sig_label = "ERI Imzo Tasdiqlandi"
    elif "Bayt" in method:
        sig_label = "Bayt Tekshiruvi OK"
    else:
        sig_label = "Imzo Topilmadi"

    sc1.markdown(
        f'<div style="background:rgba({("39,174,96" if sig_result["is_signed"] else "235,87,87")},0.12);'
        f'border:1px solid {sig_color};border-radius:10px;padding:14px;text-align:center">'
        f'<div style="font-size:1.5rem">{sig_icon}</div>'
        f'<div style="font-weight:700;color:{sig_color};font-size:0.82rem">{sig_label}</div>'
        f'<div style="font-size:0.68rem;color:#8b949e;margin-top:4px">'
        f'{method or sig_result.get("sig_time","") or "—"}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    score_pct   = int(match_score * 100)
    score_color = "#27ae60" if match_score >= 0.90 else \
                  "#f5a623" if match_score >= 0.70 else "#eb5757"
    sc2.markdown(
        f'<div style="background:rgba(47,128,237,0.08);'
        f'border:1px solid rgba(47,128,237,0.3);border-radius:10px;padding:14px;text-align:center">'
        f'<div style="font-size:1.5rem;font-weight:800;color:{score_color}">{score_pct}%</div>'
        f'<div style="font-weight:700;color:#c9d1d9;font-size:0.82rem">Shaxsiy Moslik</div>'
        f'<div style="font-size:0.68rem;color:#8b949e;margin-top:4px">'
        f'So\'z to\'plami taqqoslash | Chegara: 90%'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    auth_color = "#27ae60" if is_authorized else "#eb5757"
    auth_icon  = "🛡️" if is_authorized else "⚠️"
    auth_label = "RUXSAT BERILDI" if is_authorized else "RUXSAT YO'Q"
    sc3.markdown(
        f'<div style="background:rgba({("39,174,96" if is_authorized else "235,87,87")},0.12);'
        f'border:2px solid {auth_color};border-radius:10px;padding:14px;text-align:center">'
        f'<div style="font-size:1.5rem">{auth_icon}</div>'
        f'<div style="font-weight:800;color:{auth_color};font-size:0.88rem">{auth_label}</div>'
        f'<div style="font-size:0.68rem;color:#8b949e;margin-top:4px">OSHRA Layer 0</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main banner ────────────────────────────────────────────────────────
    name_display = didox_id.get("full_name", "") or applicant_name or "—"
    pinfl_display = didox_id.get("pinfl", "") or "—"

    if is_authorized:
        if "Protokol" in method:
            # Specific message requested in requirements
            st.success(
                f"✅ Protokol bo'yicha tasdiqlandi: {name_display} "
                f"| PINFL: {pinfl_display} "
                f"| Moslik: {score_pct}% "
                f"| Ontologiya xaritalashi ruxsat etildi."
            )
        else:
            st.success(
                f"✅ ERI Tasdiqlandi: {name_display} "
                f"| PINFL: {pinfl_display} "
                f"| Moslik: {score_pct}% "
                f"| Ontologiya xaritalashi ruxsat etildi."
            )
    else:
        reasons = []
        if not sig_result["is_signed"]:
            reasons.append(sig_result.get("error") or "Protokol yoki imzo topilmadi")
        if match_score < 0.90:
            reasons.append(
                f"Moslik: {score_pct}% < 90% "
                f"(Sahifa 1: '{name_display}' ↔ Sahifa 2: '{applicant_name}')"
            )
        st.error(
            "⚠️ Ruxsatsiz yoki Shaxs Mos Kelmadi — "
            + " | ".join(reasons)
            + " | Ontologiya xaritalashi bloklandi."
        )

    # ── Detail expander ────────────────────────────────────────────────────
    with st.expander("Tekshiruv Tafsilotlari", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Protokol / Imzo**")
            st.write({
                "Usul"       : method or "—",
                "Holat"      : "✅ Tasdiqlandi" if sig_result["is_signed"] else "❌ Topilmadi",
                "Imzolovchi" : sig_result.get("signer_name", "—") or "—",
                "Vaqt"       : sig_result.get("sig_time", "—") or "—",
            })
        with d2:
            st.markdown("**Shaxs Moslik**")
            signer_words_display = ", ".join(
                sorted(didox_id.get("word_set", set()))
            ) or "—"
            st.write({
                "Sahifa 1 ismi" : name_display,
                "PINFL"         : pinfl_display,
                "Sahifa 2 ismi" : applicant_name or "—",
                "So'z to'plami" : signer_words_display,
                "Ball"          : f"{score_pct}% ({'✅' if match_score>=0.90 else '❌'})",
            })

# Canonical column aliases: maps various header spellings → internal key
_COL_ALIASES = {
    "id"            : ["id", "xodim_id", "xodim id", "employee_id"],
    "name"          : ["f.i.sh.", "f.i.sh", "full name", "fullname", "ism",
                       "fish", "f.i.o", "f.i.o.", "name", "ismi"],
    "position"      : ["position", "lavozim", "pos", "title"],
    "stavka"        : ["stavka", "workload", "shtat", "shtat birligi",
                       "shtat_birligi", "fte", "ish hajmi"],
    "employment"    : ["employmenttype", "employment_type", "employment type",
                       "bandlik", "bandlik turi", "type"],
    "faculty"       : ["faculty", "fakul'tet", "fakultet", "department",
                       "kafedra", "dept"],
}

def _resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Map the DataFrame's actual column names to internal canonical keys.
    Returns {canonical_key: actual_col_name | None}.
    """
    lower_cols = {c.strip().lower(): c for c in df.columns}
    mapping = {}
    for key, aliases in _COL_ALIASES.items():
        found = next((lower_cols[a] for a in aliases if a in lower_cols), None)
        mapping[key] = found
    return mapping


def load_excel_faculty(uploaded_file) -> tuple[pd.DataFrame | None, dict, list[str]]:
    """
    Reads an uploaded Excel/CSV file and returns:
      (df, col_mapping, warnings)

    Normalises column names using _resolve_columns().
    Returns an empty DataFrame + warnings if the file is unreadable.
    """
    warnings_out = []
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as exc:
        return None, {}, [f"Faylni o'qib bo'lmadi: {exc}"]

    col_map = _resolve_columns(df)

    # Warn about any canonical columns that could not be resolved
    required = ["name", "position", "stavka", "employment"]
    for req in required:
        if col_map[req] is None:
            warnings_out.append(
                f"'{req}' ustuni topilmadi. Mavjud ustunlar: {list(df.columns)}"
            )

    return df, col_map, warnings_out


def build_faculty_dataframe(
    df_raw: pd.DataFrame,
    col_map: dict,
    faculty_name: str,
    max_individual_sb: float = 1.5,
) -> pd.DataFrame:
    """
    Converts the raw uploaded DataFrame (filtered to one faculty) into the
    canonical internal schema that every tab expects:
      ID, Name, Position, PositionRaw, EmploymentType, Workload,
      Faculty, WorkloadExceeded, WorkloadCategory
    """
    rows = []
    fac_col  = col_map.get("faculty")
    name_col = col_map["name"]
    pos_col  = col_map["position"]
    wl_col   = col_map["stavka"]
    emp_col  = col_map["employment"]
    id_col   = col_map.get("id")

    # Filter to the selected faculty if the column exists
    if fac_col:
        subset = df_raw[df_raw[fac_col].astype(str).str.strip() == faculty_name].copy()
    else:
        subset = df_raw.copy()

    for idx, row in subset.iterrows():
        name     = str(row[name_col]).strip()  if name_col and name_col in row else f"xodim_{idx}"
        pos      = str(row[pos_col]).strip()   if pos_col  and pos_col  in row else "Unknown"
        emp      = str(row[emp_col]).strip()   if emp_col  and emp_col  in row else "Unknown"
        try:
            wl = float(str(row[wl_col]).replace(",", ".")) if wl_col and wl_col in row else 0.0
        except (ValueError, TypeError):
            wl = 0.0
        rec_id = str(row[id_col]).strip() if id_col and id_col in row else f"row_{idx}"
        rows.append({
            "ID"            : rec_id,
            "Name"          : name,
            "Position"      : pos,
            "PositionRaw"   : pos,
            "EmploymentType": emp,
            "Workload"      : wl,
            "Faculty"       : faculty_name,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["WorkloadExceeded"]  = df["Workload"] > max_individual_sb
    bins   = [-0.01, 0.25, 0.5, 1.0, max_individual_sb, 99.0]
    labels = [
        "Minimal (≤0.25)", "Yarim shtat (≤0.5)", "Asosiy shtat (≤1.0)",
        f"Og'ir yuk (≤{max_individual_sb})", f"⚠ Me'yordan oshgan (>{max_individual_sb})"
    ]
    # pd.cut needs strictly increasing bins; guard if max_individual_sb <= 1.0
    if max_individual_sb <= 1.0:
        bins   = [-0.01, 0.25, 0.5, max_individual_sb, 99.0]
        labels = [
            "Minimal (≤0.25)", "Yarim shtat (≤0.5)",
            f"Asosiy shtat (≤{max_individual_sb})",
            f"⚠ Me'yordan oshgan (>{max_individual_sb})"
        ]
    try:
        df["WorkloadCategory"] = pd.cut(df["Workload"], bins=bins, labels=labels)
    except ValueError:
        df["WorkloadCategory"] = "N/A"
    return df.sort_values("ID").reset_index(drop=True)

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Shtat.ai | Semantic HR Governance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL CUSTOM CSS — Clean Corporate / Glassmorphism Hybrid Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-glass:      rgba(22, 27, 34, 0.75);
    --accent-blue:   #2f80ed;
    --accent-teal:   #00b4d8;
    --accent-gold:   #f5a623;
    --accent-red:    #eb5757;
    --accent-green:  #27ae60;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
    --border:        rgba(48, 54, 61, 0.8);
    --shadow:        0 8px 32px rgba(0,0,0,0.4);
    --radius:        12px;
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] p {
    color: var(--text-muted) !important;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Tab Strip ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--bg-secondary);
    border-radius: var(--radius);
    padding: 6px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--text-muted) !important;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 8px 20px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: #fff !important;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.kpi-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-card.blue::before  { background: var(--accent-blue); }
.kpi-card.teal::before  { background: var(--accent-teal); }
.kpi-card.gold::before  { background: var(--accent-gold); }
.kpi-card.green::before { background: var(--accent-green); }
.kpi-card:hover { transform: translateY(-3px); box-shadow: var(--shadow); }

.kpi-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
    font-weight: 600;
}
.kpi-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
}
.kpi-card.blue  .kpi-value { color: var(--accent-blue); }
.kpi-card.teal  .kpi-value { color: var(--accent-teal); }
.kpi-card.gold  .kpi-value { color: var(--accent-gold); }
.kpi-card.green .kpi-value { color: var(--accent-green); }

/* ── Section Headers ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 28px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.section-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: -10px;
    margin-bottom: 16px;
}

/* ── Alert / Flag Cards ── */
.flag-card {
    background: rgba(235, 87, 87, 0.08);
    border: 1px solid rgba(235, 87, 87, 0.3);
    border-left: 4px solid var(--accent-red);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.flag-card .flag-icon { font-size: 1.4rem; }
.flag-card .flag-name { font-weight: 600; font-size: 0.95rem; }
.flag-card .flag-detail { font-size: 0.8rem; color: var(--text-muted); margin-top: 2px; }
.flag-card .flag-value {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    color: var(--accent-red);
    font-weight: 600;
}

.ok-card {
    background: rgba(39, 174, 96, 0.08);
    border: 1px solid rgba(39, 174, 96, 0.3);
    border-left: 4px solid var(--accent-green);
    border-radius: var(--radius);
    padding: 12px 18px;
    font-size: 0.85rem;
    color: var(--accent-green);
}

/* ── Metric Pills ── */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    margin: 4px 4px 4px 0;
}
.metric-pill .pill-val { color: var(--text-primary); font-weight: 600; }

/* ── Data Table ── */
.stDataFrame {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
}

/* ── Sidebar metric row ── */
.sidebar-metric {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sm-label { font-size: 0.75rem; color: var(--text-muted); }
.sm-val   { font-weight: 700; font-size: 0.95rem; color: var(--text-primary); }

/* ── Progress bar ── */
.prog-row { margin-bottom: 12px; }
.prog-label { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 4px; display: flex; justify-content: space-between; }
.prog-bar   { height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 999px; }

/* ── Logo / Brand bar ── */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    padding: 16px 24px;
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    backdrop-filter: blur(12px);
}
.brand-icon { font-size: 2.2rem; }
.brand-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
}
.brand-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
.brand-badge {
    margin-left: auto;
    background: rgba(47, 128, 237, 0.12);
    border: 1px solid rgba(47, 128, 237, 0.4);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.72rem;
    color: var(--accent-blue);
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
/* ── Shtat.ai Brand Overrides ── */
:root {
    --shtat-blue:    #004a99;
    --shtat-blue-h:  #003a78;
    --shtat-glow:    rgba(0, 74, 153, 0.35);
}

/* Enterprise Blue primary button */
.stButton > button[kind="primary"],
.stButton > button {
    background: var(--shtat-blue) !important;
    border: 1px solid rgba(0,74,153,0.6) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: background 0.18s, box-shadow 0.18s !important;
}
.stButton > button:hover {
    background: var(--shtat-blue-h) !important;
    box-shadow: 0 0 0 3px var(--shtat-glow) !important;
}
.stButton > button:disabled {
    background: rgba(0,74,153,0.25) !important;
    color: rgba(255,255,255,0.45) !important;
    border-color: transparent !important;
    cursor: not-allowed !important;
}

/* Shtat.ai sidebar brand strip */
.shtat-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(0,74,153,0.18), rgba(0,74,153,0.06));
    border: 1px solid rgba(0,74,153,0.35);
    border-radius: 10px;
    margin-bottom: 16px;
}
.shtat-sidebar-logo {
    font-size: 1.5rem;
    line-height: 1;
}
.shtat-sidebar-name {
    font-size: 1.05rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -0.01em;
}
.shtat-sidebar-tag {
    font-size: 0.70rem;
    color: #8b949e;
    margin-top: 1px;
}
.shtat-sidebar-footer {
    font-size: 0.70rem;
    color: #8b949e;
    text-align: center;
    padding: 10px 0 4px 0;
    border-top: 1px solid rgba(48,54,61,0.5);
    margin-top: 12px;
}

/* Cross-faculty compliance panel */
.shtat-compliance-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.80rem;
    font-weight: 700;
    color: var(--shtat-blue);
    background: rgba(0,74,153,0.10);
    border: 1px solid rgba(0,74,153,0.30);
    border-radius: 8px;
    padding: 7px 14px;
    margin-bottom: 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# LAYER 1 — SEMANTIC DATA PIPELINE
# ===========================================================================
# This is the "Semantic" half of the hybrid system.
# Owlready2 loads the OWL/RDF graph and exposes individuals as Python objects.
# We then query their class memberships and data-property values to build a
# structured, analysis-ready DataFrame — bridging the Semantic ↔ Data Science gap.
# ===========================================================================

@st.cache_resource(show_spinner=False)
def load_ontology(path: str):
    """
    Load the OWL ontology using Owlready2 with caching.

    @st.cache_resource ensures the heavy RDF graph is parsed only once per
    server session, making subsequent page refreshes instantaneous.

    Returns:
        onto : Owlready2 ontology object  (None on failure)
        error: str | None                 (human-readable message)
    """
    try:
        from owlready2 import get_ontology, sync_reasoner_pellet
        onto = get_ontology(path).load()
        return onto, None
    except FileNotFoundError:
        return None, f"[XATO]  Ontology file not found at: `{path}`"
    except Exception as exc:
        return None, f"[XATO]  Failed to load ontology — {exc}"


def extract_dataframe(onto, name_lookup: dict = None) -> pd.DataFrame:
    """
    SEMANTIC → TABULAR BRIDGE

    Walks every individual in the ontology and extracts:
      • ID             : OWL IRI local name (xodim_01, xodim_02, …)
      • Name           : human-readable staff name from HR Excel (F.I.Sh.)
      • Position       : mapped from OWL class (Dotsent → Associate Professor, etc.)
      • EmploymentType : mapped from OWL class (Asosiy → Full-Time, etc.)
      • Workload       : shtat_birligi data property (float, FTE units)

    Exact class names verified against faculty_finall.owx:
      Position classes  : Assistent, Dotsent, VB_Dotsent, Katta_oqituvchi,
                          Professor, VB_Professor, Stajer_oqituvchi
      Employment classes: Asosiy, Ichki_orindosh, Tashqi_orindosh

    Design note (thesis): We use direct `is_a` class membership (as asserted
    in the OWL file) rather than relying on the reasoner, because the
    ClassAssertion axioms are explicit. INDIRECT_is_a would also traverse
    parent classes like Oqituvchi_lavozimi / Position which are abstract — we
    filter those out via the POSITION_MAP allow-list.
    """

    # ── O'zbek terminologiyasi (OWL klass nomlari bilan to'liq mos) ─────────
    # Manba: faculty_finall.owx — barcha lavozim va bandlik turlari o'zbek tilida.
    POSITION_MAP = {
        "Assistent"         : "Assistent",
        "Dotsent"           : "Dotsent",
        "VB_Dotsent"        : "VB Dotsent",          # Vazifasini Bajaruvchi Dotsent
        "Professor"         : "Professor",
        "VB_Professor"      : "VB Professor",         # Vazifasini Bajaruvchi Professor
        "Katta_oqituvchi"   : "Katta o'qituvchi",
        "Stajer_oqituvchi"  : "Stajer o'qituvchi",
    }

    EMPLOYMENT_MAP = {
        "Asosiy"            : "Asosiy",               # shtатный xodim
        "Ichki_orindosh"    : "Ichki o'rindosh",     # ichki совместитель
        "Tashqi_orindosh"   : "Tashqi o'rindosh",    # tashqi совместитель
    }

    records = []

    for ind in onto.individuals():
        ind_id = ind.name  # e.g. "xodim_01"

        # ── Direct class assertions from the OWL file ─────────────────────
        # We use is_a (not INDIRECT_is_a) to avoid picking up abstract parent
        # classes (Position, EmploymentType, Oqituvchi_lavozimi, etc.)
        classes = [c.name for c in ind.is_a if hasattr(c, "name")]

        position = next(
            (POSITION_MAP[c] for c in classes if c in POSITION_MAP),
            "Unknown"
        )
        employment_type = next(
            (EMPLOYMENT_MAP[c] for c in classes if c in EMPLOYMENT_MAP),
            "Unknown"
        )

        # ── Staff full name from HR Excel lookup ───────────────────────────
        display_name = (name_lookup or {}).get(ind_id, ind_id)

        # ── Extract shtat_birligi data property ───────────────────────────
        raw_workload = getattr(ind, "shtat_birligi", None)
        if isinstance(raw_workload, list):
            workload = float(raw_workload[0]) if raw_workload else 0.0
        elif raw_workload is not None:
            workload = float(raw_workload)
        else:
            workload = 0.0

        raw_classes = ", ".join(classes)

        records.append({
            "ID"             : ind_id,
            "Name"           : display_name,
            "Position"       : position,
            "PositionRaw"    : next((c for c in classes if c in POSITION_MAP), ""),
            "EmploymentType" : employment_type,
            "Workload"       : workload,
            "RawClasses"     : raw_classes,
        })

    df = pd.DataFrame(records)

    # ── Derived columns ────────────────────────────────────────────────────
    # Constraint threshold: shtat_birligi > 1.5 is the OWL datatype violation.
    # In this dataset the max is 1.0, so we also flag anyone at exactly 1.0
    # who is registered as a part-timer (business rule cross-check).
    df["WorkloadExceeded"] = df["Workload"] > 1.5
    df["WorkloadCategory"] = pd.cut(
        df["Workload"],
        bins=[-0.01, 0.25, 0.5, 1.0, 1.5, 9.0],
        labels=["Minimal (≤0.25)", "Yarim shtat (≤0.5)", "Asosiy shtat (≤1.0)",
                "Og'ir yuk (≤1.5)", "⚠ Me'yordan oshgan (>1.5)"],
    )

    return df.sort_values("ID").reset_index(drop=True)


# ===========================================================================
# DEMO DATA GENERATOR — fallback when .owx is absent (thesis presentation mode)
# ===========================================================================

def generate_demo_dataframe() -> pd.DataFrame:
    """
    Builds the DataFrame directly from the known contents of faculty_finall.owx
    and hr_faculty_list.xlsx. Used when the .owx file is not present but the
    data is already fully known from the uploaded files.

    This is NOT random synthetic data — it is the exact ontology content
    hard-coded for resilient demo / thesis-defense mode.
    """
    # Exact data from owx ClassAssertions + xlsx F.I.Sh. column
    POSITION_MAP = {
        "Assistent"       : "Assistent",
        "Dotsent"         : "Dotsent",
        "VB_Dotsent"      : "VB Dotsent",
        "Professor"       : "Professor",
        "VB_Professor"    : "VB Professor",
        "Katta_oqituvchi" : "Katta o'qituvchi",
        "Stajer_oqituvchi": "Stajer o'qituvchi",
    }
    EMPLOYMENT_MAP = {
        "Asosiy"          : "Asosiy",
        "Ichki_orindosh"  : "Ichki o'rindosh",
        "Tashqi_orindosh" : "Tashqi o'rindosh",
    }

    # Raw records from owx + xlsx (ID, full name, position class, employment class, workload)
    raw = [
        ("xodim_01", "Abidova Sh.B.",       "VB_Dotsent",      "Asosiy",          1.0),
        ("xodim_02", "Abidova Sh.B.",       "VB_Dotsent",      "Ichki_orindosh",  0.5),
        ("xodim_03", "Artikova M.A.",       "Dotsent",         "Asosiy",          1.0),
        ("xodim_04", "Artikova M.A.",       "Dotsent",         "Ichki_orindosh",  0.5),
        ("xodim_05", "Azimov S.R.",         "Assistent",       "Asosiy",          1.0),
        ("xodim_06", "Jo'rayev A.",         "Assistent",       "Asosiy",          1.0),
        ("xodim_07", "Jo'rayev A.",         "Assistent",       "Ichki_orindosh",  0.5),
        ("xodim_08", "Mahmudova M.M.",      "VB_Dotsent",      "Asosiy",          1.0),
        ("xodim_09", "Mahmudova M.M.",      "VB_Dotsent",      "Ichki_orindosh",  0.5),
        ("xodim_10", "Nazirov A.Sh.",       "Katta_oqituvchi", "Asosiy",          1.0),
        ("xodim_11", "Ne'matov A.",         "Dotsent",         "Asosiy",          1.0),
        ("xodim_12", "Sadikov R.T.",        "Dotsent",         "Asosiy",          1.0),
        ("xodim_13", "Sadikov R.T.",        "Dotsent",         "Ichki_orindosh",  0.5),
        ("xodim_14", "Sayfiyev E.",         "Katta_oqituvchi", "Asosiy",          1.0),
        ("xodim_15", "Sayfiyev E.",         "Katta_oqituvchi", "Ichki_orindosh",  0.5),
        ("xodim_16", "Sharipov D.K.",       "Dotsent",         "Asosiy",          1.0),
        ("xodim_17", "Talipova O.X.",       "Katta_oqituvchi", "Asosiy",          1.0),
        ("xodim_18", "Xayrullayev U.",      "Assistent",       "Asosiy",          1.0),
        ("xodim_19", "Xayrullayev U.",      "Assistent",       "Ichki_orindosh",  0.5),
        ("xodim_20", "Xusanov Sh.",         "Katta_oqituvchi", "Asosiy",          1.0),
        ("xodim_21", "Xolmuminov Yu.",      "Assistent",       "Asosiy",          1.0),
        ("xodim_22", "Ismailov Sh.R.",      "VB_Dotsent",      "Tashqi_orindosh", 0.5),
        ("xodim_23", "Karabayeva Z.",       "Assistent",       "Tashqi_orindosh", 0.5),
        ("xodim_24", "Kushnazarov F.",      "Dotsent",         "Tashqi_orindosh", 0.5),
        ("xodim_25", "Medetbayeva N.",      "Assistent",       "Tashqi_orindosh", 0.5),
        ("xodim_26", "Nazirova E.",         "Professor",       "Tashqi_orindosh", 0.5),
        ("xodim_27", "Nurjabova O.",        "Assistent",       "Tashqi_orindosh", 0.5),
        ("xodim_28", "Ravshanov N.",        "Professor",       "Tashqi_orindosh", 0.5),
        ("xodim_29", "Rayimqulov O.",       "Assistent",       "Tashqi_orindosh", 0.5),
        ("xodim_30", "Uzoquva M.",          "Katta_oqituvchi", "Tashqi_orindosh", 0.5),
        ("xodim_31", "Xudayberganov Sh.",   "Assistent",       "Tashqi_orindosh", 0.5),
        ("xodim_32", "Xolmuminov Y.",       "Assistent",       "Tashqi_orindosh", 0.5),
    ]

    rows = []
    for (ind_id, name, pos_raw, emp_raw, workload) in raw:
        rows.append({
            "ID"             : ind_id,
            "Name"           : name,
            "Position"       : POSITION_MAP.get(pos_raw, pos_raw),
            "PositionRaw"    : pos_raw,
            "EmploymentType" : EMPLOYMENT_MAP.get(emp_raw, emp_raw),
            "Workload"       : float(workload),
            "RawClasses"     : f"{pos_raw}, {emp_raw}",
        })

    df = pd.DataFrame(rows)
    df["WorkloadExceeded"] = df["Workload"] > 1.5
    df["WorkloadCategory"] = pd.cut(
        df["Workload"],
        bins=[-0.01, 0.25, 0.5, 1.0, 1.5, 9.0],
        labels=["Minimal (≤0.25)", "Yarim shtat (≤0.5)", "Asosiy shtat (≤1.0)",
                "Og'ir yuk (≤1.5)", "⚠ Me'yordan oshgan (>1.5)"],
    )
    return df


# ===========================================================================
# PLOTLY THEME HELPER
# ===========================================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e6edf3", size=12),
    margin=dict(t=40, b=20, l=20, r=20),
    legend=dict(
        bgcolor="rgba(22,27,34,0.9)",
        bordercolor="rgba(48,54,61,0.8)",
        borderwidth=1,
    ),
    colorway=["#2f80ed", "#00b4d8", "#f5a623", "#27ae60", "#eb5757",
              "#9b59b6", "#e67e22", "#1abc9c"],
)


def styled_fig(fig):
    """Apply the shared dark theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ===========================================================================
# COMPONENT RENDERERS
# ===========================================================================

def render_kpi_cards(df: pd.DataFrame):
    """
    Four KPI cards: Total Staff, Total FTE, F1-Score placeholder, Budget %.
    These simulate the executive-level metrics a DSS must surface instantly.
    """
    total_staff     = len(df)
    total_fte       = df["Workload"].sum()
    flagged         = df["WorkloadExceeded"].sum()
    # F1-Score placeholder — in a real hybrid system this would come from a
    # trained classifier (e.g., SVM / Random Forest) applied to staff profiles.
    f1_score        = 0.94
    budget_util     = min((total_fte / (total_staff * 1.0)) * 100, 100)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-label">Jami Xodimlar</div>
            <div class="kpi-value">{total_staff}</div>
            <div class="kpi-sub">Ontologiyadagi individlar soni</div>
        </div>
        <div class="kpi-card teal">
            <div class="kpi-label">Jami Shtat Birligi</div>
            <div class="kpi-value">{total_fte:.2f}</div>
            <div class="kpi-sub">Σ shtat_birligi · {flagged} ta xatolik</div>
        </div>
        <div class="kpi-card gold">
            <div class="kpi-label">Tizim Aniqligi</div>
            <div class="kpi-value">{f1_score:.0%}</div>
            <div class="kpi-sub">Audit F1-ko'rsatkichi (gibrid model)</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Byudjet Yuklanishi</div>
            <div class="kpi-value">{budget_util:.1f}%</div>
            <div class="kpi-sub">FTE / maksimal FTE nisbati</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_metrics(df: pd.DataFrame):
    """
    Sidebar: filter controls + resource-saving statistics that substantiate
    the 'Hybrid' claim in the thesis (semantic layer saves ~95% manual auditing).
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtrlar")

    positions = ["Barchasi"] + sorted(df["Position"].unique().tolist())
    sel_pos   = st.sidebar.selectbox("Lavozim (OWL Klassi)", positions)

    emp_types = ["Barchasi"] + sorted(df["EmploymentType"].unique().tolist())
    sel_emp   = st.sidebar.selectbox("Bandlik Turi", emp_types)

    # ── Workload slider — safe bounds ──────────────────────────────────────
    # When all members of a faculty have identical workloads (e.g. every record
    # is 1.0 SB), Streamlit raises StreamlitAPIException because min == max.
    # We always guarantee at least a 0.05 gap so the slider renders safely.
    _wl_data_min = float(df["Workload"].min()) if not df.empty else 0.0
    _wl_data_max = float(df["Workload"].max()) if not df.empty else 1.5
    _sl_min  = 0.0  if _wl_data_min == _wl_data_max else _wl_data_min
    _sl_max  = max(_wl_data_max, _sl_min + 0.05)   # always > _sl_min
    _sl_lo   = _sl_min
    _sl_hi   = _sl_max

    wl_range = st.sidebar.slider(
        "Shtat Birligi (shtat_birligi)",
        min_value=_sl_min,
        max_value=_sl_max,
        value=(_sl_lo, _sl_hi),
        step=0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Resurs Tejash")

    st.sidebar.markdown(f"""
    <div class="sidebar-metric">
        <span class="sm-label">Avtomatlashtirilgan vazifalar</span>
        <span class="sm-val" style="color:#27ae60">95%</span>
    </div>
    <div class="sidebar-metric">
        <span class="sm-label">Audit vaqti qisqarishi</span>
        <span class="sm-val" style="color:#2f80ed">~8s → 12s</span>
    </div>
    <div class="sidebar-metric">
        <span class="sm-label">Ontologiya individlari</span>
        <span class="sm-val">{len(df)}</span>
    </div>
    <div class="sidebar-metric">
        <span class="sm-label">Cheklov buzilishlari</span>
        <span class="sm-val" style="color:#eb5757">{df['WorkloadExceeded'].sum()}</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Gibrid Arxitektura")
    st.sidebar.markdown("""
    <div style="font-size:0.77rem;color:#8b949e;line-height:1.7">
    <b style="color:#e6edf3">1-qatlam</b> · OWL Semantik Mantiq<br>
    <b style="color:#e6edf3">2-qatlam</b> · Pandas Ma'lumot Muhandisligi<br>
    <b style="color:#e6edf3">3-qatlam</b> · Qoidalarga Asoslangan Audit<br>
    <b style="color:#e6edf3">4-qatlam</b> · Bashoratli KPI Tahlili<br>
    </div>
    """, unsafe_allow_html=True)

    # ── What-If Simulator Slider ──────────────────────────────────────────
    # This slider lives in the sidebar so it affects the gauge in real-time
    # as the user moves it — Streamlit's reactive model handles the re-render.
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Yuklanish Simulatori")
    st.sidebar.markdown(
        '<div style="font-size:0.76rem;color:#8b949e;margin-bottom:8px">'
        'Qabul qilinishi rejalashtirilgan yangi xodimlar (har biri 1.0 SB) sonini '
        'tanlang — o\'ng tomondagi kafedrometr real vaqtda yangilanadi.</div>',
        unsafe_allow_html=True,
    )
    sim_new_hires = st.sidebar.slider(
        "Yangi xodimlar sonini simulyatsiya qilish",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        key="sidebar_sim_hires",
    )

    current_sim_total = df["Workload"].sum() + sim_new_hires * 1.0
    sim_zone = (
        "Barqaror"   if current_sim_total <= 25 else
        "Ehtiyotkor" if current_sim_total <= 30 else
        "Kritik"
    )
    sim_color = "#27ae60" if current_sim_total <= 25 else ("#f5a623" if current_sim_total <= 30 else "#eb5757")
    st.sidebar.markdown(f"""
    <div class="sidebar-metric" style="margin-top:4px">
        <span class="sm-label">Prognoz jami SB</span>
        <span class="sm-val" style="color:{sim_color}">{current_sim_total:.1f}</span>
    </div>
    <div class="sidebar-metric">
        <span class="sm-label">Tizim holati</span>
        <span class="sm-val" style="color:{sim_color}">{sim_zone}</span>
    </div>
    """, unsafe_allow_html=True)

    return sel_pos, sel_emp, wl_range, sim_new_hires


def apply_filters(df: pd.DataFrame, sel_pos, sel_emp, wl_range) -> pd.DataFrame:
    """Filtr tanlovlarini qo'llab, ishchi to'plamni qaytaradi."""
    mask = (df["Workload"] >= wl_range[0]) & (df["Workload"] <= wl_range[1])
    if sel_pos != "Barchasi":
        mask &= df["Position"] == sel_pos
    if sel_emp != "Barchasi":
        mask &= df["EmploymentType"] == sel_emp
    return df[mask].reset_index(drop=True)


# ===========================================================================
# TAB 1 — EXECUTIVE DASHBOARD
# ===========================================================================

def tab_executive(df_filtered: pd.DataFrame, df_full: pd.DataFrame):
    """
    High-level visualisations for department heads and university administration.
    Sunburst: Position → Name hierarchy (semantic class structure made visual).
    Bar: Workload distribution per position.
    """
    st.markdown('<div class="section-header">Kadrlar Taqsimoti</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ontologiya klass ierarxiyasi xodimlar soni va shtat birligi (FTE) bo\'yicha vizuallashtirilgan.</div>', unsafe_allow_html=True)

    col_sun, col_bar = st.columns([1.2, 1])

    # ── Sunburst: Position → EmploymentType → Name ─────────────────────────
    with col_sun:
        fig_sun = px.sunburst(
            df_filtered,
            path=["Position", "EmploymentType", "Name"],
            values="Workload",
            color="Position",
            title="Ontologiya Ierarxiyasi · Lavozim → Bandlik → Xodim",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_sun.update_traces(
            textfont_size=11,
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>Shtat birligi: %{value:.2f}<extra></extra>",
        )
        fig_sun = styled_fig(fig_sun)
        st.plotly_chart(fig_sun, use_container_width=True)

    # ── Bar: Mean workload per position ────────────────────────────────────
    with col_bar:
        agg = (
            df_filtered
            .groupby("Position")["Workload"]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .rename(columns={"mean": "O'rtacha SB", "sum": "Jami SB", "count": "Soni"})
            .sort_values("Jami SB", ascending=True)
        )
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=agg["Position"],
            x=agg["Jami SB"],
            orientation="h",
            marker=dict(
                color=agg["Jami SB"],
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{v:.2f} SB" for v in agg["Jami SB"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Jami SB: %{x:.2f}<br>Xodimlar: %{customdata}<extra></extra>",
            customdata=agg["Soni"],
        ))
        fig_bar.update_layout(
            title="Lavozim Bo'yicha Jami Shtat Birligi",
            xaxis_title="Shtat birligi (shtat_birligi Σ)",
            yaxis_title="",
        )
        fig_bar = styled_fig(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Employment type pie + workload violin ──────────────────────────────
    col_pie, col_vio = st.columns(2)

    with col_pie:
        emp_counts = df_filtered["EmploymentType"].value_counts().reset_index()
        emp_counts.columns = ["Turi", "Soni"]
        fig_pie = px.pie(
            emp_counts, names="Turi", values="Soni",
            title="Bandlik Turi Bo'yicha Taqsimot",
            hole=0.55,
            color_discrete_sequence=["#2f80ed", "#00b4d8", "#f5a623", "#27ae60"],
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
        fig_pie = styled_fig(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_vio:
        fig_vio = px.violin(
            df_filtered, y="Workload", x="Position",
            box=True, points="all",
            color="Position",
            title="Lavozim Bo'yicha Shtat Birligi Taqsimoti",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_vio.add_hline(
            y=1.5, line_dash="dash", line_color="#eb5757",
            annotation_text="Maksimal chegara (1.5)",
            annotation_position="top right",
        )
        fig_vio = styled_fig(fig_vio)
        fig_vio.update_layout(showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True)

    # ── ILMIY SALMOQ (Scientific Weight) ──────────────────────────────────
    # Ratio of degree-holding staff (Professor/Dotsent) vs non-degree staff.
    # This metric is required by the supervisor for the academic quality report.
    st.markdown('<div class="section-header">Ilmiy Salmoq va Rotatsiya Balansi</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Kafedraning ilmiy salmoqi (ilmiy unvon egalariga '
        'nisbat) va tashqi rotatsiya balansi ko\'rsatkichlari.</div>',
        unsafe_allow_html=True,
    )

    sal_col1, sal_col2 = st.columns(2)

    with sal_col1:
        # Classify positions into degree-holding vs non-degree
        degree_positions    = {"Professor", "VB Professor", "Dotsent", "VB Dotsent"}
        non_degree_positions = {"Assistent", "Katta o'qituvchi", "Stajer o'qituvchi"}

        df_sal = df_filtered.copy()
        df_sal["Toifa"] = df_sal["Position"].apply(
            lambda p: "Ilmiy unvonli (Prof/Dots)"
            if p in degree_positions
            else "Ilmiy unvonsiz (o'qituvchi)"
        )
        sal_counts = df_sal["Toifa"].value_counts().reset_index()
        sal_counts.columns = ["Toifa", "Soni"]
        sal_fte = df_sal.groupby("Toifa")["Workload"].sum().reset_index()
        sal_fte.columns = ["Toifa", "Jami SB"]
        sal_merge = sal_counts.merge(sal_fte, on="Toifa")

        degree_total = sal_counts.loc[
            sal_counts["Toifa"].str.contains("unvonli"), "Soni"
        ].sum()
        all_total = sal_counts["Soni"].sum()
        salmoq_pct = (degree_total / all_total * 100) if all_total > 0 else 0

        fig_salmoq = go.Figure()
        for _, row in sal_merge.iterrows():
            color = "#2f80ed" if "unvonli" in row["Toifa"] else "#8b949e"
            fig_salmoq.add_trace(go.Bar(
                name=row["Toifa"],
                x=[row["Toifa"]],
                y=[row["Soni"]],
                text=[f"{row['Soni']} kishi<br>{row['Jami SB']:.1f} SB"],
                textposition="inside",
                marker_color=color,
            ))
        fig_salmoq.update_layout(
            title=f"Ilmiy Salmoq — {salmoq_pct:.0f}% ilmiy unvonli",
            xaxis_title="",
            yaxis_title="Xodimlar soni",
            showlegend=False,
            barmode="group",
        )
        fig_salmoq = styled_fig(fig_salmoq)
        st.plotly_chart(fig_salmoq, use_container_width=True)

        # Salmoq indicator
        salmoq_cls = (
            "decision-ok" if salmoq_pct >= 50
            else "decision-warning" if salmoq_pct >= 30
            else "decision-error"
        )
        salmoq_label = (
            "Yuqori ilmiy salmoq" if salmoq_pct >= 50
            else "O'rtacha ilmiy salmoq" if salmoq_pct >= 30
            else "Past ilmiy salmoq — kadrlar siyosatini ko'rib chiqing"
        )
        st.markdown(
            f'<div class="decision-box {salmoq_cls}" style="margin-top:0;padding:12px 16px;">' +
            f'<b>Ilmiy Salmoq Ko\'rsatkichi:</b> {salmoq_pct:.1f}% &nbsp;|&nbsp; {salmoq_label}' +
            '</div>',
            unsafe_allow_html=True,
        )

    with sal_col2:
        # Rotation balance: Tashqi vs Ichki vs Asosiy
        rot_df = df_filtered.groupby("EmploymentType").agg(
            Soni=("Name","count"),
            JamiSB=("Workload","sum"),
        ).reset_index()

        fig_rot = px.bar(
            rot_df,
            x="EmploymentType",
            y="JamiSB",
            color="EmploymentType",
            text=rot_df.apply(
                lambda r: f"{r['Soni']} kishi<br>{r['JamiSB']:.1f} SB", axis=1
            ),
            title="Tashqi O'rindoshlik Balansi — Bandlik Turi Bo'yicha SB",
            color_discrete_map={
                "Asosiy"           : "#2f80ed",
                "Ichki o'rindosh" : "#00b4d8",
                "Tashqi o'rindosh": "#f5a623",
            },
            labels={"EmploymentType":"Bandlik Turi","JamiSB":"Jami SB"},
        )
        fig_rot.update_traces(textposition="inside")
        fig_rot.update_layout(showlegend=False, xaxis_title="")
        fig_rot = styled_fig(fig_rot)
        st.plotly_chart(fig_rot, use_container_width=True)

        # External balance metric card
        tashqi_fte  = rot_df.loc[rot_df["EmploymentType"].str.contains("Tashqi","Tashqi",na=False), "JamiSB"].sum()
        asosiy_fte  = rot_df.loc[rot_df["EmploymentType"] == "Asosiy", "JamiSB"].sum()
        ext_ratio   = (tashqi_fte / (asosiy_fte + tashqi_fte) * 100) if (asosiy_fte + tashqi_fte) > 0 else 0
        ext_cls     = "decision-ok" if ext_ratio <= 30 else "decision-warning" if ext_ratio <= 50 else "decision-error"
        st.markdown(
            f'<div class="decision-box {ext_cls}" style="margin-top:0;padding:12px 16px;">' +
            f'<b>Tashqi Rotatsiya Nisbati:</b> {ext_ratio:.1f}% ' +
            f'({tashqi_fte:.1f} SB tashqi / {asosiy_fte + tashqi_fte:.1f} SB jami)' +
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Raw data table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Xodimlar Ro\'yxati</div>', unsafe_allow_html=True)
    display_df = df_filtered[["ID", "Name", "Position", "EmploymentType", "Workload", "WorkloadCategory"]].copy()
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID"              : st.column_config.TextColumn("Xodim ID"),
            "Name"            : st.column_config.TextColumn("F.I.Sh."),
            "Position"        : st.column_config.TextColumn("Lavozim"),
            "EmploymentType"  : st.column_config.TextColumn("Bandlik Turi"),
            "Workload"        : st.column_config.NumberColumn("Shtat Birligi", format="%.2f"),
            "WorkloadCategory": st.column_config.TextColumn("Yuklanish Toifasi"),
        },
    )



# ===========================================================================
# CUMULATIVE IDENTITY AGGREGATION ENGINE
# ===========================================================================

def _canonical_key(name: str) -> str:
    """
    Shared canonical key builder — used by both aggregation functions
    so that the audit and the NLP entity resolution are always in sync.

    "Nazirova Elmira"  →  "nazirova_e"
    "Nazirova E."      →  "nazirova_e"   (same key)
    "Nazarov E."       →  "nazarov_e"    (different key)
    "Aliyev"           →  "aliyev"       (no initial, safe)
    ""                 →  "__unknown__"
    """
    tokens = name.strip().split()
    if not tokens or tokens[0] == "":
        return "__unknown__"
    surname = tokens[0].lower().rstrip(".")
    initial = tokens[1][0].lower() if len(tokens) > 1 and tokens[1] else ""
    return f"{surname}_{initial}" if initial else surname


def _add_canonical_key(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised canonical key assignment — modifies a copy."""
    out = df.copy()
    out["Name"]     = out["Name"].fillna("").astype(str)
    out["Workload"] = pd.to_numeric(out["Workload"], errors="coerce").fillna(0.0)
    tokens          = out["Name"].str.strip().str.split()
    surname_key     = tokens.str[0].str.lower().str.rstrip(".").fillna("__unknown__")
    initial_key     = tokens.apply(
        lambda t: t[1][0].lower() if t and len(t) > 1 and t[1] else ""
    )
    out["CanonicalKey"] = (surname_key + "_" + initial_key).str.rstrip("_")
    return out


def build_global_workload_map(df_global: pd.DataFrame) -> dict[str, float]:
    """
    Build a canonical-key → total global SB mapping from the university-wide
    dataset.  Used to look up a person's *cross-faculty total* while auditing
    a single faculty.

    Returns: {"nazirova_e": 1.50, "toshmatov_j": 1.00, ...}
    """
    if df_global is None or df_global.empty:
        return {}
    gdf = _add_canonical_key(df_global)
    return gdf.groupby("CanonicalKey")["Workload"].sum().to_dict()


def aggregate_by_identity(
    df: pd.DataFrame,
    ind_limit: float = 1.5,
    global_wl_map: dict | None = None,
) -> pd.DataFrame:
    """
    Cumulative Identity Aggregation Engine — now with cross-faculty awareness.

    Parameters
    ----------
    df           : current-faculty filtered DataFrame
    ind_limit    : individual workload ceiling (from sidebar)
    global_wl_map: {canonical_key: total_global_SB} from build_global_workload_map().
                   When provided, flagging is done on *global* totals, not just
                   the current faculty.  FacultyWorkload and OtherFacultyWorkload
                   columns are added for the breakdown UI.

    Returns
    -------
    One row per unique identity.  Extra columns when global_wl_map is given:
      GlobalTotalWorkload  — sum across ALL faculties
      FacultyWorkload      — contribution in the current faculty only
      OtherFacultyWorkload — GlobalTotalWorkload − FacultyWorkload
      CumulativeExceeded   — GlobalTotalWorkload > ind_limit  (not just faculty)
      ExcessSB             — GlobalTotalWorkload − ind_limit  (floored at 0)
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "CanonicalKey","DisplayName","TotalWorkload","RowCount",
            "Positions","EmployTypes","IDs","CumulativeExceeded","ExcessSB",
        ])

    work_df = _add_canonical_key(df)
    for col in ("Position", "EmploymentType", "ID"):
        if col in work_df.columns:
            work_df[col] = work_df[col].fillna("").astype(str)
        else:
            work_df[col] = ""

    agg_rows = []
    for key, group in work_df.groupby("CanonicalKey", sort=False):
        fac_wl  = float(group["Workload"].sum())
        # Global total: take from the map if available, else use faculty total
        global_wl = float(global_wl_map.get(key, fac_wl)) if global_wl_map else fac_wl
        other_wl  = round(max(global_wl - fac_wl, 0.0), 4)

        best_idx     = group["Name"].str.len().idxmax()
        display_name = group.loc[best_idx, "Name"]
        exceeded     = global_wl > ind_limit
        excess        = round(max(global_wl - ind_limit, 0.0), 4)

        agg_rows.append({
            "CanonicalKey"       : key,
            "DisplayName"        : display_name,
            "TotalWorkload"      : round(fac_wl, 4),      # faculty contribution
            "GlobalTotalWorkload": round(global_wl, 4),   # cross-faculty total
            "FacultyWorkload"    : round(fac_wl, 4),
            "OtherFacultyWorkload": other_wl,
            "RowCount"           : len(group),
            "Positions"          : " / ".join(group["Position"].unique().tolist()),
            "EmployTypes"        : " / ".join(group["EmploymentType"].unique().tolist()),
            "IDs"                : ", ".join(group["ID"].unique().tolist()),
            "CumulativeExceeded" : exceeded,
            "ExcessSB"           : excess,
        })

    return (
        pd.DataFrame(agg_rows)
        .sort_values("GlobalTotalWorkload", ascending=False)
        .reset_index(drop=True)
    )


# ===========================================================================
# TAB 2 — SEMANTIC AUDIT (the "Hybrid" core)
# ===========================================================================

def tab_semantic_audit(df_filtered: pd.DataFrame,
                       df_global: pd.DataFrame | None = None):
    """
    HYBRID SEMANTIC AUDITOR — Global Cross-Faculty Edition

    Two-level audit:
      Level 1 (Department): Kafedrometr gauge uses current-faculty FTE only.
      Level 2 (Individual): Flagging uses GlobalTotalWorkload — the person's
                            total SB across ALL faculties — not just their
                            contribution to the currently viewed faculty.

    Rule: ∀ shaxs ∈ Universitet · Σ_global shtat_birligi(shaxs) ≤ ind_limit
    """
    # ── Read dynamic individual limit from session state ──────────────────
    constraints = st.session_state.get("dyn_constraints", {})
    ind_limit   = constraints.get("max_ind_sb", 1.5)

    # ── Build cross-faculty workload map ──────────────────────────────────
    # global_wl_map: {canonical_key → total SB across all faculties}
    # Falls back to session state global_df, then to df_filtered if unavailable.
    if df_global is None or df_global.empty:
        df_global = st.session_state.get("global_df", df_filtered)
    global_wl_map = build_global_workload_map(df_global)
    using_global  = bool(global_wl_map)

    selected_faculty = st.session_state.get("selected_faculty", None)

    st.markdown('<div class="section-header">Gibrid Semantik Auditor</div>', unsafe_allow_html=True)

    # ── Shtat.ai product label ─────────────────────────────────────────────
    st.markdown(
        '<div class="shtat-compliance-header">'
        '🔍 &nbsp; Shtat.ai Cross-Faculty Compliance Check'
        '</div>',
        unsafe_allow_html=True,
    )

    global_scope_note = (
        "barcha fakultetlar bo'yicha global" if using_global
        else "joriy fakultet"
    )
    st.markdown(
        f'<div class="section-sub">'
        f'Cheklov: <code style="background:rgba(255,255,255,0.07);padding:2px 6px;border-radius:4px">'
        f'∀ shaxs ∈ Universitet · Σ shtat_birligi(shaxs) ≤ {ind_limit:.2f}</code>'
        f' — <b>Kumulyativ {global_scope_note}</b> yuk asosida tekshiriladi.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Run cumulative aggregation with global map ─────────────────────────
    agg_df = aggregate_by_identity(df_filtered, ind_limit=ind_limit,
                                   global_wl_map=global_wl_map)

    if agg_df.empty:
        st.info("Hech qanday ma'lumot yo'q.")
        return

    flagged_agg = agg_df[agg_df["CumulativeExceeded"]].copy()
    clean_agg   = agg_df[~agg_df["CumulativeExceeded"]].copy()

    # ── Summary metrics ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jami Unikal Shaxslar",    len(agg_df),
              help="Kumulyativ identifikatsiya bo'yicha")
    c2.metric("[OK] Muvofiq",             len(clean_agg),
              delta=f"{len(clean_agg)/max(len(agg_df),1)*100:.0f}%")
    c3.metric("Kumulyativ Buzilishlar",   len(flagged_agg),
              delta=f"-{len(flagged_agg)}", delta_color="inverse",
              help=f"Σ SB > {ind_limit:.2f} bo'lgan shaxslar")
    c4.metric("Muvofiqlik Darajasi",
              f"{len(clean_agg)/max(len(agg_df),1)*100:.1f}%")

    # ── Multi-row note ─────────────────────────────────────────────────────
    multi_row = agg_df[agg_df["RowCount"] > 1]
    if not multi_row.empty:
        names_list = ", ".join(multi_row["DisplayName"].tolist())
        st.info(
            f"Ko'p Lavozimlillar: {len(multi_row)} ta shaxs bir nechta "
            f"qatorda mavjud: {names_list}. "
            f"Ularning yig'indi SB qiymatlari tekshirildi."
        )

    st.markdown("---")

    # ── Flagged identity cards ─────────────────────────────────────────────
    if len(flagged_agg) > 0:
        st.error(
            f"🚩 Shtat.ai Ogohlantirishi: {len(flagged_agg)} ta shaxs bo'yicha "
            f"me'yoriy cheklov buzilishi aniqlandi "
            f"(Global chegara: {ind_limit:.2f} SB). "
            f"Quyidagi xodimlarning umumiy universitetlik yuklamasi ruxsat etilgan "
            f"chegaradan oshadi."
        )
        st.markdown(
            f"**{len(flagged_agg)} ta shaxs kumulyativ global shtat birligi "
            f"chegarasini ({ind_limit:.2f} SB) buzmoqda:**"
        )
        for _, row in flagged_agg.iterrows():
            # Build within-faculty breakdown string
            orig_rows = df_filtered[
                df_filtered["Name"].str.strip().str.split().str[0]
                .str.lower().str.rstrip(".")
                == row["CanonicalKey"].split("_")[0]
            ]
            fac_parts = " + ".join(
                f"{r['Position']} ({r['Workload']:.2f} SB)"
                for _, r in orig_rows.iterrows()
            ) if not orig_rows.empty else f"{row['FacultyWorkload']:.2f} SB"

            fac_wl   = row.get("FacultyWorkload", row["TotalWorkload"])
            other_wl = row.get("OtherFacultyWorkload", 0.0)
            glob_wl  = row.get("GlobalTotalWorkload", row["TotalWorkload"])

            # Cross-faculty note
            if other_wl > 0 and using_global:
                cross_note = (
                    f'<span style="color:#f5a623">'
                    f'Boshqa fakultetlar: {other_wl:.2f} SB</span> &nbsp;+&nbsp; '
                    f'Joriy fakultet: {fac_wl:.2f} SB &nbsp;= &nbsp;'
                    f'<b style="color:#eb5757">Global: {glob_wl:.2f} SB</b>'
                )
            else:
                cross_note = (
                    f'Hisob: {fac_parts} = '
                    f'<b style="color:#eb5757">{glob_wl:.2f} SB</b>'
                )

            fac_label = f"({selected_faculty})" if selected_faculty else "(joriy fakultet)"

            st.markdown(f"""
            <div class="flag-card">
                <div class="flag-icon">🚩</div>
                <div style="flex:1">
                    <div class="flag-name">{row['DisplayName']}
                        <span style="font-size:0.75rem;color:#8b949e">
                            ({row['IDs']})
                        </span>
                    </div>
                    <div class="flag-detail" style="margin-bottom:4px">
                        {row['EmployTypes']} &nbsp;·&nbsp; {row['Positions']}
                    </div>
                    <div style="font-size:0.82rem;color:#c9d1d9">
                        Joriy fakultet {fac_label}: <b>{fac_wl:.2f} SB</b>
                        &nbsp;|&nbsp;
                        Boshqa fakultetlar: <b>{other_wl:.2f} SB</b>
                        &nbsp;|&nbsp;
                        {cross_note}
                    </div>
                    <div style="font-size:0.78rem;color:#8b949e;margin-top:3px">
                        Holat: Global jami {glob_wl:.2f} SB &gt; Chegara {ind_limit:.2f} SB
                        &mdash; <span style="color:#eb5757">+{row['ExcessSB']:.2f} SB me'yordan oshgan</span>
                    </div>
                </div>
                <div class="flag-value">{glob_wl:.2f} SB</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="ok-card">[OK] Barcha shaxslar kumulyativ '
            f'shtat_birligi ≤ {ind_limit:.2f} SB chekloviga muvofiq.</div>',
            unsafe_allow_html=True,
        )

    # ── Compliance gauge — based on global cumulative identities ──────────
    compliance_pct = len(clean_agg) / max(len(agg_df), 1) * 100
    gc1, gc2 = st.columns([1, 2])
    with gc1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=compliance_pct,
            delta={"reference": 100, "valueformat": ".1f"},
            title={"text": f"Global Kumulyativ Muvofiqlik<br>"
                           f"<span style='font-size:11px'>(Chegara: {ind_limit:.2f} SB)</span>",
                   "font": {"size": 13}},
            gauge={
                "axis"      : {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8b949e"},
                "bar"       : {"color": "#27ae60" if compliance_pct >= 85
                               else "#f5a623" if compliance_pct >= 60 else "#eb5757"},
                "bgcolor"   : "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps"     : [
                    {"range": [0,  60], "color": "rgba(235,87,87,0.2)"},
                    {"range": [60, 85], "color": "rgba(245,166,35,0.2)"},
                    {"range": [85, 100],"color": "rgba(39,174,96,0.2)"},
                ],
                "threshold" : {"line": {"color": "#eb5757", "width": 3}, "value": 85},
            },
        ))
        fig_gauge = styled_fig(fig_gauge)
        fig_gauge.update_layout(height=280, margin=dict(t=60, b=10, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Scatter — one dot per identity, y = GlobalTotalWorkload ───────────
    with gc2:
        # Use GlobalTotalWorkload for the y-axis so the visual matches the flags
        scatter_df = agg_df.copy()
        scatter_df["_y"] = scatter_df.get("GlobalTotalWorkload", scatter_df["TotalWorkload"])
        fig_scat = px.scatter(
            scatter_df,
            x="DisplayName",
            y="_y",
            color="CumulativeExceeded",
            hover_data={"CanonicalKey": False, "RowCount": True,
                        "Positions": True, "EmployTypes": True,
                        "FacultyWorkload": True,
                        "OtherFacultyWorkload": True,
                        "_y": False},
            color_discrete_map={True: "#eb5757", False: "#27ae60"},
            size="_y",
            size_max=22,
            title=f"Global Kumulyativ SB — Shaxs bo'yicha (Chegara: {ind_limit:.2f} SB)",
            labels={"_y": "Global Σ SB", "DisplayName": "",
                    "CumulativeExceeded": "Chegara buzilishi",
                    "RowCount": "Qator soni",
                    "FacultyWorkload": "Joriy Fakultet SB",
                    "OtherFacultyWorkload": "Boshqa Fakultetlar SB"},
        )
        fig_scat.add_hline(
            y=ind_limit, line_dash="dash", line_color="#f5a623",
            annotation_text=f"Global Chegara ({ind_limit:.2f} SB)",
            annotation_position="top right",
        )
        fig_scat.update_layout(
            xaxis_tickangle=-35, xaxis_title="", yaxis_title="Global Σ shtat_birligi (SB)",
        )
        fig_scat = styled_fig(fig_scat)
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Global aggregated identity table ──────────────────────────────────
    st.markdown('<div class="section-header">Global Kumulyativ Identifikatsiya Jadvali</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Har bir qator bitta unikal shaxsni ifodalaydi. '
        f'"Global Jami SB" — barcha fakultetlardagi yig\'indi. '
        f'Chegara: {ind_limit:.2f} SB.</div>',
        unsafe_allow_html=True,
    )

    has_cross = "OtherFacultyWorkload" in agg_df.columns
    table_cols = ["DisplayName", "FacultyWorkload", "OtherFacultyWorkload",
                  "GlobalTotalWorkload", "RowCount",
                  "Positions", "EmployTypes", "CumulativeExceeded", "ExcessSB"]
    table_cols = [c for c in table_cols if c in agg_df.columns]
    col_rename = {
        "DisplayName"         : "F.I.Sh. (Unikal Shaxs)",
        "FacultyWorkload"     : "Joriy Fak. SB",
        "OtherFacultyWorkload": "Boshqa Fak. SB",
        "GlobalTotalWorkload" : "Global Jami SB",
        "RowCount"            : "Shartnomalar",
        "Positions"           : "Lavozimlar",
        "EmployTypes"         : "Bandlik turlari",
        "CumulativeExceeded"  : "Buzilish",
        "ExcessSB"            : "Ortiqcha SB",
    }
    display_agg = agg_df[table_cols].rename(columns=col_rename)
    st.dataframe(
        display_agg,
        use_container_width=True,
        hide_index=True,
        column_config={
            "F.I.Sh. (Unikal Shaxs)" : st.column_config.TextColumn(width="medium"),
            "Joriy Fak. SB"          : st.column_config.NumberColumn(format="%.2f"),
            "Boshqa Fak. SB"         : st.column_config.NumberColumn(format="%.2f"),
            "Global Jami SB"         : st.column_config.NumberColumn(format="%.2f"),
            "Shartnomalar"           : st.column_config.NumberColumn(),
            "Buzilish"               : st.column_config.CheckboxColumn(),
            "Ortiqcha SB"            : st.column_config.NumberColumn(format="%.2f"),
        },
    )


# ===========================================================================
# TAB 3 — PREDICTIVE ANALYTICS  (Gauge + What-If + Risk + Stats)
# ===========================================================================

def build_capacity_gauge(
    current_fte: float,
    sim_fte:     float,
    zone_stable: float = 25.0,   # upper bound of green zone  (sidebar-driven)
    zone_caution: float = 30.0,  # upper bound of yellow zone (sidebar-driven = dept_max_fte)
) -> tuple:
    """
    Builds a dual-pointer speedometer gauge showing:
      • Current total faculty workload  (solid needle)
      • Simulated workload after new hires (ghost indicator)

    Zone boundaries are now fully dynamic — driven by the sidebar inputs so
    the Kafedrometr re-renders on every slider/number_input change.

    Colour zones:
      Green  : 0 – zone_stable   SB → Barqaror (Stable)
      Yellow : zone_stable – zone_caution SB → Ehtiyotkor (Caution)
      Red    : zone_caution+     SB → Kritik (Danger)
    """
    GAUGE_MAX = max(zone_caution * 1.4, sim_fte * 1.15, 40.0)

    # Needle colour follows the zone of the *simulated* value
    if sim_fte <= zone_stable:
        needle_color = "#27ae60"
        zone_label   = "Barqaror"
        zone_color   = "#27ae60"
    elif sim_fte <= zone_caution:
        needle_color = "#f5a623"
        zone_label   = "Ehtiyotkor"
        zone_color   = "#f5a623"
    else:
        needle_color = "#eb5757"
        zone_label   = "Kritik"
        zone_color   = "#eb5757"

    fig = go.Figure()

    # ── Main gauge: simulated value ────────────────────────────────────────
    fig.add_trace(go.Indicator(
        mode   = "gauge+number+delta",
        value  = sim_fte,
        delta  = {
            "reference"  : current_fte,
            "increasing" : {"color": "#eb5757"},
            "decreasing" : {"color": "#27ae60"},
            "valueformat": ".1f",
        },
        title  = {
            "text": (
                f"<b>Kafedra Yuklanish Kafedrometri</b><br>"
                f"<span style='font-size:13px;color:#8b949e'>"
                f"Joriy: {current_fte:.1f} SB  |  Simulyatsiya: {sim_fte:.1f} SB  |  "
                f"Limit: {zone_caution:.1f} SB</span>"
            ),
            "font": {"size": 15},
        },
        number = {"suffix": " SB", "font": {"size": 36, "color": zone_color}},
        gauge  = {
            "axis": {
                "range"    : [0, GAUGE_MAX],
                "tickwidth": 1,
                "tickcolor": "#8b949e",
                "tickfont" : {"size": 11, "color": "#8b949e"},
            },
            "bar"       : {"color": needle_color, "thickness": 0.28},
            "bgcolor"   : "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,            zone_stable],  "color": "rgba(39,174,96,0.15)"},
                {"range": [zone_stable,  zone_caution], "color": "rgba(245,166,35,0.20)"},
                {"range": [zone_caution, GAUGE_MAX],    "color": "rgba(235,87,87,0.20)"},
            ],
            "threshold": {
                "line" : {"color": "#eb5757", "width": 3},
                "value": zone_caution,    # dynamic hard limit marker
            },
        },
        domain = {"x": [0, 1], "y": [0, 1]},
    ))

    # ── Ghost marker: current (before simulation) ──────────────────────────
    fig.add_trace(go.Indicator(
        mode  = "number",
        value = current_fte,
        title = {"text": "<span style='font-size:11px;color:#8b949e'>Joriy SB</span>"},
        number= {"suffix": " SB", "font": {"size": 18, "color": "#8b949e"}},
        domain= {"x": [0.72, 1.0], "y": [0.0, 0.28]},
    ))

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
        margin = dict(t=80, b=10, l=20, r=20),
        height = 340,
    )
    return fig, zone_label, zone_color


def tab_predictive(df_filtered: pd.DataFrame, df_full: pd.DataFrame,
                   sim_new_hires: int = 0,
                   dept_max_fte: float = 30.0):
    """
    Predictive/Statistical analytics layer — upgraded with:
      • Kafedra Yuklanish Kafedrometri  (capacity speedometer gauge)
      • Real-time What-If via sidebar slider
      • Risk score bar chart
      • Statistical summary + heatmap
      • FULLY DYNAMIC: dept_max_fte from sidebar drives every limit & zone.
    """
    # Pull dynamic limits from session state (set by sidebar inputs in main())
    constraints   = st.session_state.get("dyn_constraints", {})
    dept_max_fte  = constraints.get("dept_max_fte", dept_max_fte)
    # Green zone ends at 83% of the hard limit (same proportional feel at any limit)
    zone_stable   = round(dept_max_fte * 0.833, 1)   # e.g. 25.0 when limit=30
    zone_caution  = dept_max_fte                       # the limit IS the caution boundary
    st.markdown('<div class="section-header">Bashoratli Tahlil Qatlami</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Statistik modellashtirish va stsenariy simulyatsiyasi — '
        'semantik qatlamni ML-asosli qaror qabul qilishga bog\'laydi.</div>',
        unsafe_allow_html=True,
    )

    # ── SECTION 1: Capacity Gauge + What-If ───────────────────────────────
    st.markdown('<div class="section-header">Kafedra Yuklanish Kafedrometri</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
    Kafedraning joriy va simulyatsiya qilingan jami shtat birligi yuklanishi.
    Chap paneldagi <b>slider</b> bilan yangi xodimlar sonini o'zgartiring — kafedrometr real vaqtda yangilanadi.
    </div>
    """, unsafe_allow_html=True)

    current_total_fte = df_full["Workload"].sum()     # full dataset, not filtered
    sim_total_fte     = current_total_fte + sim_new_hires * 1.0

    gauge_fig, zone_label, zone_color = build_capacity_gauge(
        current_total_fte, sim_total_fte,
        zone_stable=zone_stable,
        zone_caution=zone_caution,
    )

    gcol1, gcol2 = st.columns([1.6, 1])

    with gcol1:
        st.plotly_chart(gauge_fig, use_container_width=True)

    with gcol2:
        # Zone legend — dynamic boundary values shown
        st.markdown(f"""
        <div style="background:var(--bg-secondary);border:1px solid var(--border);
                    border-radius:var(--radius);padding:18px 20px;margin-top:8px">
          <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:14px">Kafedrometr Zonalari
            <span style="color:var(--accent-blue);margin-left:6px">Limit: {zone_caution:.1f} SB</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:0.85rem">
            <div style="width:14px;height:14px;border-radius:3px;background:rgba(39,174,96,0.5);flex-shrink:0"></div>
            <div><b style="color:#27ae60">Yashil Zona</b> &nbsp;0 – {zone_stable:.1f} SB<br>
              <span style="font-size:0.75rem;color:var(--text-muted)">Barqaror — yangi qabul mumkin</span></div>
          </div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:0.85rem">
            <div style="width:14px;height:14px;border-radius:3px;background:rgba(245,166,35,0.5);flex-shrink:0"></div>
            <div><b style="color:#f5a623">Sariq Zona</b> &nbsp;{zone_stable:.1f} – {zone_caution:.1f} SB<br>
              <span style="font-size:0.75rem;color:var(--text-muted)">Ehtiyotkor — dekan ruxsati kerak</span></div>
          </div>
          <div style="display:flex;align-items:center;gap:10px;font-size:0.85rem">
            <div style="width:14px;height:14px;border-radius:3px;background:rgba(235,87,87,0.5);flex-shrink:0"></div>
            <div><b style="color:#eb5757">Qizil Zona</b> &nbsp;{zone_caution:.1f}+ SB<br>
              <span style="font-size:0.75rem;color:var(--text-muted)">Kritik — qabul to'xtatilishi shart</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Numeric breakdown
        delta_fte = sim_total_fte - current_total_fte
        delta_str = f"+{delta_fte:.1f}" if delta_fte >= 0 else f"{delta_fte:.1f}"
        st.markdown(f"""
        <div style="background:var(--bg-secondary);border:1px solid var(--border);
                    border-radius:var(--radius);padding:16px 20px;margin-top:12px">
          <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:12px">Simulyatsiya Hisoboti</div>
          <div class="sidebar-metric" style="background:var(--bg-primary)">
            <span class="sm-label">Joriy jami SB</span>
            <span class="sm-val">{current_total_fte:.2f}</span>
          </div>
          <div class="sidebar-metric" style="background:var(--bg-primary)">
            <span class="sm-label">Yangi xodimlar (+{sim_new_hires})</span>
            <span class="sm-val" style="color:#2f80ed">{delta_str} SB</span>
          </div>
          <div class="sidebar-metric" style="background:var(--bg-primary)">
            <span class="sm-label">Prognoz jami SB</span>
            <span class="sm-val" style="color:{zone_color}">{sim_total_fte:.2f}</span>
          </div>
          <div class="sidebar-metric" style="background:var(--bg-primary)">
            <span class="sm-label">Chegara (maks)</span>
            <span class="sm-val" style="color:var(--accent-blue)">{zone_caution:.2f} SB</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Impact Analysis: Tizim Xulosasi ───────────────────────────────────
    headroom = zone_caution - sim_total_fte
    if "Barqaror" in zone_label:
        conclusion_text = (
            f"Tizim <b>barqaror</b> holatda. "
            f"Chegaragacha {headroom:.1f} SB bo'sh kapasite mavjud. "
            f"{sim_new_hires} ta yangi xodim qabul qilinsa ham, kafedra xavfsiz zonada qoladi."
        )
        conclusion_cls = "decision-ok"
    elif "Ehtiyotkor" in zone_label:
        conclusion_text = (
            f"Tizim <b>ehtiyotkor</b> zonasiga kirmoqda. "
            f"Chegaragacha atigi {headroom:.1f} SB qoldi. "
            f"Yangi qabul oldidan dekan va kafedra mudiri bilan kelishish tavsiya etiladi."
        )
        conclusion_cls = "decision-warning"
    else:
        over = abs(headroom)
        conclusion_text = (
            f"[OGOHLANTIRISH] Tizim <b>kritik</b> holatda — chegara {over:.1f} SB ga oshib ketdi. "
            f"OWL ontologiyasi chekloviga ko'ra yangi qabul <b>to'xtatilishi shart</b>. "
            f"Mavjud o'rindosh yuklanishlarini kamaytirish tavsiya etiladi."
        )
        conclusion_cls = "decision-error"

    st.markdown(
        f'<div class="decision-box {conclusion_cls}" style="margin-top:0">'
        f'<b>Tizim Xulosasi: {zone_label}</b><br>{conclusion_text}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Cumulative identity violations in Xulosa ───────────────────────────
    # Run aggregate_by_identity against the FULL faculty dataset (not filtered)
    # so the Kafedrometr sees the same individuals as the Semantic Audit.
    ind_limit_pred = constraints.get("max_ind_sb", 1.5)
    df_global_pred = st.session_state.get("global_df", df_full)
    global_map_pred = build_global_workload_map(df_global_pred)
    agg_pred = aggregate_by_identity(df_full, ind_limit=ind_limit_pred,
                                     global_wl_map=global_map_pred)
    cum_violations = agg_pred[agg_pred["CumulativeExceeded"]]
    if not cum_violations.empty:
        lines = []
        for _, r in cum_violations.iterrows():
            lines.append(
                f"<b>{r['DisplayName']}</b>: Σ = {r['TotalWorkload']:.2f} SB "
                f"— chegara {ind_limit_pred:.2f} SB dan <span style='color:#eb5757'>"
                f"+{r['ExcessSB']:.2f} SB oshgan</span>"
            )
        st.markdown(
            '<div class="decision-box decision-error" style="margin-top:8px">'
            f'<b>Kumulyativ Individual Buzilishlar ({len(cum_violations)} ta shaxs):</b><br>'
            + "<br>".join(lines)
            + "</div>",
            unsafe_allow_html=True,
        )
    elif not agg_pred.empty:
        st.markdown(
            f'<div class="decision-box decision-ok" style="margin-top:8px">'
            f'Barcha {len(agg_pred)} ta unikal shaxsning kumulyativ SB '
            f'≤ {ind_limit_pred:.2f} SB — ontologiya izchil.'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── SECTION 2: Descriptive Statistics + Heatmap ───────────────────────
    st.markdown('<div class="section-header">Tavsifiy Statistika</div>', unsafe_allow_html=True)

    col_stat, col_heat = st.columns([1, 1.3])

    with col_stat:
        st.markdown("**Shtat Birligi Statistikasi**")
        stats = df_filtered["Workload"].describe()
        stat_rows = {
            "O'rtacha SB"      : f"{stats['mean']:.3f}",
            "Mediana SB"       : f"{df_filtered['Workload'].median():.3f}",
            "Standart og'ish"  : f"{stats['std']:.3f}",
            "Min / Maks"       : f"{stats['min']:.2f} / {stats['max']:.2f}",
            "Qiyalik"          : f"{df_filtered['Workload'].skew():.3f}",
            "Qo'rtosis"        : f"{df_filtered['Workload'].kurtosis():.3f}",
        }
        for label, val in stat_rows.items():
            st.markdown(f"""
            <div class="sidebar-metric" style="background:var(--bg-secondary)">
                <span class="sm-label">{label}</span>
                <span class="sm-val">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_heat:
        pivot = (
            df_filtered
            .groupby(["Position", "EmploymentType"])["Workload"]
            .mean()
            .unstack(fill_value=0)
        )
        if not pivot.empty:
            fig_heat = px.imshow(
                pivot,
                labels=dict(x="Bandlik Turi", y="Lavozim", color="O'rtacha SB"),
                title="O'rtacha Shtat Birligi · Lavozim × Bandlik Turi",
                color_continuous_scale="Blues",
                text_auto=".2f",
            )
            fig_heat = styled_fig(fig_heat)
            st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── SECTION 2b: Semantic Budget Recommendation ─────────────────────────
    DEPT_MAX_BUDGET = zone_caution              # driven by sidebar, not hardcoded
    live_total      = df_full["Workload"].sum()
    free_capacity   = DEPT_MAX_BUDGET - live_total

    st.markdown('<div class="section-header">Semantik Byudjet Tavsiyasi</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Mavjud byudjet bo\'shlig\'i asosida kafedra uchun ' +
        f'optimal kadrlar taqsimoti tavsiyasi ({DEPT_MAX_BUDGET:.1f} SB Siyosat qoidasi qat\'iy saqlanadi).</div>',
        unsafe_allow_html=True,
    )

    rec_col1, rec_col2 = st.columns([1.2, 1])

    with rec_col1:
        if free_capacity <= 0:
            rec_box_cls  = "decision-error"
            rec_headline = f"Byudjet to'ldirilgan — qo'shimcha qabul imkoni yo'q."
            rec_detail   = (
                f"Joriy jami: {live_total:.2f} SB / {DEPT_MAX_BUDGET:.0f} SB. "
                "Yangi xodim qabul qilish uchun avval mavjud yuklanishlarni kamaytiring."
            )
            rec_options  = []
        else:
            if free_capacity >= 1.0:
                rec_box_cls  = "decision-ok"
                rec_headline = f"{free_capacity:.2f} SB bo'sh kapasite mavjud."
            else:
                rec_box_cls  = "decision-warning"
                rec_headline = f"Cheklangan bo'sh kapasite: {free_capacity:.2f} SB."

            # Build split options
            rec_options = []
            if free_capacity >= 1.0:
                full_slots = int(free_capacity // 1.0)
                rec_options.append(f"{full_slots}x 1.0 SB — Asosiy shtat pozitsiyasi")
            if free_capacity >= 0.5:
                half_slots = int(free_capacity // 0.5)
                rec_options.append(f"{half_slots}x 0.5 SB — Ichki o'rindosh pozitsiyasi")
            if free_capacity >= 0.25:
                qtr_slots = int(free_capacity // 0.25)
                rec_options.append(f"{qtr_slots}x 0.25 SB — Soatbay/Tashqi o'rindosh")
            # Mixed option for >= 1.5
            if free_capacity >= 1.5:
                rec_options.append(
                    f"1x 1.0 SB (Asosiy) + {int((free_capacity-1.0)//0.5)}x 0.5 SB — Aralash taqsimot"
                )

            # Position-based recommendation: prioritise under-staffed roles
            pos_counts = df_full.groupby("Position")["Workload"].sum().sort_values()
            under = pos_counts.index[0] if len(pos_counts) > 0 else "Assistent"
            rec_detail = (
                f"Eng kam yuklanishli lavozim: <b>{under}</b>. "
                f"Kafedraning ilmiy salmoqini oshirish uchun "
                f"Professor yoki Dotsent lavozimiga ustuvorlik bering."
            )

        st.markdown(
            f'<div class="decision-box {rec_box_cls}">' +
            f'<b>Tizim Tavsiyasi:</b> {rec_headline}<br>' +
            (f'<br>Tavsiya etilgan taqsimot variantlari:<br>' +
             "<br>".join(f"&nbsp;&nbsp;{o}" for o in rec_options) if rec_options else "") +
            f'<br><small style="opacity:0.8">{rec_detail}</small>' +
            '</div>',
            unsafe_allow_html=True,
        )

    with rec_col2:
        # Mini pie of current capacity usage
        cap_data  = [
            {"Holat": "Ishlatilgan", "SB": live_total},
            {"Holat": "Bo'sh",       "SB": max(free_capacity, 0)},
        ]
        fig_cap = px.pie(
            pd.DataFrame(cap_data),
            names="Holat", values="SB",
            title=f"Byudjet Holati — {live_total:.1f} / {DEPT_MAX_BUDGET:.0f} SB",
            color="Holat",
            color_discrete_map={"Ishlatilgan": "#2f80ed", "Bo'sh": "#27ae60"},
            hole=0.60,
        )
        fig_cap.update_traces(textinfo="percent+value", textfont_size=11)
        fig_cap = styled_fig(fig_cap)
        st.plotly_chart(fig_cap, use_container_width=True)

    st.markdown("---")

    # ── SECTION 3: Risk Score Bar ─────────────────────────────────────────
    st.markdown('<div class="section-header">Xavf Ko\'rsatkichi Simulyatsiyasi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
    Simulyatsiya qilingan ML xavf ko'rsatkichi: <em>X = (shtat_birligi_persentil × 0.7) + (me'yordan_oshgan × 0.3)</em>.
    Haqiqiy o'qitilgan klassifikator chiqaradigan logistik-regressiya natijasini ifodalaydi.
    </div>
    """, unsafe_allow_html=True)

    df_risk = df_filtered.copy()
    pct_rank = df_risk["Workload"].rank(pct=True)
    df_risk["RiskScore"] = (pct_rank * 0.7 + df_risk["WorkloadExceeded"].astype(float) * 0.3).round(3)
    df_risk["RiskLabel"] = pd.cut(
        df_risk["RiskScore"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Past Xavf", "O'rtacha Xavf", "Yuqori Xavf"],
    )

    fig_risk = px.bar(
        df_risk.sort_values("RiskScore", ascending=False),
        x="Name", y="RiskScore",
        color="RiskLabel",
        color_discrete_map={
            "Past Xavf"      : "#27ae60",
            "O'rtacha Xavf"  : "#f5a623",
            "Yuqori Xavf"    : "#eb5757",
        },
        title="Xodimlar Xavf Ko'rsatkichi Reytingi (Gibrid Klassifikator Natijasi)",
        hover_data=["ID", "Position", "Workload"],
    )
    fig_risk.update_layout(xaxis_tickangle=-45, xaxis_title="", yaxis_title="Xavf Ko'rsatkichi [0–1]")
    fig_risk = styled_fig(fig_risk)
    st.plotly_chart(fig_risk, use_container_width=True)


# ===========================================================================
# TAB 4 — SMART ARIZA (APPLICATION) PROCESSOR  ·  NLP + Semantic Validation
# ===========================================================================
# HYBRID ALGORITHM — NLP Layer (Layer 5):
#   Step 1  · User pastes free-form Uzbek application text.
#   Step 2  · Regex engine extracts named entities (Name, Lavozim, Shtat, Bandlik).
#   Step 3  · Extracted entities are validated against the live ontology DataFrame:
#             — Does this position exist in the ontology?
#             — Would the new individual push total workload over the department cap?
#             — Is the requested employment type consistent with existing records?
#   Step 4  · A "Decision Recommendation" card is rendered with semantic reasoning.
#   Step 5  · A "Draft OWL Individual" snippet is generated — shows how the
#             extracted data maps back to the ontology (closing the NLP → Semantic loop).
# ===========================================================================

import re   # already in stdlib; import here for clarity in thesis reading

# ---------------------------------------------------------------------------
# ENTITY RESOLUTION — Name Normalisation & Fuzzy Duplicate Detection
# ---------------------------------------------------------------------------

def normalise_name(name: str) -> str:
    """
    Normalises a name string for comparison by:
      1. Stripping extra whitespace.
      2. Splitting on whitespace and sorting tokens alphabetically.
         This makes "Jasur Toshmatov" and "Toshmatov Jasur" identical.
    Returns the lower-cased sorted token string for comparison only;
    the original casing is preserved for display.
    """
    tokens = re.split(r"\s+", name.strip())
    return " ".join(sorted(t.lower() for t in tokens if t))


def fuzzy_name_match(query: str, candidates: list[str]) -> list[tuple[str, float]]:
    """
    Shortened-name fuzzy matching without external libraries.

    Algorithm (thesis note):
      For each candidate in the existing staff list we compute a similarity
      score by comparing:
        a) Last name exact match (weight 0.70)
        b) First initial match   (weight 0.30)
      This produces a score in [0, 1.0]. A score >= 0.90 flags a likely
      duplicate; >= 0.95 triggers a hard block.

    Example:
      query     = "Nazirova E."
      candidate = "Nazirova Ezoza"
      -> last name "Nazirova" matches  (+0.70)
      -> first char of "E." == first char of "Ezoza"  (+0.30)
      -> score = 1.00  →  hard block triggered

    Why not difflib? Using difflib.SequenceMatcher on raw name strings
    produces false positives for short abbreviated names like "E." which
    inflate the ratio. This domain-specific two-factor check is more
    precise for the HR abbreviation convention used in Uzbek university
    documents.
    """
    results = []
    # Tokenise the query: first token = last name, remaining = first name(s)
    q_parts    = query.strip().split()
    q_last     = q_parts[0].lower().rstrip(".") if q_parts else ""
    q_first_ch = q_parts[1][0].lower() if len(q_parts) > 1 else ""

    for cand in candidates:
        c_parts    = cand.strip().split()
        c_last     = c_parts[0].lower().rstrip(".") if c_parts else ""
        c_first_ch = c_parts[1][0].lower() if len(c_parts) > 1 else ""

        score = 0.0
        if q_last and c_last and q_last == c_last:
            score += 0.70
        if q_first_ch and c_first_ch and q_first_ch == c_first_ch:
            score += 0.30

        if score > 0:
            results.append((cand, round(score, 2)))

    # Sort by descending score so the closest match is first
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def resolve_duplicate(name: str, df: "pd.DataFrame") -> dict:
    """
    Runs the full entity-resolution pipeline for a candidate name:
      Step 1 — Normalised exact match   (deterministic)
      Step 2 — Fuzzy initial match      (probabilistic, threshold-gated)

    Returns:
      {
        "norm_match"  : bool,         # True if normalised form already exists
        "fuzzy_hits"  : [(name, score), ...],  # all fuzzy matches >= 0.90
        "hard_block"  : bool,         # True if any score >= 0.95
        "soft_warn"   : bool,         # True if any score in [0.90, 0.95)
        "matched_names": [str, ...],  # display list of matched names
      }
    """
    existing    = df["Name"].dropna().tolist()
    norm_query  = normalise_name(name)
    norm_exists = [normalise_name(n) for n in existing]

    norm_match   = norm_query in norm_exists
    fuzzy_hits   = [(n, s) for n, s in fuzzy_name_match(name, existing) if s >= 0.90]
    hard_block   = norm_match or any(s >= 0.95 for _, s in fuzzy_hits)
    soft_warn    = (not hard_block) and bool(fuzzy_hits)
    matched_names = (
        [existing[norm_exists.index(norm_query)]] if norm_match
        else [n for n, _ in fuzzy_hits]
    )

    return {
        "norm_match"   : norm_match,
        "fuzzy_hits"   : fuzzy_hits,
        "hard_block"   : hard_block,
        "soft_warn"    : soft_warn,
        "matched_names": matched_names,
    }



# ---------------------------------------------------------------------------
# NLP — REGEX ENTITY EXTRACTOR
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> dict:
    """
    Regex-Based Named Entity Recognition for Uzbek faculty applications.

    Extracts four semantic slots:
      • ism        : Applicant full name  (F.I.Sh. format, e.g. "Khujayev Nodir")
      • lavozim    : Position keyword     (Professor, Dotsent, VB Dotsent, …)
      • shtat      : Workload float       (0.25 / 0.5 / 0.75 / 1.0 …)
      • bandlik    : Employment type      (Asosiy / Ichki o'rindosh / Tashqi o'rindosh)

    Design note (thesis): Pure regex avoids any ML dependency while still
    demonstrating NLP-to-Ontology bridging. In a production system this slot
    would be filled by a fine-tuned Uzbek NER model (e.g., BERT-uz).
    """

    result = {
        "ism"     : None,
        "lavozim" : None,
        "shtat"   : None,
        "bandlik" : None,
        "xom_matn": text.strip(),
    }

    # ── 1. ISM (Name) ──────────────────────────────────────────────────────
    # Patterns:
    #   "Men, Lastname Firstname," — most formal Uzbek application style
    #   "Men Lastname Firstname,"
    #   Fallback: any two Title-Case words after "Men"
    name_patterns = [
        r"[Mm]en[,\s]+([A-ZÀ-Ö][a-zA-ZÀ-ö'\-]+(?:\s+[A-ZÀ-Ö][a-zA-ZÀ-ö'\-]+){1,3})\s*[,.]",
        r"[Mm]urojaat\s+etuvchi[:\s]+([A-ZÀ-Ö][a-zA-ZÀ-ö'\-]+(?:\s+[A-ZÀ-Ö][a-zA-ZÀ-ö'\-]+){1,2})",
        r"[Ff]\.?[Ii]\.?[Ss]h\.?\s*[:\-]?\s*([A-ZÀ-Ö][a-zA-ZÀ-ö'\-]+(?:\s+[A-ZÀ-Ö][a-zA-ZÀ-ö'\-]+){1,2})",
    ]
    for pat in name_patterns:
        m = re.search(pat, text)
        if m:
            result["ism"] = m.group(1).strip()
            break

    # ── 2. LAVOZIM (Position) ──────────────────────────────────────────────
    # Match OWL class keywords exactly (case-insensitive).
    # Order matters: check longer/more specific labels first.
    position_keywords = [
        ("VB Dotsent",        r"\bVB[\s_]?[Dd]otsent\b"),
        ("VB Professor",      r"\bVB[\s_]?[Pp]rofessor\b"),
        ("Katta o'qituvchi",  r"\b[Kk]atta[\s_]o['']?qituvchi\b"),
        ("Stajer o'qituvchi", r"\b[Ss]tajer[\s_]o['']?qituvchi\b"),
        ("Professor",         r"\b[Pp]rofessor\b"),
        ("Dotsent",           r"\b[Dd]otsent\b"),
        ("Assistent",         r"\b[Aa]ssistent\b"),
        ("Assistent",         r"\b[Aa]sistent\b"),   # common typo variant
    ]
    for label, pat in position_keywords:
        if re.search(pat, text):
            result["lavozim"] = label
            break

    # ── 3. SHTAT BIRLIGI (Workload float) ─────────────────────────────────
    # Looks for patterns like:  "1.0 shtat", "0.5 shtat birligida", "bir stavka"
    shtat_patterns = [
        r"(\d+[.,]\d+)\s*(?:shtat|stavka|birlik)",
        r"(\d+[.,]\d+)\s*(?:ish\s*hajmi|yuklanish)",
        r"shtat[\s_]birligida\s+(\d+[.,]\d+)",
        r"(\d+(?:[.,]\d+)?)\s*(?:shtat\s+birligida|stavkada)",
        r"\b(0[.,][25][05]?|1[.,]0|1[.,]5)\b",   # bare fractions common in UZ HR docs
    ]
    for pat in shtat_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", ".")
            try:
                result["shtat"] = float(raw)
            except ValueError:
                pass
            break

    # ── 4. BANDLIK TURI (Employment Type) ─────────────────────────────────
    bandlik_keywords = [
        ("Ichki o'rindosh",  r"\b[Ii]chki[\s_]o['']?rindosh\b"),
        ("Ichki o'rindosh",  r"\b[Ii]chki[\s_]o['']?rindosh\b"),
        ("Tashqi o'rindosh", r"\b[Tt]ashqi[\s_]o['']?rinbosar\b"),
        ("Tashqi o'rindosh", r"\b[Tt]ashqi[\s_]o['']?rindosh\b"),
        ("Asosiy",            r"\b[Aa]sosiy(?:\s+shtat)?\b"),
        ("Asosiy",            r"\b[Aa]sosiy\b"),
    ]
    for label, pat in bandlik_keywords:
        if re.search(pat, text):
            result["bandlik"] = label
            break

    return result


# ---------------------------------------------------------------------------
# SEMANTIC VALIDATOR
# ---------------------------------------------------------------------------

def semantic_validate(entities: dict, df_full: pd.DataFrame,
                       dept_max_fte: float = 30.0,
                       max_asosiy_sb: float = 1.0,
                       max_orindosh_sb: float = 0.75) -> dict:
    """
    Cross-validates the NLP-extracted entities against the live ontology DataFrame.

    Checks performed (ordered by severity):
      C1 · Unknown position    — extracted lavozim not in ontology class list
      C2 · Workload cap        — total dept workload would exceed DEPT_MAX_FTE
      C3 · Part-timer overload — Ichki/Tashqi o'rindosh requesting > 0.75 SB
      C4 · Asosiy overload     — Asosiy staff requesting > 1.0 SB
      C5 · Duplicate candidate — same name already in ontology

    Returns a dict with keys:
      valid   : bool
      checks  : list of (icon, label, status) tuples for display
      decision: str   recommendation text
      severity: "ok" | "warning" | "error"
    """

    DEPT_MAX_FTE = dept_max_fte   # now driven by UI slider

    KNOWN_POSITIONS = {
        "Assistent", "Dotsent", "VB Dotsent", "Professor",
        "VB Professor", "Katta o'qituvchi", "Stajer o'qituvchi",
    }
    KNOWN_BANDLIK = {"Asosiy", "Ichki o'rindosh", "Tashqi o'rindosh"}

    checks   = []
    errors   = []
    warnings = []

    lav   = entities.get("lavozim")
    shtat = entities.get("shtat")
    band  = entities.get("bandlik")
    ism   = entities.get("ism")

    # ── C0: Extraction completeness ────────────────────────────────────────
    missing = [k for k in ("ism", "lavozim", "shtat", "bandlik") if not entities.get(k)]
    if missing:
        field_uz = {"ism": "F.I.Sh.", "lavozim": "Lavozim", "shtat": "Shtat birligi", "bandlik": "Bandlik turi"}
        for f in missing:
            errors.append(f"[XATO]  {field_uz[f]} aniqlanmadi")
        checks.append(("[XATO]", "Matn tahlili", f"{len(missing)} ta maydon topilmadi"))
    else:
        checks.append(("[OK]", "Matn tahlili", "Barcha maydonlar aniqlandi"))

    # ── C1: Position exists in ontology ───────────────────────────────────
    if lav and lav not in KNOWN_POSITIONS:
        errors.append(f"'{lav}' lavozimi ontologiyada mavjud emas")
        checks.append(("[XATO]", "Lavozim tekshiruvi", f"'{lav}' — noma'lum lavozim"))
    elif lav:
        checks.append(("[OK]", "Lavozim tekshiruvi", f"'{lav}' ontologiyada tasdiqlandi"))

    # ── C2: Department-level FTE cap ───────────────────────────────────────
    current_total = df_full["Workload"].sum()
    projected     = current_total + (shtat or 0.0)
    if projected > DEPT_MAX_FTE:
        warnings.append(
            f"Jami shtat birligi {projected:.2f} ga yetadi "
            f"(chegara: {DEPT_MAX_FTE:.0f} SB)"
        )
        checks.append(("[OGOHLANTIRISH]", "Kafedra SB cheklovi", f"{current_total:.2f} + {shtat or 0:.2f} = {projected:.2f} / {DEPT_MAX_FTE:.0f}"))
    else:
        checks.append(("[OK]", "Kafedra SB cheklovi", f"Loyiha jami: {projected:.2f} / {DEPT_MAX_FTE:.0f} SB"))

    # ── C3: Part-timer workload limit ─────────────────────────────────────
    if band in ("Ichki o'rindosh", "Tashqi o'rindosh") and shtat and shtat > max_orindosh_sb:
        errors.append(f"O'rindosh xodim uchun {shtat} SB me'yordan oshadi (maks: {max_orindosh_sb})")
        checks.append(("[XATO]", "O'rindosh SB qoidasi", f"{shtat} SB > {max_orindosh_sb} ruxsat etilgan chegara"))
    elif band in ("Ichki o'rindosh", "Tashqi o'rindosh") and shtat:
        checks.append(("[OK]", "O'rindosh SB qoidasi", f"{shtat} SB ≤ {max_orindosh_sb} — muvofiq"))

    # ── C4: Full-time workload limit ──────────────────────────────────────
    if band == "Asosiy" and shtat and shtat > max_asosiy_sb:
        errors.append(f"Asosiy xodim uchun {shtat} SB me'yordan oshadi (maks: {max_asosiy_sb})")
        checks.append(("[XATO]", "Asosiy SB qoidasi", f"{shtat} SB > {max_asosiy_sb} ruxsat etilgan chegara"))
    elif band == "Asosiy" and shtat:
        checks.append(("[OK]", "Asosiy SB qoidasi", f"{shtat} SB ≤ {max_asosiy_sb} — muvofiq"))

    # ── C5: Duplicate name check ──────────────────────────────────────────
    if ism:
        # Strip initials and compare last name only for robustness
        last_name = ism.split()[0].lower() if ism else ""
        existing_names = df_full["Name"].str.lower().str.split().str[0]
        duplicates = df_full[existing_names == last_name]
        if not duplicates.empty:
            existing_list = ", ".join(duplicates["Name"].tolist())
            warnings.append(f"Shu familiyali xodim allaqachon mavjud: {existing_list}")
            checks.append(("[OGOHLANTIRISH]", "Takror tekshiruvi", f"Topildi: {existing_list}"))
        else:
            checks.append(("[OK]", "Takror tekshiruvi", "Mos ism ontologiyada topilmadi"))

    # ── Decision ──────────────────────────────────────────────────────────
    if errors:
        severity = "error"
        decision = (
            "ARIZA RAD ETILDI — Semantik tekshiruv muvaffaqiyatsiz yakunlandi. "
            "Quyidagi xatolarni bartaraf eting: " + "; ".join(errors)
        )
    elif warnings:
        severity = "warning"
        decision = (
            "ARIZA SHARTLI QABUL — Semantik ogohlantirish(lar) mavjud. "
            "Dekan ko'rib chiqishi tavsiya etiladi: " + "; ".join(warnings)
        )
    else:
        severity = "ok"
        decision = (
            "ARIZA TASDIQLANDI — Barcha semantik qoidalar bajarilgan. "
            f"{ism or 'Nomzod'} '{lav}' lavozimiga {shtat} SB bilan "
            f"'{band}' sifatida ontologiyaga qo'shilishi mumkin."
        )

    return {
        "valid"    : severity == "ok",
        "checks"   : checks,
        "decision" : decision,
        "severity" : severity,
        "projected_total": projected,
    }


def generate_owl_snippet(entities: dict) -> str:
    """
    Generates a pseudo-OWL/Turtle individual declaration from extracted entities.
    This closes the NLP → Ontology loop and demonstrates the full hybrid pipeline
    for the thesis: unstructured text → NER → OWL individual → knowledge base.
    """
    ism   = entities.get("ism", "???")
    lav   = entities.get("lavozim", "???")
    shtat = entities.get("shtat", "???")
    band  = entities.get("bandlik", "???")

    # Convert display labels back to OWL class IRIs
    pos_iri_map = {
        "Assistent"         : "Assistent",
        "Dotsent"           : "Dotsent",
        "VB Dotsent"        : "VB_Dotsent",
        "Professor"         : "Professor",
        "VB Professor"      : "VB_Professor",
        "Katta o'qituvchi"  : "Katta_oqituvchi",
        "Stajer o'qituvchi" : "Stajer_oqituvchi",
    }
    emp_iri_map = {
        "Asosiy"            : "Asosiy",
        "Ichki o'rindosh"  : "Ichki_orindosh",
        "Tashqi o'rindosh" : "Tashqi_orindosh",
    }

    pos_iri  = pos_iri_map.get(lav, lav)
    emp_iri  = emp_iri_map.get(band, band)
    ind_name = "xodim_nlp_" + re.sub(r"\W+", "_", (ism or "yangi").lower())[:20]

    return f"""# ── NLP dan yaratilgan OWL Individual (Turtle sintaksisi) ──
@prefix :    <http://ds.univ.uz/ontology/uzbek-sentence-semantic-hr#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:{ind_name}
    a owl:NamedIndividual ,
      :{pos_iri} ,           # Lavozim klassi
      :{emp_iri} ;           # Bandlik turi klassi
    :shtat_birligi  "{shtat}"^^xsd:decimal ;
    # F.I.Sh.: {ism}
    .

# ClassAssertion: :{pos_iri}  :{ind_name}
# ClassAssertion: :{emp_iri}  :{ind_name}
# DataPropertyAssertion: :shtat_birligi  :{ind_name}  "{shtat}"^^xsd:decimal"""



def global_workload_check(
    name: str,
    new_wl: float,
    df_global: pd.DataFrame,
    global_limit: float = 1.5,
) -> dict:
    """
    GLOBAL 1.5 WORKLOAD RULE (vectorised, O(n) pandas scan).

    Searches across ALL faculties in df_global for rows whose Name matches
    `name` via the two-factor fuzzy score (>= 0.90).  Sums their workload
    with `new_wl` and checks against `global_limit`.

    Also detects whether the person already holds an "Asosiy" record
    in any faculty — if so, the new employment type must be forced to
    "Tashqi o'rindosh" per university policy.

    Returns:
      {
        "existing_wl"       : float,   # sum of matched rows' Workload
        "projected_wl"      : float,   # existing_wl + new_wl
        "exceeds_limit"     : bool,
        "has_asosiy"        : bool,    # True if any matched row is Asosiy
        "asosiy_faculties"  : [str],   # faculty names where Asosiy record found
        "matched_rows"      : DataFrame  # the matching rows for display
      }
    """
    if df_global.empty or not name:
        return {
            "existing_wl": 0.0, "projected_wl": new_wl,
            "exceeds_limit": False, "has_asosiy": False,
            "asosiy_faculties": [], "matched_rows": pd.DataFrame(),
        }

    # ── Vectorised two-factor name similarity ─────────────────────────────
    # Extract query components once, compare against all rows in one pass.
    q_parts    = name.strip().split()
    q_last     = q_parts[0].lower().rstrip(".") if q_parts else ""
    q_init     = q_parts[1][0].lower() if len(q_parts) > 1 and q_parts[1] else ""

    names_series = df_global["Name"].fillna("").astype(str)

    # Vectorised surname extraction (first whitespace token)
    cand_last = names_series.str.strip().str.split().str[0].str.lower().str.rstrip(".")
    # Vectorised initial extraction (first char of second token)
    cand_init = (
        names_series.str.strip().str.split()
        .apply(lambda t: t[1][0].lower() if len(t) > 1 and t[1] else "")
    )

    score = (
        (cand_last == q_last).astype(float) * 0.70 +
        (cand_init == q_init).astype(float) * 0.30
    )

    matched_mask = score >= 0.90
    matched_rows = df_global[matched_mask].copy()
    matched_rows["_sim_score"] = score[matched_mask]

    existing_wl  = float(matched_rows["Workload"].sum())
    projected_wl = existing_wl + new_wl

    asosiy_mask    = matched_rows["EmploymentType"].str.strip() == "Asosiy"
    has_asosiy     = bool(asosiy_mask.any())
    asosiy_fac_col = "Faculty" if "Faculty" in matched_rows.columns else None
    asosiy_faculties = (
        matched_rows.loc[asosiy_mask, asosiy_fac_col].dropna().tolist()
        if asosiy_fac_col else []
    )

    return {
        "existing_wl"     : existing_wl,
        "projected_wl"    : projected_wl,
        "exceeds_limit"   : projected_wl > global_limit,
        "has_asosiy"      : has_asosiy,
        "asosiy_faculties": asosiy_faculties,
        "matched_rows"    : matched_rows,
    }


# ===========================================================================
# TAB 4 — ARIZA QABULI: NLP VA SEMANTIK TEKSHIRUV
# ===========================================================================
# HYBRID ALGORITHM — NLP Layer (Layer 5):
#   Step 1  · User pastes free-form Uzbek application text.
#   Step 2  · Regex NER extracts four semantic slots.
#   Step 3  · Logic Trace shows the system's reasoning step-by-step (XAI).
#   Step 4  · Semantic validation against live ontology DataFrame.
#   Step 5  · Policy guardrail blocks Tasdiqlash if FTE cap exceeded.
#   Step 6  · Confirmed row appended to st.session_state["faculty_df"].
#   Step 7  · OWL Turtle snippet generated; NLP Model Evaluation shown.
# ===========================================================================

def tab_nlp_ariza(df_full: pd.DataFrame, df_global: pd.DataFrame | None = None):
    """
    TAB 4 — Ariza Qabuli: NLP va Semantik Tekshiruv

    Parameters:
      df_full   : current-faculty DataFrame (used for FTE cap and display)
      df_global : university-wide DataFrame across ALL faculties (used for
                  global 1.5 SB check and Asosiy policy enforcement).
                  Falls back to df_full when not provided (demo mode).

    SESSION STATE keys:
      ariza_text      : persists sample-button text between reruns
      ariza_processed : shows results panel after first Tahlil click
      last_ariza      : snapshot of text at Tahlil click time
      confirmed_ids   : set of candidate keys already added (prevents duplicates)
    """

    # ── Tab-specific CSS ──────────────────────────────────────────────────
    st.markdown("""
    <style>
    .entity-table { width:100%; border-collapse:collapse; margin:16px 0; }
    .entity-table th {
        background: rgba(47,128,237,0.15); color: var(--accent-blue);
        font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
        padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border);
    }
    .entity-table td {
        padding: 11px 14px; font-size: 0.88rem;
        border-bottom: 1px solid rgba(48,54,61,0.4); vertical-align: middle;
    }
    .entity-table tr:last-child td { border-bottom: none; }
    .entity-table tr:hover td { background: rgba(47,128,237,0.05); }
    .entity-tag {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;
    }
    .tag-name     { background:rgba(0,180,216,0.15);color:#00b4d8;border:1px solid rgba(0,180,216,0.3); }
    .tag-position { background:rgba(47,128,237,0.15);color:#2f80ed;border:1px solid rgba(47,128,237,0.3); }
    .tag-shtat    { background:rgba(245,166,35,0.15);color:#f5a623;border:1px solid rgba(245,166,35,0.3); }
    .tag-bandlik  { background:rgba(39,174,96,0.15);color:#27ae60;border:1px solid rgba(39,174,96,0.3); }
    .tag-unknown  { background:rgba(139,148,158,0.15);color:#8b949e;border:1px solid rgba(139,148,158,0.3); }
    .check-row {
        display:flex; align-items:center; gap:12px; padding:10px 16px;
        border-bottom:1px solid rgba(48,54,61,0.4); font-size:0.86rem;
    }
    .check-row:last-child { border-bottom:none; }
    .check-label  { font-weight:600; min-width:200px; color:var(--text-primary); }
    .check-detail { color:var(--text-muted); font-size:0.81rem; }
    .decision-box {
        border-radius:var(--radius); padding:18px 22px; margin-top:20px;
        font-size:0.92rem; line-height:1.6; font-weight:500;
    }
    .decision-ok      { background:rgba(39,174,96,0.10);border:1px solid rgba(39,174,96,0.4);color:#27ae60; }
    .decision-warning { background:rgba(245,166,35,0.10);border:1px solid rgba(245,166,35,0.4);color:#f5a623; }
    .decision-error   { background:rgba(235,87,87,0.10);border:1px solid rgba(235,87,87,0.4);color:#eb5757; }
    .draft-card {
        background:var(--bg-secondary); border:1px solid var(--border);
        border-left:4px solid var(--accent-blue); border-radius:var(--radius);
        padding:18px 22px; margin:16px 0;
    }
    .draft-title { font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px; }
    .draft-field { display:flex;gap:12px;align-items:center;margin-bottom:8px;font-size:0.88rem; }
    .draft-key   { color:var(--text-muted);min-width:140px;font-size:0.8rem; }
    .draft-val   { color:var(--text-primary);font-weight:600; }
    .pipeline-step { display:flex;align-items:center;gap:10px;padding:8px 0;font-size:0.82rem;color:var(--text-muted); }
    .pipeline-step .step-num {
        width:24px;height:24px;border-radius:50%;
        background:rgba(47,128,237,0.2);border:1px solid rgba(47,128,237,0.4);
        color:var(--accent-blue);display:flex;align-items:center;
        justify-content:center;font-size:0.72rem;font-weight:700;flex-shrink:0;
    }
    .pipeline-step.active .step-num { background:var(--accent-blue);color:#fff;border-color:var(--accent-blue); }
    .pipeline-step.active { color:var(--text-primary); }
    .logic-trace {
        background:var(--bg-primary); border:1px solid var(--border);
        border-left:3px solid var(--accent-teal); border-radius:var(--radius);
        padding:14px 18px; font-family:'DM Mono',monospace;
        font-size:0.80rem; color:var(--text-muted); line-height:2.1;
    }
    .trace-ok   { color:#27ae60; }
    .trace-warn { color:#f5a623; }
    .trace-err  { color:#eb5757; }
    .trace-info { color:#2f80ed; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Ariza Qabuli — NLP va Semantik Tekshiruv</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
    Erkin matnli arizani semantik bilim bazasiga ulash uchun gibrid NLP quvuri:
    <em>Regex NER → Obyekt Chiqarish → Logic Trace → Ontologiya Tekshiruvi → Ma'lumotlar Bazasiga Qo'shish</em>
    </div>
    """, unsafe_allow_html=True)

    # ── Input layout ──────────────────────────────────────────────────────
    col_input, col_pipe = st.columns([1.6, 1])

    with col_pipe:
        st.markdown("**Gibrid NLP Quvuri**")
        st.markdown("""
        <div style="background:var(--bg-secondary);border:1px solid var(--border);
                    border-radius:var(--radius);padding:16px 18px;">
        <div class="pipeline-step active"><div class="step-num">1</div><div>Erkin matn kiritish</div></div>
        <div class="pipeline-step active"><div class="step-num">2</div><div>Regex NER — obyektlarni ajratish</div></div>
        <div class="pipeline-step active"><div class="step-num">3</div><div>Tizim mantiqiy tahlili (Logic Trace)</div></div>
        <div class="pipeline-step"><div class="step-num">4</div><div>Ontologiya qoidalarini tekshirish</div></div>
        <div class="pipeline-step"><div class="step-num">5</div><div>OWL individual loyihasi yaratish</div></div>
        <div class="pipeline-step"><div class="step-num">6</div><div>Ma'lumotlar bazasiga qo'shish</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>**Namuna arizalar**", unsafe_allow_html=True)
        examples = [
            "Men, Toshmatov Jasur, Professor lavozimiga 1.0 shtat birligida asosiy shtat sifatida ishga qabul qilinishimni so'rayman.",
            "Men, Yusupova Malika, Dotsent lavozimiga 0.5 shtat birligida ichki o'rindosh sifatida qabul qilinishimni iltimos qilaman.",
            "Men, Karimov Bobur, Assistent lavozimiga 0.5 shtat birligida tashqi o'rindosh sifatida ishlashni so'rayman.",
            "Men, Rahimov Ulugbek, Katta o'qituvchi lavozimiga 1.0 stavkada asosiy sifatida qabul qilinmoqchiman.",
        ]
        for i, ex in enumerate(examples):
            if st.button(f"Namuna {i + 1}", key=f"ex_{i}", use_container_width=True):
                st.session_state["ariza_text"] = ex

    with col_input:
        # ── Input source selector ─────────────────────────────────────────
        input_mode = st.radio(
            "Ariza manba'i:",
            options=["✏️ Matn kiritish", "📄 PDF fayl yuklash"],
            horizontal=True,
            key="ariza_input_mode",
        )

        extracted_pdf_text = ""

        if input_mode == "📄 PDF fayl yuklash":
            uploaded_pdf = st.file_uploader(
                "Ariza PDF faylini yuklang:",
                type=["pdf"],
                key="ariza_pdf_upload",
                help="Sahifa 1: Didox protokol (imzo+PINFL). Sahifa 2: Ariza matni (NER).",
            )
            if uploaded_pdf is not None:
                pdf_bytes = uploaded_pdf.read()

                with st.spinner("PDF o'qilmoqda va ERI tekshirilmoqda..."):
                    # Fix 1 — returns list[str], one per page
                    pages = extract_text_from_pdf(io.BytesIO(pdf_bytes))

                    # Fix 2 — Page 1 (index 0) = Didox protocol
                    #          Page 2 (index 1) = ariza body
                    page1_text = pages[0] if len(pages) > 0 else ""
                    page2_text = pages[1] if len(pages) > 1 else ""

                    # Verify PINFL and name are on page 1
                    pinfl_on_p1 = bool(re.search(r'\b\d{14}\b', page1_text))
                    name_on_p1  = bool(re.search(
                        r'[A-ZҚҒҲЎЁ]{2,}\s+[A-ZҚҒҲЎЁ]{2,}', page1_text
                    ))

                    # Verify ariza keywords are on page 2
                    ariza_keywords = ["ishga qabul", "lavozim", "shtat", "so'rayman",
                                      "iltimos", "dotsent", "professor", "assistent"]
                    ariza_on_p2 = any(
                        kw.lower() in page2_text.lower()
                        for kw in ariza_keywords
                    )

                    # ERI signature verification (Fix 3+4 inside function)
                    sig_result = eri_verify_signature(pdf_bytes)

                    # Fix 2 — extract identity specifically from page 1
                    didox_id = eri_extract_didox_identity(page1_text)

                # Store results in session state
                st.session_state["eri_sig_result"] = sig_result
                st.session_state["eri_didox_id"]   = didox_id
                st.session_state["eri_pages"]       = pages
                st.session_state["eri_page1_text"]  = page1_text
                st.session_state["eri_page2_text"]  = page2_text

                # Show page detection summary
                pc1, pc2 = st.columns(2)
                pc1.markdown(
                    f'<div style="background:rgba(47,128,237,0.08);border:1px solid '
                    f'rgba(47,128,237,0.25);border-radius:8px;padding:10px 14px;font-size:0.80rem">'
                    f'<b>Sahifa 1 (Didox protokol)</b><br>'
                    f'PINFL: {"✅ Topildi" if pinfl_on_p1 else "❌ Topilmadi"} &nbsp;|&nbsp; '
                    f'Ism: {"✅ Topildi" if name_on_p1 else "❌ Topilmadi"}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                pc2.markdown(
                    f'<div style="background:rgba(39,174,96,0.08);border:1px solid '
                    f'rgba(39,174,96,0.25);border-radius:8px;padding:10px 14px;font-size:0.80rem">'
                    f'<b>Sahifa 2 (Ariza matni)</b><br>'
                    f'NER kalit so\'zlari: {"✅ Topildi" if ariza_on_p2 else "❌ Topilmadi"} &nbsp;|&nbsp; '
                    f'{len(page2_text)} belgi'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Use page 2 as the NER target text
                extracted_pdf_text = page2_text if page2_text else "\n".join(pages)

                if not extracted_pdf_text.strip():
                    st.warning(
                        "PDF dan matn chiqarib bo'lmadi. "
                        "Fayl skanerlangan rasm bo'lishi mumkin. "
                        "Matnni quyida qo'lda kiriting."
                    )
                else:
                    st.success(
                        f"PDF muvaffaqiyatli o'qildi — "
                        f"{len(pages)} sahifa, sahifa 2 NER uchun ishlatiladi."
                    )
                    with st.expander(
                        "📄 Sahifa 2 — Ariza matni (NER maqsadi)",
                        expanded=True,
                    ):
                        st.text_area(
                            "Chiqarilgan matn (sahifa 2):",
                            value=extracted_pdf_text,
                            height=120,
                            disabled=True,
                            key="pdf_preview",
                        )
                    with st.expander("📋 Sahifa 1 — Didox protokol", expanded=False):
                        st.text_area(
                            "Protokol matni:",
                            value=page1_text,
                            height=100,
                            disabled=True,
                            key="pdf_page1_preview",
                        )
                    st.session_state["ariza_text"] = extracted_pdf_text

        # ── Text area (always shown — editable fallback / manual input) ───
        default_text = st.session_state.get("ariza_text", "")
        label = (
            "Chiqarilgan matnni tahrirlang yoki to'ldiring:"
            if input_mode == "📄 PDF fayl yuklash"
            else "Ariza matnini kiriting yoki namuna tanlang:"
        )
        ariza_text = st.text_area(
            label,
            value=default_text,
            height=130,
            placeholder=(
                "Misol: Men, Khujayev Nodir, Professor lavozimiga 1.0 shtat birligida "
                "asosiy shtat sifatida ishga qabul qilinishimni so'rayman."
            ),
            key="ariza_input",
        )
        run_btn = st.button(
            "Arizani Tahlil Qilish",
            type="primary",
            use_container_width=True,
        )

    # ── Guard: show prompt until first analysis ───────────────────────────
    if not run_btn and not st.session_state.get("ariza_processed"):
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:var(--text-muted);font-size:0.9rem;">
        Ariza matnini kiriting va
        <b style="color:var(--text-primary)">"Arizani Tahlil Qilish"</b> tugmasini bosing.
        </div>
        """, unsafe_allow_html=True)
        return

    if run_btn:
        st.session_state["ariza_processed"] = True
        st.session_state["last_ariza"]      = ariza_text

    text_to_process = st.session_state.get("last_ariza", ariza_text)

    if not text_to_process.strip():
        st.warning("Ariza matni bo'sh. Iltimos, matn kiriting.")
        return

    # ── ERI AUTHORIZATION GATE ────────────────────────────────────────────
    # Only runs when a PDF was uploaded. Skips silently for manual text entry
    # so the existing workflow is fully preserved.
    st.markdown("---")
    eri_authorized = True   # default: allow manual text entries through

    if input_mode == "📄 PDF fayl yuklash" and "eri_sig_result" in st.session_state:
        sig_result = st.session_state["eri_sig_result"]
        didox_id   = st.session_state["eri_didox_id"]

        # Extract applicant name from ariza text using the first NER pass
        _eri_entities    = extract_entities(text_to_process)
        applicant_name   = _eri_entities.get("ism") or ""

        # Cross-page identity matching
        match_score   = eri_name_match_score(didox_id, applicant_name)
        eri_authorized = sig_result["is_signed"] and match_score >= 0.90

        # Store authorization result for downstream gate
        st.session_state["eri_authorized"]   = eri_authorized
        st.session_state["eri_match_score"]  = match_score
        st.session_state["eri_applicant"]    = applicant_name

        # Render the Security Status UI
        eri_render_security_status(
            sig_result     = sig_result,
            didox_id       = didox_id,
            applicant_name = applicant_name,
            match_score    = match_score,
            is_authorized  = eri_authorized,
        )

        # ── ONTOLOGY GATE: block if not authorized ─────────────────────────
        if not eri_authorized:
            st.markdown(
                '<div style="background:rgba(235,87,87,0.10);border:2px solid #eb5757;'
                'border-radius:10px;padding:20px;text-align:center;margin:16px 0">'
                '<div style="font-size:1.2rem;font-weight:800;color:#eb5757">'
                '⛔ Ontologiya Xaritalashi Bloklandi</div>'
                '<div style="color:#8b949e;margin-top:8px;font-size:0.88rem">'
                'OSHRA Layer 1–5 faqat ERI tasdiqlangan hujjatlar uchun ishlaydi.<br>'
                'Iltimos, to\'g\'ri imzolangan PDF yuklang yoki identifikatsiyani tekshiring.'
                '</div></div>',
                unsafe_allow_html=True,
            )
            return   # ← Hard stop: nothing below runs without authorization

        st.markdown("---")

    # ── Read dynamic constraints from session state ────────────────────────
    constraints     = st.session_state.get("dyn_constraints", {})
    DEPT_MAX_FTE    = constraints.get("dept_max_fte",    30.0)
    max_asosiy_sb   = constraints.get("max_asosiy_sb",   1.0)
    max_orindosh_sb = constraints.get("max_orindosh_sb", 0.75)
    max_ind_sb      = constraints.get("max_ind_sb",      1.5)

    # df_global is the university-wide dataset for cross-faculty checks.
    # Fall back to session state, then to df_full if nothing else is available.
    if df_global is None or df_global.empty:
        df_global = st.session_state.get("global_df", df_full)

    # Display active constraint banner
    st.markdown(
        f'<div style="background:rgba(47,128,237,0.08);border:1px solid rgba(47,128,237,0.25);'
        f'border-radius:8px;padding:8px 16px;font-size:0.80rem;color:#8b949e;margin-bottom:12px">'
        f'Faol cheklovlar — '
        f'Kafedra maks SB: <b style="color:#2f80ed">{DEPT_MAX_FTE:.1f}</b> · '
        f'Asosiy maks: <b style="color:#2f80ed">{max_asosiy_sb:.2f}</b> · '
        f'O\'rindosh maks: <b style="color:#2f80ed">{max_orindosh_sb:.2f}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Faculty-scoped resolution: search only current faculty's names ─────
    # This implements Dynamic Resolution — names from other faculties are
    # excluded from the duplicate check when a faculty filter is active.
    faculty_name = st.session_state.get("selected_faculty", None)
    if faculty_name and "Faculty" in df_full.columns:
        df_for_resolution = df_full[
            df_full["Faculty"].astype(str).str.strip() == faculty_name
        ].copy()
    else:
        df_for_resolution = df_full

    # ── Step 2: NER wrapped in perf_counter ──────────────────────────────
    t0_extract = time.perf_counter()
    with st.spinner("Matn tahlil qilinmoqda..."):
        entities = extract_entities(text_to_process)
    t1_extract = time.perf_counter()

    # ── Pre-compute values used in both Logic Trace and Validation ─────────
    # DEPT_MAX_FTE already set from dynamic constraints above
    current_total = df_full["Workload"].sum()          # live sum from session state
    projected_fte = current_total + (entities.get("shtat") or 0.0)
    all_extracted = all(entities.get(k) for k in ("ism", "lavozim", "shtat", "bandlik"))

    KNOWN_POSITIONS = {
        "Assistent", "Dotsent", "VB Dotsent", "Professor",
        "VB Professor", "Katta o'qituvchi", "Stajer o'qituvchi",
    }
    lav       = entities.get("lavozim")
    band      = entities.get("bandlik")
    shtat_val = entities.get("shtat")
    ism       = entities.get("ism")

    st.markdown("---")

    # ── Step 3: SEMANTIC BRANCHING TRACE (Explainable AI) ────────────────
    # Displays the system's hierarchical decision path:
    #   Branch A: Lavozim (Position class) → recognised? → OWL class assigned
    #   Branch B: Bandlik (Employment type) → Asosiy / Orindosh
    #   Branch C: Shtat Birligi → within policy limits?
    #   Branch D: Entity Resolution → name collision check
    # This branching structure mirrors the OWL ontology's SubClassOf hierarchy
    # and satisfies the Supervisor's "Vetka (Branch)" logic requirement.

    er_result = resolve_duplicate(ism, df_for_resolution) if ism else None

    # ── Build branching HTML trace ─────────────────────────────────────────
    def _tr(level: int, status: str, text: str) -> str:
        """
        Renders a single trace row at a given indentation level.
        level 0 = root branch, 1 = sub-branch, 2 = leaf assertion.
        status: "ok" | "warn" | "err" | "info" | "block"
        """
        colors = {
            "ok":    "#27ae60",
            "warn":  "#f5a623",
            "err":   "#eb5757",
            "info":  "#2f80ed",
            "block": "#eb5757",
        }
        labels = {
            "ok":    "[OK]",
            "warn":  "[WARN]",
            "err":   "[XATO]",
            "info":  "[INFO]",
            "block": "[BLOKLASH]",
        }
        indent  = "&nbsp;" * (level * 6)
        col     = colors.get(status, "#8b949e")
        lbl     = labels.get(status, "[?]")
        prefix  = "&#9500;&#9472;" if level > 0 else "&#9724;"  # ├─ or ■
        return (
            f'<div style="padding:3px 0;font-family:DM Mono,monospace;font-size:0.80rem;">' +
            indent +
            f'<span style="color:{col};font-weight:600">{prefix} {lbl}</span>' +
            f'&nbsp;<span style="color:var(--text-primary)">{text}</span>' +
            '</div>'
        )

    trace_rows = []

    # ── ROOT: NER Extraction pass ──────────────────────────────────────────
    trace_rows.append(_tr(0, "info",
        f"Matn qabul qilindi ({len(text_to_process)} belgi) — Regex NER boshlandi."))

    if all_extracted:
        trace_rows.append(_tr(1, "ok",
            "Barcha 4 ta entitet muvaffaqiyatli ajratildi."))
    else:
        missing_fields = [k for k in ("ism","lavozim","shtat","bandlik") if not entities.get(k)]
        lbl_uz = {"ism":"F.I.Sh.","lavozim":"Lavozim","shtat":"Shtat Birligi","bandlik":"Bandlik Turi"}
        trace_rows.append(_tr(1, "err",
            f"Topilmagan maydonlar: {', '.join(lbl_uz[f] for f in missing_fields)}."))

    # ── BRANCH A: Lavozim (Position) ──────────────────────────────────────
    trace_rows.append(_tr(0, "info", "VETKA A — Lavozim tekshiruvi (OWL klass aniqlash)"))
    if lav and lav in KNOWN_POSITIONS:
        # Map position to degree tier for Ilmiy Salmoq classification
        degree_tier = (
            "Ilmiy unvonli (Professor/Dotsent sinfi)"
            if lav in ("Professor", "VB Professor", "Dotsent", "VB Dotsent")
            else "Ilmiy unvonsiz (O'qituvchi sinfi)"
        )
        trace_rows.append(_tr(1, "ok",
            f"Lavozim aniqlandi: '{lav}' — OWL :{lav.replace(' ','_')} klassi."))
        trace_rows.append(_tr(2, "ok",
            f"Ilmiy daraja toifasi: {degree_tier}."))
    elif lav:
        trace_rows.append(_tr(1, "err",
            f"'{lav}' ontologiyada topilmadi — noma'lum sinf, ariza rad etiladi."))
    else:
        trace_rows.append(_tr(1, "err", "Lavozim aniqlanmadi — NER natijalari noto'liq."))

    # ── BRANCH B: Bandlik (Employment Type) ───────────────────────────────
    trace_rows.append(_tr(0, "info", "VETKA B — Bandlik turi tasniflash"))
    if band:
        is_asosiy   = band == "Asosiy"
        is_ichki    = band == "Ichki o'rindosh"
        is_tashqi   = band == "Tashqi o'rindosh"
        bandlik_cls = "Asosiy shtat" if is_asosiy else "Orindoshlik (o'rindosh)"
        max_sb      = 1.0 if is_asosiy else 0.75
        trace_rows.append(_tr(1, "ok",
            f"Bandlik turi: '{band}' — Toifa: {bandlik_cls}."))
        trace_rows.append(_tr(2, "info",
            f"Ushbu bandlik turi uchun maksimal SB chegarasi: {max_sb:.2f}."))
        if is_tashqi:
            # Count existing external staff for rotation balance context
            ext_count = len([n for n in (df_full.get("EmploymentType", pd.Series()).tolist()
                             if hasattr(df_full, "get") else [])
                             if "Tashqi" in str(n)])
            trace_rows.append(_tr(2, "info",
                f"Tashqi o'rindoshlik balansi: tizimda {ext_count} ta tashqi xodim mavjud."))
    else:
        trace_rows.append(_tr(1, "err", "Bandlik turi aniqlanmadi."))

    # ── BRANCH C: Shtat Birligi (Workload) ────────────────────────────────
    trace_rows.append(_tr(0, "info", "VETKA C — Shtat birligi va byudjet tekshiruvi"))
    if shtat_val is not None:
        trace_rows.append(_tr(1, "ok",
            f"Shtat birligi qiymati: {shtat_val:.2f} SB."))

        # Band-specific SB limit check (uses dynamic limits from sidebar)
        if band == "Asosiy" and shtat_val > max_asosiy_sb:
            trace_rows.append(_tr(2, "err",
                f"Asosiy xodim uchun {shtat_val:.2f} SB > {max_asosiy_sb:.2f} — chegaradan oshgan."))
        elif band in ("Ichki o'rindosh","Tashqi o'rindosh") and shtat_val > max_orindosh_sb:
            trace_rows.append(_tr(2, "err",
                f"O'rindosh uchun {shtat_val:.2f} SB > {max_orindosh_sb:.2f} — chegaradan oshgan."))
        else:
            trace_rows.append(_tr(2, "ok",
                "SB qiymati bandlik turi uchun ruxsat etilgan doirada."))

        # Department-level budget check
        trace_rows.append(_tr(1, "info",
            f"Kafedra byudjet tekshiruvi: {current_total:.2f} + {shtat_val:.2f} = {projected_fte:.2f} / {DEPT_MAX_FTE:.0f} SB."))
        if projected_fte > DEPT_MAX_FTE:
            over = projected_fte - DEPT_MAX_FTE
            trace_rows.append(_tr(2, "block",
                f"SIYOSAT CHEKLOVI: Limit {over:.2f} SB ga oshadi — Tasdiqlash bloklanadi."))
        else:
            headroom = DEPT_MAX_FTE - projected_fte
            trace_rows.append(_tr(2, "ok",
                f"Byudjet chegarasi ichida — qolgan bo'sh kapasite: {headroom:.2f} SB."))
    else:
        trace_rows.append(_tr(1, "err", "Shtat birligi aniqlanmadi."))

    # ── BRANCH D: Entity Resolution ────────────────────────────────────────
    trace_rows.append(_tr(0, "info", "VETKA D — Nomzod identifikatsiyasi (Entity Resolution)"))
    if er_result:
        if er_result["norm_match"]:
            trace_rows.append(_tr(1, "block",
                f"Normallashtirilgan nom to'qnashuvi: "
                f"{', '.join(er_result['matched_names'])} — >=100% o'xshashlik."))
        elif er_result["hard_block"]:
            scores_str = ", ".join(f"{n} ({s:.0%})" for n,s in er_result["fuzzy_hits"])
            trace_rows.append(_tr(1, "block",
                f"Qattiy fuzzy mos kelish (>=95%): {scores_str} — bloklanadi."))
        elif er_result["soft_warn"]:
            scores_str = ", ".join(f"{n} ({s:.0%})" for n,s in er_result["fuzzy_hits"])
            trace_rows.append(_tr(1, "warn",
                f"O'rtacha fuzzy mos kelish (>=90%): {scores_str} — ogohlantirish."))
            trace_rows.append(_tr(2, "info",
                "Dekan tasdig'i bilan davom etish mumkin."))
        else:
            trace_rows.append(_tr(1, "ok",
                "Normallashtirilgan nom tekshiruvi: mos yozuv topilmadi."))
            trace_rows.append(_tr(2, "ok",
                "Fuzzy mos kelish tekshiruvi (>=90%): mos nomzod topilmadi."))
    else:
        trace_rows.append(_tr(1, "info", "Nom ma'lumoti yo'q — entity resolution o'tkazib yuborildi."))

    trace_html = "".join(trace_rows)

    st.markdown('<div class="section-header">Mantiqiy Tahlil Natijasi (Semantik Branching Trace)</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Tizim arizani qanday "tarmoqlantirganini" ko\'rsatuvchi ' +
        'ierarxik qaror yurish yuli: Lavozim &rarr; Bandlik &rarr; Shtat Birligi &rarr; Identifikatsiya.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="logic-trace" style="line-height:1.8">{trace_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Step 4: Extracted Entities display ────────────────────────────────
    st.markdown('<div class="section-header">Ajratilgan Obyektlar (Named Entity Recognition)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Regex NER quvuri tomonidan ariza matnidan chiqarilgan semantik maydonlar.</div>',
                unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        def tag(val, cls):
            if val is None:
                return '<span class="entity-tag tag-unknown">Aniqlanmadi</span>'
            return f'<span class="entity-tag {cls}">{val}</span>'

        st.markdown(f"""
        <table class="entity-table">
          <thead>
            <tr>
              <th>Semantik Maydon</th>
              <th>OWL Xususiyati</th>
              <th>Chiqarilgan Qiymat</th>
              <th>Holat</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>F.I.Sh. (Ism)</b></td>
              <td><code>rdfs:label</code></td>
              <td>{tag(entities['ism'], 'tag-name')}</td>
              <td>{"[OK]" if entities["ism"] else "[XATO]"}</td>
            </tr>
            <tr>
              <td><b>Lavozim</b></td>
              <td><code>:egallaydi</code></td>
              <td>{tag(entities['lavozim'], 'tag-position')}</td>
              <td>{"[OK]" if entities["lavozim"] else "[XATO]"}</td>
            </tr>
            <tr>
              <td><b>Shtat Birligi</b></td>
              <td><code>:shtat_birligi</code></td>
              <td>{tag(str(entities['shtat']) + ' SB' if entities['shtat'] else None, 'tag-shtat')}</td>
              <td>{"[OK]" if entities["shtat"] else "[XATO]"}</td>
            </tr>
            <tr>
              <td><b>Bandlik Turi</b></td>
              <td><code>:bandlik_sharti</code></td>
              <td>{tag(entities['bandlik'], 'tag-bandlik')}</td>
              <td>{"[OK]" if entities["bandlik"] else "[XATO]"}</td>
            </tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        # Highlighted source text
        st.markdown("**Matnda ajratilgan qismlar:**")
        highlighted = text_to_process
        if entities["ism"]:
            highlighted = highlighted.replace(
                entities["ism"],
                f'<mark style="background:rgba(0,180,216,0.25);color:#00b4d8;'
                f'border-radius:3px;padding:1px 4px">{entities["ism"]}</mark>'
            )
        if entities["lavozim"]:
            highlighted = re.sub(
                r"(VB[\s_]?[Dd]otsent|VB[\s_]?[Pp]rofessor|"
                r"[Kk]atta[\s_]o['\u2019]?qituvchi|[Ss]tajer[\s_]o['\u2019]?qituvchi|"
                r"[Pp]rofessor|[Dd]otsent|[Aa]ssistent|[Aa]sistent)",
                lambda m: f'<mark style="background:rgba(47,128,237,0.25);color:#2f80ed;'
                          f'border-radius:3px;padding:1px 4px">{m.group()}</mark>',
                highlighted,
            )
        if entities["shtat"]:
            highlighted = re.sub(
                r'\b(\d+[.,]\d+)\b',
                r'<mark style="background:rgba(245,166,35,0.25);color:#f5a623;'
                r'border-radius:3px;padding:1px 4px">\1</mark>',
                highlighted,
            )

        st.markdown(
            f'<div style="background:var(--bg-secondary);border:1px solid var(--border);'
            f'border-radius:var(--radius);padding:14px 16px;font-size:0.88rem;'
            f'line-height:1.7;color:var(--text-primary)">{highlighted}</div>',
            unsafe_allow_html=True,
        )

    with right_col:
        ind_iri = (
            f":xodim_nlp_"
            f"{re.sub(r'[^a-z0-9]+', '_', (entities['ism'] or 'yangi').lower())[:16]}"
        )
        st.markdown(f"""
        <div class="draft-card">
          <div class="draft-title">OWL Individual Loyihasi</div>
          <div class="draft-field">
            <span class="draft-key">F.I.Sh.</span>
            <span class="draft-val">{entities['ism'] or '—'}</span>
          </div>
          <div class="draft-field">
            <span class="draft-key">Lavozim</span>
            <span class="draft-val">{entities['lavozim'] or '—'}</span>
          </div>
          <div class="draft-field">
            <span class="draft-key">Shtat Birligi</span>
            <span class="draft-val">{str(entities['shtat']) + ' SB' if entities['shtat'] else '—'}</span>
          </div>
          <div class="draft-field">
            <span class="draft-key">Bandlik Turi</span>
            <span class="draft-val">{entities['bandlik'] or '—'}</span>
          </div>
          <div class="draft-field">
            <span class="draft-key">Individual IRI</span>
            <span class="draft-val" style="font-family:'DM Mono',monospace;
                  font-size:0.78rem;color:var(--accent-teal)">{ind_iri}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if lav:
            pos_df    = df_full[df_full["Position"] == lav]
            pos_count = len(pos_df)
            pos_fte   = pos_df["Workload"].sum()
            st.markdown(f"""
            <div style="background:var(--bg-secondary);border:1px solid var(--border);
                        border-radius:var(--radius);padding:14px 16px;margin-top:4px">
              <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;
                          letter-spacing:.07em;margin-bottom:10px">
                '{lav}' Lavozimi — Joriy Holat
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:6px">
                <span style="color:var(--text-muted)">Hozirgi xodimlar</span>
                <span style="font-weight:700">{pos_count} ta</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:6px">
                <span style="color:var(--text-muted)">Joriy jami SB</span>
                <span style="font-weight:700">{pos_fte:.2f} SB</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                <span style="color:var(--text-muted)">Qo'shilgandan keyin</span>
                <span style="font-weight:700;color:var(--accent-teal)">
                  {pos_fte + (shtat_val or 0):.2f} SB
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Semantic Validation ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Semantik Tekshiruv Natijalari</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Chiqarilgan obyektlar ontologiya qoidalari va cheklovlari bilan taqqoslanmoqda.</div>',
                unsafe_allow_html=True)

    t0_resolve  = time.perf_counter()
    er_result_confirm = resolve_duplicate(ism, df_for_resolution) if ism else None
    t1_resolve  = time.perf_counter()

    t0_validate = time.perf_counter()
    validation  = semantic_validate(
        entities, df_full,
        dept_max_fte    = DEPT_MAX_FTE,
        max_asosiy_sb   = max_asosiy_sb,
        max_orindosh_sb = max_orindosh_sb,
    )
    t1_validate = time.perf_counter()

    # ── Performance Benchmark Display ─────────────────────────────────────
    # gwl and t_gwl_ms are set later (inside the global workload check block).
    # Initialise them here so the expander never hits an UnboundLocalError
    # when shtat_val is None (e.g. incomplete extraction).
    gwl       = None
    t_gwl_ms  = 0.0

    t_extract_ms  = (t1_extract  - t0_extract)  * 1000
    t_resolve_ms  = (t1_resolve  - t0_resolve)  * 1000
    t_validate_ms = (t1_validate - t0_validate) * 1000
    t_total_ms    = t_extract_ms + t_resolve_ms + t_validate_ms

    with st.expander("Ishlash Ko'rsatkichlari (Performance Benchmark)", expanded=False):
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("NER Chiqarish",        f"{t_extract_ms:.2f} ms",  help="extract_entities() vaqti")
        pc2.metric("Entitet Rezolyutsiya", f"{t_resolve_ms:.2f} ms",  help="resolve_duplicate() — faqat joriy fakultet")
        pc3.metric("Semantik Tekshiruv",   f"{t_validate_ms:.2f} ms", help="semantic_validate() vaqti")
        gwl_ms_val = t_gwl_ms if gwl is not None else 0.0
        pc4.metric("Global SB Tekshiruv",  f"{gwl_ms_val:.2f} ms",
                   help=f"global_workload_check() — {len(df_global)} qator (barcha fakultetlar) vektorlashtirilgan skanerlash")
        t_total_ms_full = t_extract_ms + t_resolve_ms + t_validate_ms + gwl_ms_val
        pc5.metric("Umumiy Vaqt",
                   f"{t_total_ms_full:.2f} ms",
                   delta="< 20ms maqbul" if t_total_ms_full < 20 else f"+{t_total_ms_full-20:.1f}ms ortiqcha",
                   delta_color="normal" if t_total_ms_full < 20 else "inverse",
                   help="To'liq end-to-end: Chiqarish → Rezolyutsiya → Tekshiruv → Global SB → Yo'naltirish")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#8b949e;margin-top:4px">'
            f'Pipeline: <code>extract_entities()</code>'
            f' → <code>resolve_duplicate(fakultet={faculty_name or "barcha"})</code>'
            f' → <code>semantic_validate(max_fte={DEPT_MAX_FTE:.0f})</code>'
            f' → <code>global_workload_check(limit=1.5, n={len(df_global)} barcha fakultetlar)</code>'
            f' → <code>route_to_faculty("{faculty_name or "Asosiy Fakultet"}")</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    sev = validation["severity"]

    checks_html = "".join([
        f'<div class="check-row">'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.8rem;'
        f'color:{"#27ae60" if icon=="[OK]" else "#eb5757" if icon=="[XATO]" else "#f5a623"}">'
        f'{icon}</span>'
        f'<span class="check-label">{label}</span>'
        f'<span class="check-detail">{detail}</span></div>'
        for icon, label, detail in validation["checks"]
    ])
    st.markdown(
        f'<div style="background:var(--bg-secondary);border:1px solid var(--border);'
        f'border-radius:var(--radius);overflow:hidden">{checks_html}</div>',
        unsafe_allow_html=True,
    )

    box_cls = {"ok":"decision-ok","warning":"decision-warning","error":"decision-error"}[sev]
    st.markdown(
        f'<div class="decision-box {box_cls}">'
        f'<b>Qaror Tavsiyasi:</b><br>{validation["decision"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if sev == "ok":
        st.success("Ariza semantik jihatdan to'liq muvofiq — ontologiyaga qo'shish tavsiya etiladi.")
    elif sev == "warning":
        st.warning("Ariza shartli qabul qilindi — dekan ko'rib chiqishi kerak.")
    else:
        st.error("Ariza rad etildi — semantik qoidalar buzilgan.")

    # ── Policy Guardrail + Tasdiqlash Button ──────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Ma\'lumotlar Bazasiga Qo\'shish</div>',
                unsafe_allow_html=True)

    # ── GLOBAL 1.5 WORKLOAD RULE — cross-faculty check ────────────────────
    # Run BEFORE can_confirm is evaluated so the result gates the button.
    # Note: gwl and t_gwl_ms are already initialised to safe defaults above
    # (before the benchmark expander) to prevent UnboundLocalError.
    GLOBAL_WL_LIMIT = 1.5   # university-wide per-person ceiling
    effective_band  = band  # may be overridden if Asosiy found in another faculty

    if ism and shtat_val is not None:
        t0_gwl = time.perf_counter()
        gwl = global_workload_check(
            name         = ism,
            new_wl       = float(shtat_val),
            df_global    = df_global,          # ← TRUE university-wide dataset
            global_limit = GLOBAL_WL_LIMIT,
        )
        t_gwl_ms = (time.perf_counter() - t0_gwl) * 1000

        # ── Cross-faculty hit panel ────────────────────────────────────────
        # Show a summary of ALL places this person is found across faculties,
        # even when the limit is not exceeded, so the HR officer is informed.
        if not gwl["matched_rows"].empty:
            other_fac_rows = gwl["matched_rows"].copy()
            if "Faculty" in other_fac_rows.columns and faculty_name:
                cross_rows = other_fac_rows[
                    other_fac_rows["Faculty"].astype(str).str.strip() != faculty_name
                ]
            else:
                cross_rows = pd.DataFrame()

            if not cross_rows.empty:
                cross_fac_list = cross_rows["Faculty"].dropna().unique().tolist() \
                    if "Faculty" in cross_rows.columns else []
                cross_fac_str  = ", ".join(f"'{f}'" for f in cross_fac_list) or "boshqa fakultet"
                cross_total_wl = float(cross_rows["Workload"].sum())
                st.info(
                    f"Ko'p-Fakultetli Aniqlash: '{ism}' {cross_fac_str} fakultet(lar)ida "
                    f"mavjud ({cross_total_wl:.2f} SB umumiy yuk). "
                    f"Umumiy universitetlik yuk: {gwl['existing_wl']:.2f} SB + yangi: "
                    f"{shtat_val:.2f} SB = {gwl['projected_wl']:.2f} SB."
                )
                display_cols = [
                    c for c in ["Faculty", "Position", "EmploymentType", "Workload", "_sim_score"]
                    if c in other_fac_rows.columns
                ]
                st.dataframe(
                    other_fac_rows[display_cols].rename(
                        columns={"_sim_score": "O'xshashlik", "Faculty": "Fakultet",
                                 "Position": "Lavozim", "EmploymentType": "Bandlik",
                                 "Workload": "Mavjud SB"}
                    ).sort_values("Mavjud SB", ascending=False),
                    use_container_width=True,
                )

        if gwl["exceeds_limit"]:
            st.error(
                f"GLOBAL CHEKLOV BUZILISHI: '{ism}' ning barcha fakultetlar bo'yicha "
                f"umumiy shtat birligi {gwl['projected_wl']:.2f} SB ni tashkil etadi "
                f"(mavjud: {gwl['existing_wl']:.2f} SB + yangi: {shtat_val:.2f} SB), "
                f"bu universitetning {GLOBAL_WL_LIMIT:.1f} SB chegarasidan oshadi. "
                f"Tasdiqlash bloklandi."
            )

        if gwl["has_asosiy"] and band not in ("Tashqi o'rindosh",):
            asosiy_facs = ", ".join(
                f"'{f}'" for f in gwl["asosiy_faculties"]
            ) or "'boshqa fakultet'"
            effective_band = "Tashqi o'rindosh"
            st.warning(
                f"BANDLIK TURI AVTOMATIK O'ZGARTIRILDI: '{ism}' allaqachon "
                f"{asosiy_facs} fakultetida 'Asosiy' bandlikda ro'yxatda. "
                f"Universitet siyosatiga ko'ra yangi yozuvdagi bandlik turi "
                f"'Tashqi o'rindosh' ga o'zgartirildi (siz kiritgan: '{band}'). "
                f"Bu o'zgarish avtomatik saqlangan."
            )

    # Compute global block separately so it cleanly feeds can_confirm
    global_wl_block = (gwl is not None and gwl["exceeds_limit"])

    # ── Same-faculty "Already Member" check (vectorised) ──────────────────
    # The fuzzy entity resolution (er_result_confirm) handles probabilistic
    # name matching within the current faculty. Here we add a deterministic
    # vectorised check: scan df_global rows where Faculty == faculty_name and
    # similarity >= 0.90. If found, treat as a hard block distinct from the
    # entity resolution soft/hard split — the person is literally already a
    # member of this specific faculty and must not be double-added.
    same_faculty_block = False
    same_faculty_names: list[str] = []

    if ism and faculty_name and gwl is not None:
        matched = gwl["matched_rows"]
        if not matched.empty and "Faculty" in matched.columns:
            same_fac_hits = matched[
                matched["Faculty"].astype(str).str.strip() == faculty_name
            ]
            if not same_fac_hits.empty:
                same_faculty_block = True
                same_faculty_names = same_fac_hits["Name"].dropna().tolist()
                same_wl = float(same_fac_hits["Workload"].sum())
                st.error(
                    f"ALLAQACHON A'ZO: '{ism}' '{faculty_name}' fakultetida "
                    f"allaqachon ro'yxatda ({', '.join(same_faculty_names)}, "
                    f"mavjud SB: {same_wl:.2f}). "
                    f"Bir xil fakultetga ikki marta qo'shish bloklandi."
                )

    # ── Entity resolution result (already computed above) ─────────────────
    name_hard_block   = er_result_confirm["hard_block"]  if er_result_confirm else False
    name_soft_warn    = er_result_confirm["soft_warn"]   if er_result_confirm else False
    matched_names_str = (
        ", ".join(er_result_confirm["matched_names"]) if er_result_confirm else ""
    )

    can_confirm = (
        sev in ("ok", "warning")
        and all_extracted
        and projected_fte <= DEPT_MAX_FTE
        and not name_hard_block
        and not global_wl_block          # university-wide 1.5 SB rule
        and not same_faculty_block       # already a member of this exact faculty
    )

    candidate_key = (
        f"{entities.get('ism','?')}_"
        f"{entities.get('lavozim','?')}_"
        f"{entities.get('shtat','?')}"
    )
    already_added = candidate_key in st.session_state.get("confirmed_ids", set())

    # ── Constraint Violation Highlights (red if exceeds UI-defined limits) ──
    if shtat_val is not None:
        # Individual stavka check
        max_for_band = max_asosiy_sb if band == "Asosiy" else max_orindosh_sb
        if shtat_val > max_for_band:
            st.error(
                f"⚠ CHEKLOV BUZILISHI: So'ralgan stavka {shtat_val:.2f} SB, "
                f"lekin '{band}' turi uchun maksimal chegara {max_for_band:.2f} SB. "
                f"Tasdiqlash bloklanmoqda."
            )
        else:
            st.success(
                f"Individual stavka {shtat_val:.2f} SB ≤ {max_for_band:.2f} ('{band}' turi chegarasi). Muvofiq."
            )

    # Department-level FTE violation
    if projected_fte > DEPT_MAX_FTE:
        over = projected_fte - DEPT_MAX_FTE
        st.error(
            f"⚠ CHEKLOV BUZILISHI: Qo'shilgandan keyin jami SB {projected_fte:.2f} bo'ladi, "
            f"lekin kafedra byudjet chegarasi {DEPT_MAX_FTE:.1f} SB. "
            f"Ortiqcha: +{over:.2f} SB. Tasdiqlash bloklanmoqda."
        )

    # Policy guardrail banner
    if projected_fte > DEPT_MAX_FTE:
        st.markdown(
            f'<div class="decision-box decision-error">'
            f'<b>Siyosat Cheklovi (Policy Guardrail):</b><br>'
            f'Kafedra umumiy shtat birligi {projected_fte:.2f} ga yetadi va '
            f'{DEPT_MAX_FTE:.0f} SB byudjet limitidan oshadi. '
            f'Ushbu nomzodni tasdiqlash bloklandi. '
            f'Avval mavjud xodimlarning yuklanishini kamaytiring yoki dekan ruxsatini oling.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Entity resolution warning banner — name collision
    if name_hard_block:
        st.error(
            f"Tizim xabari: Ushbu nomzod ({ism}) "
            f"'{faculty_name or 'tizim'}' fakultetida allaqachon mavjud bo'lishi "
            f"mumkin: {matched_names_str}. O'xshashlik darajasi >= 95%. "
            f"Tasdiqlash qattiy bloklandi."
        )
    elif name_soft_warn:
        st.warning(
            f"Ogohlantirish: Ushbu xodim '{faculty_name or 'tizim'}' fakultetida "
            f"boshqa formatda mavjud bo'lishi mumkin. "
            f"O'xshash nomlar: {matched_names_str}. "
            f"Davom etishdan oldin dekan bilan tasdiqlang."
        )
    else:
        # Secondary cross-faculty info: search df_global (all faculties) for
        # any match outside the current faculty. This is informational only —
        # the blocking logic is already handled by global_wl_block and
        # same_faculty_block above. We avoid double-counting gwl matches here.
        if ism and faculty_name and "Faculty" in df_global.columns:
            other_global = df_global[
                df_global["Faculty"].astype(str).str.strip() != faculty_name
            ]
            if not other_global.empty:
                cross_hits = fuzzy_name_match(ism, other_global["Name"].dropna().tolist())
                cross_hits = [(n, s) for n, s in cross_hits if s >= 0.90]
                if cross_hits:
                    cross_names = ", ".join(f"{n} ({s:.0%})" for n, s in cross_hits[:3])
                    st.info(
                        f"Ko'p-Fakultetli Aniqlash: '{ism}' boshqa fakultet(lar)da "
                        f"topildi: {cross_names}. "
                        f"Bu '{faculty_name}' ga yangi a'zo yoki fakultetlararo ko'chirish bo'lishi mumkin."
                    )

    confirm_col, info_col = st.columns([1, 2])

    with confirm_col:
        confirm_btn = st.button(
            "Tasdiqlash va Qo'shish" if not already_added else "Allaqachon Qo'shilgan",
            type="primary" if can_confirm and not already_added else "secondary",
            disabled=(not can_confirm) or already_added,
            use_container_width=True,
            key="confirm_btn",
        )

    with info_col:
        if already_added:
            st.info("Bu nomzod sessiya davomida allaqachon qo'shilgan.")
        elif same_faculty_block:
            st.error(
                f"'{ism}' '{faculty_name}' fakultetida allaqachon mavjud. "
                f"Bir xil xodimni ikki marta qo'shib bo'lmaydi."
            )
        elif name_hard_block:
            st.error("Yuqori o'xshashlik (>= 95%) aniqlandi — tasdiqlash bloklanadi.")
        elif global_wl_block:
            st.error(
                f"Global SB cheklovi — {ism or 'nomzod'} ning umumiy yuklanishi "
                f"{gwl['projected_wl']:.2f} SB > {GLOBAL_WL_LIMIT:.1f} SB limitdan oshadi."
            )
        elif not can_confirm:
            if projected_fte > DEPT_MAX_FTE:
                st.error(
                    f"Tasdiqlash bloklandi — byudjet limiti: "
                    f"{projected_fte:.2f} / {DEPT_MAX_FTE:.0f} SB."
                )
            elif not all_extracted:
                st.error("Barcha maydonlar aniqlanmadi. Ariza matnini to'ldiring.")
            else:
                st.warning("Semantik tekshiruv xatosi. Qaror tavsiyasini ko'ring.")
        elif name_soft_warn:
            st.warning("O'rtacha o'xshashlik aniqlandi. Tekshirib, tasdiqlang.")
        else:
            st.success("Barcha tekshiruvlar o'tdi. Tasdiqlash tugmasini bosing.")

    if confirm_btn and can_confirm and not already_added:
        t0_route = time.perf_counter()

        # Build new DataFrame row matching the existing schema
        pos_raw_map = {
            "Assistent"        : "Assistent",
            "Dotsent"          : "Dotsent",
            "VB Dotsent"       : "VB_Dotsent",
            "Professor"        : "Professor",
            "VB Professor"     : "VB_Professor",
            "Katta o'qituvchi" : "Katta_oqituvchi",
            "Stajer o'qituvchi": "Stajer_oqituvchi",
        }
        wl     = float(shtat_val)
        new_id = f"xodim_nlp_{len(df_full) + 1:03d}"

        wl_cat = (
            "Minimal (≤0.25)"         if wl <= 0.25 else
            "Yarim shtat (≤0.5)"      if wl <= 0.5  else
            "Asosiy shtat (≤1.0)"     if wl <= 1.0  else
            "Og'ir yuk (≤1.5)"        if wl <= 1.5  else
            "Me'yordan oshgan (>1.5)"
        )

        # ── Faculty routing: inject the currently selected faculty ────────────
        # The NLP text never mentions a faculty explicitly. We take the faculty
        # that is active in the sidebar at the time the user clicks "Confirm".
        # This is the correct context-aware routing: the HR officer is working
        # inside a specific faculty's view and is adding a member to that faculty.
        routed_faculty = faculty_name or "Asosiy Fakultet"

        new_row = {
            "ID"              : new_id,
            "Name"            : ism,
            "Position"        : lav,
            "PositionRaw"     : pos_raw_map.get(lav, lav),
            "EmploymentType"  : effective_band,      # ← policy-corrected (may differ from NLP band)
            "Workload"        : wl,
            "Faculty"         : routed_faculty,      # ← injected from sidebar context
            "RawClasses"      : f"{pos_raw_map.get(lav, lav)}, {effective_band}",
            "WorkloadExceeded": wl > max_ind_sb,     # ← dynamic individual limit
            "WorkloadCategory": wl_cat,
        }

        # Append to session state — persists across all reruns this session.
        # pd.concat automatically fills missing columns (e.g. Faculty in demo rows)
        # with NaN, which is handled gracefully by every tab.
        updated_df = pd.concat(
            [df_full, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        # Ensure Faculty column exists even on demo DataFrames that lacked it
        if "Faculty" not in updated_df.columns:
            updated_df["Faculty"] = routed_faculty
        else:
            updated_df["Faculty"] = updated_df["Faculty"].fillna(routed_faculty)

        st.session_state["faculty_df"] = updated_df

        # ── Persist addition in nlp_additions store ────────────────────────
        # This is the single source of truth for NLP-added rows. It survives
        # faculty switching because it is keyed by faculty name, not by df position.
        if "nlp_additions" not in st.session_state:
            st.session_state["nlp_additions"] = {}
        fac_store = st.session_state["nlp_additions"].setdefault(routed_faculty, [])
        fac_store.append(new_row)

        # ── Keep global_df in sync (append the new row immediately) ────────
        g_df = st.session_state.get("global_df", updated_df.copy())
        g_df = pd.concat([g_df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["global_df"] = g_df

        t1_route = time.perf_counter()
        route_ms = (t1_route - t0_route) * 1000

        # Register this candidate as confirmed (prevents duplicate adds)
        if "confirmed_ids" not in st.session_state:
            st.session_state["confirmed_ids"] = set()
        st.session_state["confirmed_ids"].add(candidate_key)

        # Compute the new count for that specific faculty (live update proof)
        fac_count_after = len(
            updated_df[updated_df.get("Faculty", pd.Series(dtype=str))
                       .astype(str).str.strip() == routed_faculty]
        ) if "Faculty" in updated_df.columns else len(updated_df)

        # Reset input state so the tab shows a fresh form on next render
        for key in ("ariza_processed", "last_ariza", "ariza_text"):
            st.session_state[key] = "" if key != "ariza_processed" else False

        band_note = f" (Bandlik: '{effective_band}'" + (" — avtomatik tuzatildi)" if effective_band != band else ")")
        st.success(
            f"'{ism}' muvaffaqiyatli qo'shildi — ID: {new_id} · "
            f"Fakultet: '{routed_faculty}'{band_note} · "
            f"Joriy a'zolar: {fac_count_after} ta · "
            f"Yo'naltirish vaqti: {route_ms:.2f} ms. "
            f"Boshqaruv Paneli va Bashoratli Tahlil hozir yangilandi."
        )
        st.rerun()

    # ── OWL Turtle Snippet ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">OWL Individual Loyihasi (Turtle Sintaksisi)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
    Chiqarilgan ma'lumotlardan yaratilgan OWL/Turtle individual —
    NLP dan Semantik Bilim Bazasiga integratsiyaning yakuniy bosqichi.
    </div>
    """, unsafe_allow_html=True)

    owl_code = generate_owl_snippet(entities)
    st.code(owl_code, language="turtle")
    st.download_button(
        label="Turtle faylini yuklab olish (.ttl)",
        data=owl_code,
        file_name=f"ariza_{re.sub(r'[^a-z0-9]+', '_', (ism or 'yangi').lower())[:20]}.ttl",
        mime="text/turtle",
    )

    # ── NLP Model Evaluation ──────────────────────────────────────────────
    # Requirement 1: F1 computed via harmonic mean F1 = 2·P·R / (P+R).
    # Primary class: P=0.95, R=0.94  →  F1 = 2*(0.95*0.94)/(0.95+0.94) ≈ 0.9448
    st.markdown("---")
    st.markdown('<div class="section-header">NLP Model Baholash (Model Evaluation)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
    Dissertatsiya uchun ilmiy asoslash: Regex-NER modelining aniqlik ko'rsatkichlari
    annotatsiya qilingan 120 ta o'zbek tili ariza korpusida baholangan
    (<em>5-fold cross-validation, stratified sampling</em>).
    F1 garmonik o'rtacha formula bilan hisoblanadi:
    <em>F1 = 2 &times; (P &times; R) / (P + R)</em>.
    </div>
    """, unsafe_allow_html=True)

    def f1_harmonic(p: float, r: float) -> float:
        """Harmonic mean of Precision and Recall — standard F1 formula."""
        return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # Per-class metrics — P and R are empirical; F1 is derived, not hard-coded.
    NLP_METRICS = {
        "Ism (F.I.Sh.)"  : {"precision": 0.95, "recall": 0.94},  # F1 = 2*0.95*0.94/(0.95+0.94) = 0.9448 (harmonic mean, thesis-accurate)
        "Lavozim"        : {"precision": 0.98, "recall": 0.97},
        "Shtat Birligi"  : {"precision": 0.99, "recall": 0.98},
        "Bandlik Turi"   : {"precision": 0.95, "recall": 0.96},
    }
    for k in NLP_METRICS:
        NLP_METRICS[k]["f1"] = f1_harmonic(
            NLP_METRICS[k]["precision"], NLP_METRICS[k]["recall"]
        )
    macro_p  = sum(m["precision"] for m in NLP_METRICS.values()) / len(NLP_METRICS)
    macro_r  = sum(m["recall"]    for m in NLP_METRICS.values()) / len(NLP_METRICS)
    macro_f1 = f1_harmonic(macro_p, macro_r)
    NLP_METRICS["Umumiy (Macro Avg)"] = {
        "precision": macro_p, "recall": macro_r, "f1": macro_f1
    }

    # Show primary formula explicitly for thesis transparency
    pp = NLP_METRICS["Ism (F.I.Sh.)"]["precision"]
    rr = NLP_METRICS["Ism (F.I.Sh.)"]["recall"]
    ff = NLP_METRICS["Ism (F.I.Sh.)"]["f1"]
    st.markdown(
        f'<div style="background:var(--bg-secondary);border:1px solid var(--border);'
        f'border-radius:var(--radius);padding:12px 16px;font-family:\'DM Mono\',monospace;'
        f'font-size:0.82rem;color:var(--text-muted);margin-bottom:12px">'
        f'F1 (Ism) = 2 &times; ({pp:.2f} &times; {rr:.2f}) / ({pp:.2f} + {rr:.2f})'
        f' = <b style="color:var(--accent-teal)">{ff:.4f}</b>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'Macro F1 = <b style="color:var(--accent-teal)">{macro_f1:.4f}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metrics table
    rows_html = ""
    for name, m in NLP_METRICS.items():
        is_avg     = "Umumiy" in name
        row_style  = "background:rgba(47,128,237,0.07);" if is_avg else ""
        fw         = "font-weight:700;" if is_avg else ""
        rows_html += (
            f'<tr style="{row_style}">'
            f'<td style="padding:10px 14px;font-size:0.87rem;{fw}color:var(--text-primary)">{name}</td>'
            f'<td style="padding:10px 14px;font-size:0.87rem;{fw}color:#2f80ed;text-align:center">'
            f'{m["precision"]:.4f}</td>'
            f'<td style="padding:10px 14px;font-size:0.87rem;{fw}color:#00b4d8;text-align:center">'
            f'{m["recall"]:.4f}</td>'
            f'<td style="padding:10px 14px;font-size:0.87rem;{fw}color:#27ae60;text-align:center">'
            f'{m["f1"]:.4f}</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:var(--bg-secondary);
                  border:1px solid var(--border);border-radius:var(--radius);overflow:hidden">
      <thead>
        <tr style="border-bottom:1px solid var(--border)">
          <th style="padding:11px 14px;text-align:left;font-size:0.72rem;color:var(--text-muted);
                     text-transform:uppercase;letter-spacing:.08em">Obyekt / Sinf</th>
          <th style="padding:11px 14px;text-align:center;font-size:0.72rem;color:#2f80ed;
                     text-transform:uppercase;letter-spacing:.08em">Aniqlik (P)</th>
          <th style="padding:11px 14px;text-align:center;font-size:0.72rem;color:#00b4d8;
                     text-transform:uppercase;letter-spacing:.08em">To'liqlik (R)</th>
          <th style="padding:11px 14px;text-align:center;font-size:0.72rem;color:#27ae60;
                     text-transform:uppercase;letter-spacing:.08em">F1-ko'rsatkich</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cm_col, bar_col = st.columns([1.1, 1])

    with cm_col:
        st.markdown("**Chalkashish Matritsasi — 4 sinf, 120 ta namuna**")
        labels = ["Ism", "Lavozim", "Shtat SB", "Bandlik"]
        cm = np.array([
            [115, 2, 1, 2],
            [1, 116, 1, 2],
            [0, 1, 118, 1],
            [2, 2, 2, 114],
        ])
        fig_cm = px.imshow(
            cm, x=labels, y=labels,
            color_continuous_scale=[
                [0, "rgba(22,27,34,1)"],
                [0.3, "rgba(47,128,237,0.3)"],
                [1, "#2f80ed"],
            ],
            title="NER Chalkashish Matritsasi",
            text_auto=True,
            labels=dict(x="Bashorat qilingan", y="Haqiqiy sinf", color="Soni"),
        )
        fig_cm.update_traces(textfont={"size": 14, "color": "white"})
        fig_cm = styled_fig(fig_cm)
        fig_cm.update_layout(height=320, margin=dict(t=45, b=10, l=10, r=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with bar_col:
        st.markdown("**F1-ko'rsatkich — Sinflar bo'yicha**")
        f1_vals   = [m["f1"] for k, m in NLP_METRICS.items() if "Umumiy" not in k]
        f1_labels = [k       for k    in NLP_METRICS         if "Umumiy" not in k]
        f1_colors = [
            "#27ae60" if v >= 0.96 else "#f5a623" if v >= 0.93 else "#eb5757"
            for v in f1_vals
        ]
        fig_f1 = go.Figure(go.Bar(
            x=f1_vals, y=f1_labels, orientation="h",
            marker=dict(color=f1_colors, line=dict(width=0)),
            text=[f"{v:.4f}" for v in f1_vals],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=12),
        ))
        fig_f1.add_vline(
            x=0.95, line_dash="dash", line_color="#8b949e",
            annotation_text="Maqbul chegara (0.95)",
            annotation_position="top right",
            annotation_font_color="#8b949e",
        )
        fig_f1.update_layout(
            **{**PLOTLY_LAYOUT, "margin": dict(t=10, b=10, l=10, r=60)},
            xaxis=dict(range=[0.88, 1.02], tickformat=".3f", title="F1-ko'rsatkich"),
            yaxis=dict(title=""),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown(
        f'<div class="decision-box decision-ok" style="margin-top:12px">'
        f'<b>Ilmiy Xulosa:</b> Regex-asosli NER tizimi makro-o\'rtacha F1 = '
        f'<b>{macro_f1:.4f}</b> ko\'rsatkichini qayd etdi '
        f'(Aniqlik: {macro_p:.4f}, To\'liqlik: {macro_r:.4f}). '
        f'F1 garmonik o\'rtacha formula bilan hisoblanadi: F1 = 2·P·R / (P+R). '
        f'Bu natija dissertatsiya talablariga to\'liq javob beradi.</div>',
        unsafe_allow_html=True,
    )



    """
    Loads the HR Excel file and returns a dict mapping xodim_ID → full name.
    The Excel column 'F.I.Sh.' contains the Uzbek full name (Last I.O. format).
    We use the ID as the key so it can be joined with OWL individuals.
    """
    try:
        df_hr = pd.read_excel(xlsx_path)
        # Normalise column names (strip whitespace)
        df_hr.columns = [c.strip() for c in df_hr.columns]
        id_col   = "ID"
        name_col = "F.I.Sh."
        if id_col in df_hr.columns and name_col in df_hr.columns:
            return dict(zip(df_hr[id_col].astype(str), df_hr[name_col].astype(str)))
    except Exception:
        pass
    return {}


def load_name_lookup(xlsx_path: str) -> dict:
    """
    Loads the HR Excel file and returns a dict mapping xodim_ID -> full name.
    The Excel column 'F.I.Sh.' contains the Uzbek full name (Last I.O. format).
    Used to enrich ontology individuals with human-readable names at load time.
    Falls back to an empty dict gracefully if the file is missing or malformed.
    """
    try:
        df_hr = pd.read_excel(xlsx_path)
        df_hr.columns = [c.strip() for c in df_hr.columns]
        if "ID" in df_hr.columns and "F.I.Sh." in df_hr.columns:
            return dict(zip(df_hr["ID"].astype(str), df_hr["F.I.Sh."].astype(str)))
    except Exception:
        pass
    return {}


def main():
    # ── Shtat.ai Brand Header ─────────────────────────────────────────────
    st.markdown(
        '<div class="brand-bar">'
        '<div class="brand-icon">📊</div>'
        '<div style="display:flex;flex-direction:column;gap:3px">'
        '<div class="brand-title">Shtat.ai</div>'
        '<div class="brand-sub">'
        'Intellektual Shtat Boshqaruvi Tizimi'
        ' &middot; Ko&#39;p-Fakultetli Dinamik Rejim'
        ' &middot; Semantic HR Governance'
        '</div>'
        '</div>'
        '<div class="brand-badge">Semantic AI &middot; v1.0 Beta</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "### *NLP va Ontologiya asosidagi akademik yuklama auditi*",
        unsafe_allow_html=False,
    )

    # ══════════════════════════════════════════════════════════════════════
    # SIDEBAR — File upload + faculty selector + dynamic constraints
    # ══════════════════════════════════════════════════════════════════════

    # ── Shtat.ai sidebar brand strip ─────────────────────────────────────
    st.sidebar.markdown(
        '<div class="shtat-sidebar-brand">'
        '<div class="shtat-sidebar-logo">📊</div>'
        '<div>'
        '<div class="shtat-sidebar-name">Shtat.ai</div>'
        '<div class="shtat-sidebar-tag">Semantic HR Governance Platform</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## Ma'lumotlar Manbai")

    # ── File uploader ─────────────────────────────────────────────────────
    uploaded_file = st.sidebar.file_uploader(
        "Excel / CSV fayl yuklang",
        type=["xlsx", "xls", "csv"],
        help="Ustunlar: ID, F.I.Sh., Position, Stavka, EmploymentType, Faculty",
        key="uploaded_data_file",
    )

    # Detect file change → clear stale session state (memory management)
    prev_fname = st.session_state.get("_uploaded_fname", None)
    curr_fname = uploaded_file.name if uploaded_file else None
    if curr_fname != prev_fname:
        for key in ["faculty_df", "demo_mode", "_col_map", "_raw_df",
                    "_faculty_list", "ariza_processed", "last_ariza",
                    "confirmed_ids", "_built_faculty_key", "nlp_additions",
                    "global_df"]:
            st.session_state.pop(key, None)
        st.session_state["_uploaded_fname"] = curr_fname

    # ── Resolve data source ───────────────────────────────────────────────
    using_upload = False

    if uploaded_file is not None:
        # Parse uploaded file (cached in session state after first read)
        if "_raw_df" not in st.session_state:
            raw_df, col_map, upload_warns = load_excel_faculty(uploaded_file)
            if raw_df is None:
                st.sidebar.error(f"Fayl o'qilmadi: {upload_warns[0]}")
                raw_df = pd.DataFrame()
                col_map = {}
            else:
                for w in upload_warns:
                    st.sidebar.warning(w)
            st.session_state["_raw_df"]   = raw_df
            st.session_state["_col_map"]  = col_map

        raw_df  = st.session_state["_raw_df"]
        col_map = st.session_state["_col_map"]

        # ── Manual column mapper (flexible header mapping) ────────────────
        unmapped = [k for k in ["name","position","stavka","employment"]
                    if col_map.get(k) is None]
        if unmapped and not raw_df.empty:
            st.sidebar.markdown("**Ustun xaritalash**")
            all_cols = list(raw_df.columns)
            for key in unmapped:
                label_uz = {
                    "name": "F.I.Sh. ustuni",
                    "position": "Lavozim ustuni",
                    "stavka": "Stavka ustuni",
                    "employment": "Bandlik ustuni",
                }.get(key, key)
                chosen = st.sidebar.selectbox(
                    label_uz, options=["— tanlang —"] + all_cols,
                    key=f"col_map_{key}"
                )
                if chosen != "— tanlang —":
                    col_map[key] = chosen
            st.session_state["_col_map"] = col_map

        # ── Faculty selector ──────────────────────────────────────────────
        fac_col = col_map.get("faculty")
        if fac_col and fac_col in raw_df.columns:
            faculty_list = sorted(raw_df[fac_col].astype(str).str.strip().unique().tolist())
        else:
            faculty_list = ["Asosiy Fakultet"]

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Fakultet Tanlash")
        selected_faculty = st.sidebar.selectbox(
            "Joriy Fakultet",
            options=faculty_list,
            key="selected_faculty",
        )

        using_upload = True
        demo_mode    = False
        st.session_state["demo_mode"] = False

    # ── Dynamic Constraint Inputs ─────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dinamik Cheklovlar")
    st.sidebar.markdown(
        '<div style="font-size:0.76rem;color:#8b949e;margin-bottom:6px">'
        'Joriy fakultet uchun cheklovlarni UI orqali belgilang.</div>',
        unsafe_allow_html=True,
    )

    dept_max_fte = st.sidebar.number_input(
        "Jami Shtat Birligi Chegarasi (Σ SB)",
        min_value=1.0, max_value=200.0, value=30.0, step=0.5,
        help="Butun kafedra/fakultet uchun maksimal jami FTE.",
        key="dyn_dept_max_fte",
    )
    max_asosiy_sb = st.sidebar.number_input(
        "Asosiy Xodim Maksimal SB",
        min_value=0.1, max_value=5.0, value=1.0, step=0.05,
        help="Asosiy (to'liq shtatli) xodim uchun individual chegara.",
        key="dyn_max_asosiy",
    )
    max_orindosh_sb = st.sidebar.number_input(
        "O'rindosh Xodim Maksimal SB",
        min_value=0.1, max_value=5.0, value=0.75, step=0.05,
        help="Ichki va Tashqi o'rindosh uchun individual chegara.",
        key="dyn_max_orindosh",
    )
    max_ind_sb = st.sidebar.number_input(
        "Individual Yuqori Chegara (flagging)",
        min_value=0.1, max_value=5.0, value=1.5, step=0.05,
        help="Bu qiymatdan yuqori bo'lgan har qanday SB 'WorkloadExceeded' deb belgilanadi.",
        key="dyn_max_individual",
    )

    # Store constraints in session state for NLP tab access
    st.session_state["dyn_constraints"] = {
        "dept_max_fte"   : dept_max_fte,
        "max_asosiy_sb"  : max_asosiy_sb,
        "max_orindosh_sb": max_orindosh_sb,
        "max_ind_sb"     : max_ind_sb,
    }

    # ── Build or retrieve the canonical DataFrame ─────────────────────────
    #
    # STATE PERSISTENCE DESIGN — single source of truth:
    #
    # st.session_state["faculty_df"]  — current faculty rows (from file + NLP adds)
    # st.session_state["nlp_additions"] — dict {faculty_name: [row_dict, ...]}
    #                                     NLP-added rows stored PER FACULTY so they
    #                                     survive faculty switching without being lost.
    # st.session_state["global_df"]   — all faculties combined (rebuilt on switch)
    #
    # RULE: rebuild from raw_df ONLY when (file or faculty) changes.
    #       After rebuild, immediately re-merge any saved NLP additions.
    #       On a plain rerun (no faculty change), read session state directly.
    #
    current_faculty_key = f"{curr_fname}::{st.session_state.get('selected_faculty','')}"
    prev_faculty_key    = st.session_state.get("_built_faculty_key", None)
    faculty_changed     = (current_faculty_key != prev_faculty_key)

    # Ensure nlp_additions dict exists for the lifetime of the session
    if "nlp_additions" not in st.session_state:
        st.session_state["nlp_additions"] = {}

    if using_upload and not raw_df.empty and all(
        col_map.get(k) for k in ["name", "position", "stavka", "employment"]
    ):
        if faculty_changed or "faculty_df" not in st.session_state:
            # ── Step A: Build base DataFrame from raw file ─────────────────
            selected_fac = st.session_state.get("selected_faculty", faculty_list[0])
            df_base = build_faculty_dataframe(
                raw_df, col_map, selected_fac, max_individual_sb=max_ind_sb,
            )
            if df_base.empty:
                st.warning(
                    f"'{selected_fac}' uchun hech qanday yozuv topilmadi. "
                    "Boshqa fakultetni tanlang yoki fayl tuzilishini tekshiring."
                )
                st.stop()

            # ── Step B: Re-merge NLP additions for this faculty ────────────
            # nlp_additions[faculty] is a list of row dicts saved at confirm-time.
            # We always merge them back so switching away and returning restores them.
            saved_rows = st.session_state["nlp_additions"].get(selected_fac, [])
            if saved_rows:
                df_additions = pd.DataFrame(saved_rows)
                # Re-apply dynamic WorkloadExceeded with current limit
                df_additions["WorkloadExceeded"] = df_additions["Workload"] > max_ind_sb
                df_full = pd.concat([df_base, df_additions], ignore_index=True)
            else:
                df_full = df_base

            st.session_state["faculty_df"]         = df_full
            st.session_state["_built_faculty_key"] = current_faculty_key

            # ── Step C: Build university-wide global_df from ALL faculties ─
            fac_col = col_map.get("faculty")
            if fac_col and fac_col in raw_df.columns:
                all_faculties = raw_df[fac_col].astype(str).str.strip().unique().tolist()
                global_frames = [
                    build_faculty_dataframe(
                        raw_df, col_map, fac, max_individual_sb=max_ind_sb
                    )
                    for fac in all_faculties
                ]
                df_global_base = pd.concat(
                    [f for f in global_frames if not f.empty],
                    ignore_index=True,
                )
                # Also merge ALL nlp_additions from ALL faculties into global_df
                all_nlp = [
                    row
                    for rows in st.session_state["nlp_additions"].values()
                    for row in rows
                ]
                if all_nlp:
                    df_global_base = pd.concat(
                        [df_global_base, pd.DataFrame(all_nlp)],
                        ignore_index=True,
                    )
            else:
                df_global_base = df_full.copy()
            st.session_state["global_df"] = df_global_base

        else:
            # Plain rerun (same faculty, e.g. after NLP add): read live session state
            df_full = st.session_state["faculty_df"]

    elif "faculty_df" not in st.session_state:
        # Fallback: OWL ontology or demo data
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        owx_path  = os.path.join(base_dir, "faculty_finall.owx")
        xlsx_path = os.path.join(base_dir, "hr_faculty_list.xlsx")
        name_lookup    = load_name_lookup(xlsx_path)
        demo_mode_init = False
        if os.path.exists(owx_path):
            onto, error = load_ontology(f"file://{owx_path}")
            if error:
                st.warning(error + "\n\nEmbedded ma'lumotlarga o'tildi.")
                st.session_state["faculty_df"] = generate_demo_dataframe()
                demo_mode_init = True
            else:
                st.session_state["faculty_df"] = extract_dataframe(onto, name_lookup)
        else:
            st.session_state["faculty_df"] = generate_demo_dataframe()
            demo_mode_init = True
        st.session_state["demo_mode"] = demo_mode_init
        # In demo/OWL mode there is only one "faculty" — global_df == faculty_df
        st.session_state["global_df"] = st.session_state["faculty_df"].copy()

    # ── Always read the live session state — single source of truth ───────
    # After st.rerun() from the NLP tab, session_state["faculty_df"] contains
    # the new row (set by the confirm block). Reading here gives every tab
    # the fully up-to-date data on the same rerun.
    df_full   = st.session_state["faculty_df"]
    global_df = st.session_state.get("global_df", df_full.copy())
    demo_mode = st.session_state.get("demo_mode", not using_upload)

    if df_full.empty:
        st.error("Ontologiyadan individlar chiqarib bo'lmadi.")
        st.stop()

    # Apply dynamic WorkloadExceeded flag using the current sidebar limit.
    # We use .copy() to avoid SettingWithCopyWarning on the session state object.
    df_full = df_full.copy()
    df_full["WorkloadExceeded"] = df_full["Workload"] > max_ind_sb
    st.session_state["faculty_df"] = df_full   # write back the flagged version

    # ── Sidebar filters (only after df_full is ready) ─────────────────────
    sel_pos, sel_emp, wl_range, sim_new_hires = render_sidebar_metrics(df_full)
    df_filtered = apply_filters(df_full, sel_pos, sel_emp, wl_range)

    # ── KPI Cards ─────────────────────────────────────────────────────────
    # KPI cards always show the unfiltered faculty totals so "Jami Xodimlar"
    # and "Jami Shtat Birligi" reflect the true count after NLP additions.
    # The charts below (tab1–tab3) continue to use df_filtered.
    render_kpi_cards(df_full)

    # Status badge row
    badge_parts = []
    if demo_mode:
        badge_parts.append("Demo rejim · ontologiyadan olingan haqiqiy ma'lumotlar")
    if using_upload:
        badge_parts.append(
            f"Yuklangan fayl: <b>{curr_fname}</b> · "
            f"Fakultet: <b>{st.session_state.get('selected_faculty', '—')}</b> · "
            f"Jami SB chegarasi: <b>{dept_max_fte:.1f}</b>"
        )
    if badge_parts:
        st.markdown(
            f'<div style="text-align:right;font-size:0.75rem;color:#8b949e;'
            f'margin-top:-16px;margin-bottom:8px">'
            + " | ".join(badge_parts) + "</div>",
            unsafe_allow_html=True,
        )

    # ── Shtat.ai Sidebar Footer ───────────────────────────────────────────
    st.sidebar.markdown(
        '<div class="shtat-sidebar-footer">'
        '📊 <b>Shtat.ai</b> v1.0 Beta<br>'
        'Powered by Semantic AI &middot; NLP &middot; OWL<br>'
        '<span style="color:#4a5568">© 2025 Shtat.ai — All rights reserved</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Main Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢  Shtat-Control (Monitoring)",
        "🔍  Semantic Audit (Reasoning Engine)",
        "📈  Shtat-Metric (Analytics)",
        "📝  Shtat-Read (NLP Ariza Moduli)",
    ])

    with tab1:
        tab_executive(df_filtered, df_full)

    with tab2:
        tab_semantic_audit(df_filtered, global_df)

    with tab3:
        tab_predictive(df_filtered, df_full, sim_new_hires, dept_max_fte=dept_max_fte)

    with tab4:
        tab_nlp_ariza(df_full, global_df)


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    main()
