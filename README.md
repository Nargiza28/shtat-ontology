# Shtat.ai — Semantic HR Governance Platform
### Ontological Model of the Semantic Structure of a Sentence in Uzbek

> **Master's Thesis Project** · TATU (Muhammad Al-Xorazmiy nomidagi Toshkent Axborot Texnologiyalari Universiteti)
> **Specialty:** 70610304 — Data Science
> **Author:** Xudoykulova Nargiza Ravshanovna
> **Supervisor:** PhD, Dotsent E.Sh. Nazirova
> **Registration:** Software Certificate № DGU 63730, Republic of Uzbekistan (05.05.2026)

---

## What This Project Does

Shtat.ai implements **OSHRA** — the *Ontology-Steered Hybrid Reasoning Algorithm* — which automatically validates Uzbek official-administrative documents using a combination of:

- **OWL 2 DL ontology** (`faculty_finall.owx`) encoding formal semantic constraints
- **Regex-based NER pipeline** extracting four semantic slots from Uzbek ariza sentences
- **Entity Resolution (STF-ER)** cross-validating the signer identity against the document subject
- **ERI/Didox protocol verification** authenticating the digital signature layer

The system processes a sentence like:

```
Men, Nargiza Xudoykulova, Kompyuter injiniringi kafedrasi uchun
1.0 shtatda asosiy o'rindosh sifatida Dotsent lavozimiga
qabul qilinishimni so'rayman.
```

...and maps it to an OWL ABox assertion:

```turtle
:xodim_nlp_nargiza_xudoykulova
    a owl:NamedIndividual, :Dotsent, :Asosiy ;
    :shtat_birligi "1.0"^^xsd:decimal ;
    :hasRecipient :Kompyuter_injiniringi_kafedrasi .
```

---

## Quick Start

```bash
# 1. Clone or download the project folder
cd shtatai_project/

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run main_v9_eri4.py
```

Open your browser at **http://localhost:8501**

> **Java 11+ must be installed** for the Pellet OWL reasoner.
> See `setup_guide.txt` § 1 for platform-specific instructions.

---

## Project Structure

```
shtatai_project/
│
├── main_v9_eri4.py          # Main application — OSHRA + ERI + Streamlit GUI
├── faculty_finall.owx       # OWL 2 DL ontology (5 classes, 14 properties, 7 rules)
├── hr_faculty_list.xlsx     # Sample faculty HR data
├── requirements.txt         # Python dependencies with pinned versions
├── setup_guide.txt          # Step-by-step installation guide for professors
└── README.md                # This file
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OSHRA Pipeline                        │
│                                                         │
│  Layer 0  ERI/Didox Protocol Verification               │
│           фиш + ЖШШИР → Identity_Valid                  │
│                  ↓                                       │
│  Layer 1  OWL Ontology Loading (faculty_finall.owx)     │
│           5 classes · 14 properties · Pellet reasoner   │
│                  ↓                                       │
│  Layer 2  Relational Audit (C0–C5 constraints)          │
│           Completeness · Class membership · FTE limits  │
│                  ↓                                       │
│  Layer 3  Predictive Analytics (Kafedrometr)            │
│           Stable / Caution / Critical zones             │
│                  ↓                                       │
│  Layer 4  NLP Entity Extraction — extract_entities()    │
│           ism · lavozim · shtat · bandlik               │
│                  ↓                                       │
│  Layer 5  Entity Resolution — STF-ER                    │
│           resolve(S,A) = 0.70×Lastname + 0.30×First     │
│           → AUTH(D) = Identity_Valid ∧ Action_Valid     │
└─────────────────────────────────────────────────────────┘
```

---

## Dashboard Tabs

| Tab | Uzbek Name | Description |
|-----|-----------|-------------|
| 📊 Monitoring | Shtat-Control | Faculty workload KPI cards and charts |
| 🔍 Audit | Semantic Audit | Cross-faculty cumulative workload check |
| 📈 Analytics | Shtat-Metric | Kafedrometr capacity gauge |
| 📋 NLP | Shtat-Read | Ariza sentence input and OSHRA processing |

---

## Ontology Model

The OWL 2 DL ontology encodes the semantic structure of Uzbek administrative sentences:

| Class | Subclasses | Role |
|-------|-----------|------|
| `Event` | RequestEvent, DirectiveEvent | Predicate / speech act |
| `Participant` | IndividualActor, InstitutionalActor | Agent / Recipient |
| `Context` | QuantityContext, TemporalContext | Circumstances |
| `Modality` | Asosiy, Orindosh, Prohibition | Normative force |
| `DocumentEntity` | — | The document itself |

**Validation rules (C0–C5):**

| Code | Rule | Condition |
|------|------|-----------|
| C0 | Completeness | All mandatory slots ≠ ∅ |
| C1 | Class membership | lavozim ∈ KNOWN_POSITIONS |
| C2 | FTE ceiling | Σ shtat_birligi ≤ dept_max |
| C3 | Part-time limit | Orindosh → SB ≤ 0.75 |
| C4 | Full-time limit | Asosiy → SB ≤ 1.0 |
| C5 | Uniqueness | ¬∃x: ν(ism) = ν(x) in ABox |

---

## Evaluation Results

Evaluated on 120 annotated Uzbek ariza sentences (5-fold stratified cross-validation):

| Semantic Slot | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| ism (Applicant Name) | 0.9907 | 0.9060 | 0.9464 |
| lavozim (Position) | 1.0000 | 0.9496 | 0.9741 |
| shtat (Workload) | 1.0000 | 0.9746 | 0.9871 |
| bandlik (Employment Type) | 1.0000 | 0.9573 | 0.9782 |
| **Macro Average** | **0.9977** | **0.9469** | **0.9714** |

Inter-annotator agreement: **Cohen κ = 0.91** (almost perfect, Landis & Koch scale)

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.35.0 | Web GUI framework |
| `pandas` | 2.1.4 | Data layer / Excel loading |
| `owlready2` | 0.46 | OWL ontology loading + Pellet reasoner |
| `plotly` | 5.20.0 | Interactive charts |
| `pdfplumber` | 0.11.0 | PDF text extraction (Didox protocol) |
| `pymupdf` | 1.24.3 | PDF fallback (PyMuPDF/fitz) |
| `pyhanko` | 0.35.1 | ERI digital signature detection |
| `rdflib` | 7.0.0 | OWL Turtle ABox generation |
| `lxml` | 5.2.1 | XML/OWL parsing |
| `numpy` | 1.26.4 | Numerical operations |
| `openpyxl` | 3.1.2 | Excel file I/O |
| `cryptography` | 41.0.7 | Cryptographic operations (pyhanko dep.) |

**System requirement:** Java 11+ (for Pellet OWL reasoner via owlready2)

---

## Troubleshooting

**Pellet reasoner not working:**
```bash
java -version   # must show Java 11 or higher
```

**Virtual environment not activated:**
```bash
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

**Port already in use:**
```bash
streamlit run main_v9_eri4.py --server.port 8502
```

For full troubleshooting, see `setup_guide.txt`.

---

## Citation

```
Xudoykulova, N.R. (2026). Ontological Model of the Semantic Structure of a
Sentence in Uzbek. Master's Thesis. Tashkent University of Information
Technologies named after Muhammad al-Khorazmi (TATU).
Software: Shtat.ai, Certificate № DGU 63730, Republic of Uzbekistan.
```

---

*Built with Python · OWL 2 DL · Streamlit · Pellet Reasoner*
