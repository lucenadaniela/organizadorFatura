# extrator.py
import io
import json
import re
import hashlib
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False


APP_TITLE = "Resumo da fatura (regras + parcelamento)"
RULES_FILE = "regras_pagamentos.json"
PARCELAS_FILE = "regras_parcelamento.json"
PREFS_TRANS_FILE = "preferencias_transporte.json"     # uber/99 por pessoa (sem UI)
OVERRIDES_FILE = "overrides_lancamentos.json"         # edições manuais por lançamento


# =========================
# Helpers
# =========================
def norm(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", " ", s)

def norm_compact(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def brl(v: float) -> str:
    s = f"{v:,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")

def parse_valor_regra(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and pd.notna(x):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    s = s.replace("R$", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def valor_bate(v1: float, v2: float, tol: float = 0.01) -> bool:
    try:
        return abs(float(v1) - float(v2)) <= tol
    except Exception:
        return False


# =========================
# Persistência: Preferências Uber/99 (sem UI)
# =========================
def load_prefs_transporte() -> dict:
    if Path(PREFS_TRANS_FILE).exists():
        try:
            return json.loads(Path(PREFS_TRANS_FILE).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_prefs_transporte(data: dict):
    Path(PREFS_TRANS_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# =========================
# Persistência: Overrides (edições manuais por lançamento)
# =========================
def load_overrides() -> dict:
    if Path(OVERRIDES_FILE).exists():
        try:
            return json.loads(Path(OVERRIDES_FILE).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_overrides(data: dict):
    Path(OVERRIDES_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def lanc_id(data: str, desc: str, valor: float) -> str:
    base = f"{data}|{norm_compact(desc)}|{float(valor):.2f}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]


# =========================
# Regras (manuais)
# =========================
def load_rules() -> pd.DataFrame:
    if Path(RULES_FILE).exists():
        try:
            data = json.loads(Path(RULES_FILE).read_text(encoding="utf-8"))
            df = pd.DataFrame(data)
            for c in ["tipo", "palavra_chave", "valor", "pessoa", "categoria"]:
                if c not in df.columns:
                    df[c] = ""
            return df.fillna("")
        except Exception:
            pass

    # Defaults (sem uber Daiane sem valor)
    return pd.DataFrame([
        {"tipo": "fixo", "palavra_chave": "spotify",         "valor": "40,9",  "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "netflix",         "valor": "20,9",  "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "amazon prime",    "valor": "19,9",  "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "apple.com/bill",  "valor": "19,9",  "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "academia",        "valor": "149,9", "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "academia",        "valor": "133",   "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "nucel",           "valor": "30",    "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "variavel", "palavra_chave": "shein",         "valor": "136,50", "pessoa": "Maria",  "categoria": "Roupas"},
        {"tipo": "variavel", "palavra_chave": "mercado livre", "valor": "55,87",  "pessoa": "Maria",  "categoria": "Carregador"},
    ]).fillna("")

def save_rules(df: pd.DataFrame):
    df = df.fillna("")
    Path(RULES_FILE).write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# =========================
# Parcelamentos (auto)
# =========================
def load_parcelas_rules() -> dict:
    if Path(PARCELAS_FILE).exists():
        try:
            return json.loads(Path(PARCELAS_FILE).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_parcelas_rules(data: dict):
    Path(PARCELAS_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# =========================
# Parcelas
# =========================
RE_PARCELA = re.compile(r"(?:parcela\s*)?(?P<atual>\d{1,2})\s*(?:/|de)\s*(?P<total>\d{1,2})", re.IGNORECASE)

def extrair_parcela(desc: str):
    m = RE_PARCELA.search(desc or "")
    if not m:
        return "", None, None
    return f"{int(m.group('atual'))}/{int(m.group('total'))}", int(m.group("atual")), int(m.group("total"))

def remover_texto_parcela(desc: str) -> str:
    return RE_PARCELA.sub("", desc or "").strip()

def gerar_id_parcelamento(desc: str, valor: float):
    base = f"{norm_compact(remover_texto_parcela(desc))}|{float(valor):.2f}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:10]


# =========================
# Classificação (manual + parcelamento)
# =========================
def classify_manual(desc: str, valor_lanc: float, rules_df: pd.DataFrame, default_person: str, default_cat: str):
    d = norm(desc)
    d_comp = norm_compact(desc)

    rules = rules_df.copy().fillna("")
    rules["kw_norm"] = rules["palavra_chave"].astype(str).apply(norm)
    rules["kw_comp"] = rules["palavra_chave"].astype(str).apply(norm_compact)
    rules["kw_len"] = rules["kw_norm"].apply(len)

    rules["valor_float"] = rules["valor"].apply(parse_valor_regra)
    rules["tem_valor"] = rules["valor_float"].apply(lambda v: 1 if v is not None else 0)

    # prioridade: regra com valor > keyword longa > ordem na tabela
    rules = rules.reset_index().rename(columns={"index": "__ordem"})
    rules = rules.sort_values(
        ["tem_valor", "kw_len", "__ordem"],
        ascending=[False, False, True],
        kind="mergesort"
    )

    for _, r in rules.iterrows():
        kw = r["kw_norm"]
        kw_comp = r["kw_comp"]
        if not kw:
            continue

        if (kw not in d) and (kw_comp not in d_comp):
            continue

        vr = r["valor_float"]
        # se a regra tiver valor, exige bater. se estiver vazio, vale pra qualquer valor ✅
        if vr is not None and not valor_bate(valor_lanc, vr, tol=0.01):
            continue

        pessoa = r.get("pessoa", "") or default_person
        categoria = r.get("categoria", "") or default_cat
        return pessoa, categoria, "manual"

    return default_person, default_cat, "fallback"

def classify(desc: str, valor: float, id_parc: str, regras_parc: dict,
             rules_df: pd.DataFrame, default_person: str, default_cat: str):
    # prioridade: parcelamento salvo
    if id_parc and id_parc in regras_parc and not regras_parc[id_parc].get("concluido", False):
        r = regras_parc[id_parc]
        return (
            r.get("pessoa", default_person),
            r.get("categoria", default_cat),
            "parcelamento:auto"
        )
    return classify_manual(desc, valor, rules_df, default_person, default_cat)


# =========================
# Detectores Uber / 99
# =========================
RE_99_TOKEN = re.compile(r"(^|[^0-9])99([^0-9]|$)")

def is_uber(desc: str) -> bool:
    return "uber" in norm_compact(desc)

def is_99(desc: str) -> bool:
    dc = norm_compact(desc)
    if "99app" in dc or "99taxi" in dc or "99pop" in dc:
        return True
    raw = (desc or "").lower()
    return bool(RE_99_TOKEN.search(raw))


# =========================
# Parser PDF Nubank (texto selecionável)
# =========================
LINHA_RE = re.compile(r"^(?P<dia>\d{2})\s(?P<mes>[A-Z]{3})\s(?P<desc>.+?)\sR\$\s(?P<valor>[\d\.,]+)$")
MESES = {"JAN":"01","FEV":"02","MAR":"03","ABR":"04","MAI":"05","JUN":"06","JUL":"07","AGO":"08","SET":"09","OUT":"10","NOV":"11","DEZ":"12"}

def parse_nubank_pdf(file_bytes: bytes, ano: int):
    if not PDF_OK:
        raise RuntimeError("pdfplumber não está instalado (requirements.txt).")

    rows = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.splitlines():
                m = LINHA_RE.match(raw.strip())
                if not m:
                    continue
                mes = MESES.get(m.group("mes"))
                if not mes:
                    continue
                data = f"{ano}-{mes}-{m.group('dia')}"
                desc = m.group("desc").strip()
                valor = float(m.group("valor").replace(".", "").replace(",", "."))
                rows.append({"data": data, "descricao": desc, "valor": valor})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date.astype(str)
    return df


# =========================
# UI: Config + CSS (Tema Claro)
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed"  # ótimo pro celular
)

st.markdown("""
<style>
/* =========================
   BASE / MOBILE
   ========================= */
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}
.block-container {
  padding-top: 0.8rem;
  padding-bottom: 2rem;
  max-width: 1050px;
}
@media (max-width: 768px){
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}

/* =========================
   FUNDO (TEMA CLARO)
   ========================= */
.stApp{
  background: linear-gradient(180deg, #fbfbfd 0%, #f3f4f6 100%);
  color: #0f172a;
}
h1 { font-size: 1.35rem; margin-bottom: 0.15rem; color: #0f172a; }
h2 { font-size: 1.10rem; margin-top: 0.75rem; color: #0f172a; }
h3 { font-size: 1.00rem; margin-top: 0.6rem; color: #0f172a; }
.stCaption, small {
  color: rgba(15,23,42,0.80) !important;
  opacity: 1 !important;
}

/* =========================
   CARDS
   ========================= */
.card{
  background: #ffffff;
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 16px;
  padding: 14px 14px;
  margin: 8px 0;
  box-shadow: 0 6px 18px rgba(15,23,42,0.06);
}
.card-title{
  font-size: 0.85rem;
  color: rgba(15,23,42,0.70);
  margin-bottom: 8px;
}
.card-big{
  font-size: 1.45rem;
  font-weight: 750;
  color: #0f172a;
  letter-spacing: 0.2px;
}
.card-sub{
  font-size: 0.82rem;
  color: rgba(15,23,42,0.65);
  margin-top: 6px;
}

/* =========================
   ABAS (TABS) - CONTRASTE
   ========================= */
.stTabs [data-baseweb="tab"]{
  color: #0f172a !important;
  font-weight: 700 !important;
  background: #ffffff !important;
  border: 1px solid rgba(15,23,42,0.15) !important;
  border-radius: 12px !important;
  padding: 8px 12px !important;
}
.stTabs [aria-selected="true"]{
  outline: 2px solid rgba(37,99,235,0.35) !important;
}
@media (max-width: 768px){
  .stTabs [data-baseweb="tab"]{ font-size: 0.95rem !important; }
}

/* =========================
   FILE UPLOADER - LEGÍVEL
   ========================= */
[data-testid="stFileUploader"] section{
  background: #ffffff !important;
  border: 1px dashed rgba(15,23,42,0.25) !important;
  border-radius: 14px !important;
}
[data-testid="stFileUploader"] *{
  color: #0f172a !important;
  font-weight: 600;
}
@media (max-width: 768px){
  [data-testid="stFileUploader"] *{ font-size: 0.95rem !important; }
}

/* =========================
   DATAFRAMES / TABELAS
   ========================= */
[data-testid="stDataFrame"]{
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(15,23,42,0.10);
  background: #ffffff;
}
thead tr th{ font-size: 0.85rem; }
tbody tr td{ font-size: 0.90rem; }

/* =========================
   BOTÕES - CORRIGE TEXTO INVISÍVEL
   (seu problema da print)
   ========================= */
.stButton button,
button[kind="primary"],
button[kind="secondary"]{
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid rgba(15,23,42,0.18) !important;
  border-radius: 12px !important;
  padding: 0.55rem 0.85rem !important;
}

.stButton button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover{
  background: #f8fafc !important;
  color: #0f172a !important;
}

/* garante o texto/ícone dentro do botão */
.stButton button *{
  color: #0f172a !important;
}
.stButton button svg{
  fill: #0f172a !important;
  color: #0f172a !important;
}

/* =========================
   PEQUENO POLIMENTO
   ========================= */
hr { border-color: rgba(15,23,42,0.10) !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Carregamentos
# =========================
rules = load_rules()
regras_parc = load_parcelas_rules()
prefs_trans = load_prefs_transporte()
overrides = load_overrides()

# Defaults Uber/99 (sem UI). Se quiser mudar: edite preferencias_transporte.json no repo.
if not prefs_trans:
    prefs_trans = {
        "uber_categoria": "Uber",
        "n99_categoria": "99",
    }
    save_prefs_transporte(prefs_trans)

# Sidebar: Configurações (mobile-friendly)
st.sidebar.header("⚙️ Configurações")
default_person = st.sidebar.text_input("Pessoa padrão (sem regra)", value="Pendente")
default_cat = st.sidebar.text_input("Categoria padrão (sem regra)", value="Revisar")
ano = st.sidebar.number_input("Ano da fatura (PDF)", min_value=2020, max_value=2100, value=2026, step=1)
mostrar_categoria = st.sidebar.checkbox("Mostrar resumo por categoria", value=True)
mostrar_detalhes = st.sidebar.checkbox("Mostrar detalhes", value=True)

st.sidebar.caption("Uber/99 vêm de preferencias_transporte.json (sem UI).")
st.sidebar.caption("Pendências editadas salvam em overrides_lancamentos.json.")


# Upload
up = st.file_uploader("📤 Upload", type=["pdf", "csv"])


# =========================
# App: Abas
# =========================
tab_resumo, tab_pend, tab_regras, tab_parc, tab_det = st.tabs(
    ["📌 Resumo", "📝 Pendências", "🧠 Regras", "💳 Parcelas", "🧾 Detalhes"]
)

# =========================
# Regras (tab própria)
# =========================
with tab_regras:
    st.subheader("Regras")
    st.caption("Deixe 'valor' vazio para valer para qualquer valor. Use valor só para diferenciar compras na mesma loja.")
    rules_edited = st.data_editor(
        rules,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "tipo": st.column_config.SelectboxColumn("tipo", options=["fixo", "variavel", "outros"]),
            "palavra_chave": st.column_config.TextColumn("palavra_chave"),
            "valor": st.column_config.TextColumn("valor", help="Opcional. Ex.: 136,50"),
            "pessoa": st.column_config.TextColumn("pessoa"),
            "categoria": st.column_config.TextColumn("categoria"),
        },
        key="rules_editor"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Salvar regras", key="save_rules_btn"):
            save_rules(rules_edited)
            st.success("Regras salvas ✅")
    with c2:
        if st.button("🗑️ Resetar regras (apagar arquivo)", key="reset_rules_btn"):
            Path(RULES_FILE).unlink(missing_ok=True)
            st.warning("Arquivo removido. Recarregue (F5).")


# =========================
# Processamento
# =========================
if not up:
    with tab_resumo:
        st.info("Suba uma fatura para ver o resumo.")
    with tab_pend:
        st.info("Suba uma fatura para ver pendências.")
    with tab_parc:
        st.info("Suba uma fatura para gerenciar parcelamentos.")
    with tab_det:
        st.info("Suba uma fatura para ver detalhes.")
    st.stop()

# carregar dataframe
if up.name.lower().endswith(".csv"):
    df = pd.read_csv(up)
    df.columns = [c.strip().lower() for c in df.columns]
    if "descrição" in df.columns and "descricao" not in df.columns:
        df["descricao"] = df["descrição"]
    if not {"data", "descricao", "valor"}.issubset(set(df.columns)):
        st.error("CSV precisa ter colunas: data, descricao, valor")
        st.stop()
    df = df[["data", "descricao", "valor"]].copy()
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=["descricao", "valor"])
else:
    try:
        df = parse_nubank_pdf(up.getvalue(), ano=ano)
    except Exception as e:
        st.error(f"Erro lendo PDF: {e}")
        st.stop()

if df.empty:
    st.warning("Não consegui extrair lançamentos do arquivo.")
    st.stop()

# enriquecer parcelas
parc = df["descricao"].astype(str).apply(extrair_parcela)
df["parcela_txt"] = parc.apply(lambda x: x[0])
df["parcela_atual"] = parc.apply(lambda x: x[1])
df["parcela_total"] = parc.apply(lambda x: x[2])
df["desc_base"] = df["descricao"].astype(str).apply(remover_texto_parcela)
df["id_parcelamento"] = df.apply(lambda r: gerar_id_parcelamento(r["descricao"], r["valor"]), axis=1)

# id do lançamento (para overrides)
df["lanc_id"] = df.apply(lambda r: lanc_id(r["data"], r["descricao"], r["valor"]), axis=1)

# classificar (regras + parcelamento)
rules_live = rules_edited.fillna("") if "rules_edited" in locals() else rules.fillna("")
pessoas, cats, fonte = [], [], []
for _, r in df.iterrows():
    p, c, f = classify(
        str(r["descricao"]),
        float(r["valor"]),
        r["id_parcelamento"],
        regras_parc,
        rules_live,
        default_person,
        default_cat
    )
    pessoas.append(p); cats.append(c); fonte.append(f)
df["pessoa"] = pessoas
df["categoria"] = cats
df["fonte_regra"] = fonte

# flags uber/99
df["is_uber"] = df["descricao"].astype(str).apply(is_uber)
df["is_99"] = df["descricao"].astype(str).apply(is_99)

# aplica preferências Uber/99 (sem UI)

# aplica preferências Uber/99 (sem UI)
prefs_trans = load_prefs_transporte()  # se já tiver carregado, pode manter e remover esta linha

uber_pessoa = str(prefs_trans.get("uber_pessoa", "")).strip()
uber_cat    = str(prefs_trans.get("uber_categoria", "Uber")).strip()

n99_pessoa  = str(prefs_trans.get("n99_pessoa", "")).strip()
n99_cat     = str(prefs_trans.get("n99_categoria", "99")).strip()


if n99_pessoa:
    m = df["is_99"]
    df.loc[m, "pessoa"] = n99_pessoa
    df.loc[m, "categoria"] = n99_cat
    df.loc[m, "fonte_regra"] = "transporte:99"

# aplica overrides manuais (prioridade máxima)
if overrides:
    m = df["lanc_id"].isin(overrides.keys())
    if m.any():
        df.loc[m, "pessoa"] = df.loc[m, "lanc_id"].map(lambda k: overrides[k].get("pessoa"))
        df.loc[m, "categoria"] = df.loc[m, "lanc_id"].map(lambda k: overrides[k].get("categoria"))
        df.loc[m, "fonte_regra"] = "override:manual"


# =========================
# TAB: RESUMO
# =========================
with tab_resumo:
    total_geral = float(df["valor"].sum())
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Total geral</div>
      <div class="card-big">{brl(total_geral)}</div>
      <div class="card-sub">Lançamentos: {len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Totais por pessoa (no celular: 2 colunas)
    totais = df.groupby("pessoa", as_index=False)["valor"].sum().sort_values("valor", ascending=False)
    st.subheader("Totais por pessoa")
    ncols = 2  # mobile friendly
    cols = st.columns(ncols)
    for i, row in enumerate(totais.itertuples(index=False)):
        cols[i % ncols].markdown(
            f"""<div class="card">
                  <div class="card-title">{row.pessoa}</div>
                  <div class="card-big">{brl(float(row.valor))}</div>
                </div>""",
            unsafe_allow_html=True
        )

    # Transporte (contagem + total)
    st.subheader("Transporte")
    u_total = float(df.loc[df["is_uber"], "valor"].sum())
    u_qtd = int(df["is_uber"].sum())
    n_total = float(df.loc[df["is_99"], "valor"].sum())
    n_qtd = int(df["is_99"].sum())

    cU, c9 = st.columns(2)
   cU.markdown(f"""<div class="card">
    <div class="card-title">Uber</div>
    <div class="card-big">{brl(u_total)}</div>
    <div class="card-sub">Qtd: {u_qtd}</div>
</div>""", unsafe_allow_html=True)

c9.markdown(f"""<div class="card">
    <div class="card-title">99</div>
    <div class="card-big">{brl(n_total)}</div>
    <div class="card-sub">Qtd: {n_qtd}</div>
</div>""", unsafe_allow_html=True)

    if mostrar_categoria:
        st.subheader("Resumo por categoria")
        resumo_cat = (
            df.pivot_table(index="pessoa", columns="categoria", values="valor", aggfunc="sum", fill_value=0)
            .sort_index()
        )
        st.dataframe(resumo_cat, use_container_width=True, height=320)


# =========================
# TAB: PENDÊNCIAS (editável)
# =========================
with tab_pend:
    st.subheader("Pendências (editável)")
    st.caption("Edite pessoa/categoria e clique em salvar. Isso fica gravado em overrides_lancamentos.json.")
    pend = df[(df["pessoa"].str.lower() == "pendente") | (df["categoria"].str.lower() == "revisar")].copy()

    if pend.empty:
        st.success("Nada pendente 🎯")
    else:
        pessoas_opts = sorted(set([str(x) for x in df["pessoa"].dropna().unique().tolist()] + ["Pendente"]))
        cats_opts = sorted(set([str(x) for x in df["categoria"].dropna().unique().tolist()] + ["Revisar"]))

        pend_edit = st.data_editor(
            pend[["data", "descricao", "valor", "pessoa", "categoria", "lanc_id", "fonte_regra"]],
            use_container_width=True,
            hide_index=True,
            disabled=["data", "descricao", "valor", "lanc_id", "fonte_regra"],
            column_config={
                "pessoa": st.column_config.SelectboxColumn("pessoa", options=pessoas_opts),
                "categoria": st.column_config.SelectboxColumn("categoria", options=cats_opts),
            },
            height=420,
            key="pend_editor"
        )

        if st.button("💾 Salvar alterações das pendências", key="save_pend_btn"):
            for _, r in pend_edit.iterrows():
                overrides[str(r["lanc_id"])] = {
                    "pessoa": str(r["pessoa"]),
                    "categoria": str(r["categoria"]),
                }
            save_overrides(overrides)
            st.success("Salvo! ✅")
            st.rerun()


# =========================
# TAB: PARCELAS
# =========================
with tab_parc:
    st.subheader("Parcelamentos")
    st.caption("Quando você ensina 1x, as próximas parcelas caem automático pelo id do parcelamento.")
    df_parc = df[df["parcela_total"].notna()].copy()

    if df_parc.empty:
        st.info("Nenhum parcelado detectado nesta fatura.")
    else:
        df_parc["label"] = df_parc.apply(
            lambda r: f"{r['id_parcelamento']} | {r['desc_base'][:45]} | {brl(float(r['valor']))} | {r['parcela_txt']}",
            axis=1
        )
        sel = st.selectbox("Escolha o parcelamento", options=df_parc["label"].unique().tolist())
        sel_id = sel.split(" | ")[0].strip()

        ex = df_parc[df_parc["id_parcelamento"] == sel_id].iloc[0]
        pessoa_sel = st.text_input("Pessoa (para este parcelamento)", value=str(ex["pessoa"]))
        cat_sel = st.text_input("Categoria (para este parcelamento)", value=str(ex["categoria"]))

        cA, cB, cC = st.columns(3)
        with cA:
            if st.button("✅ Salvar", key="save_parc_btn"):
                regras_parc[sel_id] = {
                    "pessoa": pessoa_sel,
                    "categoria": cat_sel,
                    "parcelas_total": int(ex["parcela_total"]) if pd.notna(ex["parcela_total"]) else None,
                    "concluido": False,
                    "desc_base": str(ex["desc_base"]),
                }
                save_parcelas_rules(regras_parc)
                st.success("Parcelamento salvo! Próximas parcelas vão cair automático ✅")
        with cB:
            if st.button("🧾 Concluir", key="finish_parc_btn"):
                if sel_id in regras_parc:
                    regras_parc[sel_id]["concluido"] = True
                    save_parcelas_rules(regras_parc)
                    st.success("Marcado como concluído.")
        with cC:
            if st.button("🗑️ Remover", key="rm_parc_btn"):
                if sel_id in regras_parc:
                    regras_parc.pop(sel_id, None)
                    save_parcelas_rules(regras_parc)
                    st.warning("Regra removida.")

        st.divider()
        st.subheader("Parcelados nesta fatura")
        st.dataframe(
            df_parc[["data", "descricao", "valor", "parcela_txt", "id_parcelamento", "pessoa", "categoria", "fonte_regra"]],
            use_container_width=True,
            height=320
        )


# =========================
# TAB: DETALHES
# =========================
with tab_det:
    st.subheader("Detalhes")
    if not mostrar_detalhes:
        st.info("Ative 'Mostrar detalhes' no menu ⚙️ (sidebar).")
    else:
        st.dataframe(
            df[["data","descricao","valor","pessoa","categoria","fonte_regra","parcela_txt","id_parcelamento","lanc_id"]],
            use_container_width=True,
            height=520
        )
