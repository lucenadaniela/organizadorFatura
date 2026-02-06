import io
import json
import re
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False

APP_TITLE = "Totais por pessoa (fatura)"
RULES_FILE = "regras_pagamentos.json"
PARCELAS_FILE = "regras_parcelamento.json"

# ---------- helpers ----------
def norm(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", " ", s)

def parse_valor_regra(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and pd.notna(x):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    s = s.replace("R$", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def valor_bate(v1: float, v2: float, tol: float = 0.01) -> bool:
    return abs(float(v1) - float(v2)) <= tol

# ---------- regras ----------
def load_rules() -> pd.DataFrame:
    if Path(RULES_FILE).exists():
        data = json.loads(Path(RULES_FILE).read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
        for c in ["tipo", "palavra_chave", "valor", "pessoa", "categoria"]:
            if c not in df.columns:
                df[c] = ""
        return df.fillna("")
    # fallback simples (edite no app “completo” ou crie o json)
    return pd.DataFrame([
        {"tipo":"fixo","palavra_chave":"spotify","valor":"","pessoa":"Yves","categoria":"Fixos"},
        {"tipo":"variavel","palavra_chave":"uber","valor":"","pessoa":"Daiane","categoria":"Uber"},
    ]).fillna("")

def load_parcelas_rules() -> dict:
    if Path(PARCELAS_FILE).exists():
        return json.loads(Path(PARCELAS_FILE).read_text(encoding="utf-8"))
    return {}

# ---------- parcelas ----------
RE_PARCELA = re.compile(r"(?:parcela\s*)?(?P<atual>\d{1,2})\s*(?:/|de)\s*(?P<total>\d{1,2})", re.IGNORECASE)

def extrair_parcela(desc: str):
    m = RE_PARCELA.search(desc or "")
    if not m:
        return "", None, None
    return f"{int(m.group('atual'))}/{int(m.group('total'))}", int(m.group("atual")), int(m.group("total"))

def remover_texto_parcela(desc: str) -> str:
    return RE_PARCELA.sub("", desc or "").strip()

def gerar_id_parcelamento(desc: str, valor: float):
    base = f"{remover_texto_parcela(desc)}|{float(valor):.2f}".lower()
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:10]

# ---------- classificação ----------
def classify_manual(desc: str, valor_lanc: float, rules_df: pd.DataFrame, default_person: str, default_cat: str):
    d = norm(desc)
    rules = rules_df.copy().fillna("")
    rules["kw_norm"] = rules["palavra_chave"].astype(str).apply(norm)
    rules["kw_len"] = rules["kw_norm"].apply(len)
    rules["valor_float"] = rules["valor"].apply(parse_valor_regra)
    rules["tem_valor"] = rules["valor_float"].apply(lambda v: 1 if v is not None else 0)
    rules = rules.sort_values(["tem_valor", "kw_len"], ascending=[False, False])

    for _, r in rules.iterrows():
        kw = r["kw_norm"]
        if not kw or kw not in d:
            continue
        vr = r["valor_float"]
        if vr is not None and not valor_bate(valor_lanc, vr):
            continue
        return r.get("pessoa","") or default_person, r.get("categoria","") or default_cat, "manual"
    return default_person, default_cat, "fallback"

def classify(desc, valor, id_parc, regras_parc, rules_df, default_person, default_cat):
    # regra automática de parcelamento (se existir e estiver ativa)
    if id_parc and id_parc in regras_parc and not regras_parc[id_parc].get("concluido", False):
        r = regras_parc[id_parc]
        return r.get("pessoa", default_person), r.get("categoria", default_cat), "parcelamento:auto"
    return classify_manual(desc, valor, rules_df, default_person, default_cat)

# ---------- parser PDF Nubank ----------
LINHA_RE = re.compile(r"^(?P<dia>\d{2})\s(?P<mes>[A-Z]{3})\s(?P<desc>.+?)\sR\$\s(?P<valor>[\d\.,]+)$")
MESES = {"JAN":"01","FEV":"02","MAR":"03","ABR":"04","MAI":"05","JUN":"06","JUL":"07","AGO":"08","SET":"09","OUT":"10","NOV":"11","DEZ":"12"}

def parse_nubank_pdf(file_bytes: bytes, ano: int):
    if not PDF_OK:
        raise RuntimeError("pdfplumber não instalado. Instale: pip install pdfplumber")
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

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

rules = load_rules()
regras_parc = load_parcelas_rules()

with st.expander("⚙️ Config (rápida)", expanded=True):
    default_person = st.text_input("Pessoa padrão (se não casar regra)", value="Pendente")
    default_cat = st.text_input("Categoria padrão (se não casar regra)", value="Revisar")
    ano = st.number_input("Ano da fatura", min_value=2020, max_value=2100, value=2026, step=1)
    mostrar_detalhes = st.checkbox("Mostrar detalhes (lista de lançamentos)", value=False)
    mostrar_pendencias = st.checkbox("Mostrar pendências (Pendente/Revisar)", value=True)

up = st.file_uploader("Suba a fatura (PDF Nubank com texto selecionável) ou CSV (data, descricao, valor)", type=["pdf", "csv"])

if up:
    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(up)
        # exige colunas: data, descricao, valor
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        if "descrição" in df.columns and "descricao" not in df.columns:
            df["descricao"] = df["descrição"]
        need = {"data","descricao","valor"}
        if not need.issubset(set(df.columns)):
            st.error("CSV precisa ter colunas: data, descricao, valor")
            st.stop()
        df = df[["data","descricao","valor"]].copy()
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date.astype(str)
        df = df.dropna(subset=["descricao","valor"])
    else:
        df = parse_nubank_pdf(up.getvalue(), ano=ano)

    if df.empty:
        st.warning("Não consegui extrair lançamentos.")
        st.stop()

    # parcelas + id
    parc = df["descricao"].astype(str).apply(extrair_parcela)
    df["parcela_atual"] = parc.apply(lambda x: x[1])
    df["parcela_total"] = parc.apply(lambda x: x[2])
    df["id_parcelamento"] = df.apply(lambda r: gerar_id_parcelamento(r["descricao"], r["valor"]), axis=1)

    # classificar
    pessoas, cats, fontes = [], [], []
    for _, r in df.iterrows():
        p, c, fonte = classify(str(r["descricao"]), float(r["valor"]), r["id_parcelamento"], regras_parc, rules, default_person, default_cat)
        pessoas.append(p); cats.append(c); fontes.append(fonte)

    df["pessoa"] = pessoas
    df["categoria"] = cats
    df["fonte_regra"] = fontes

    # ---- VISUAL ENXUTO ----
    total_geral = df["valor"].sum()
    st.metric("Total geral", f"R$ {total_geral:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    totais = df.groupby("pessoa", as_index=False)["valor"].sum().sort_values("valor", ascending=False)

    st.subheader("Totais por pessoa")
    cols = st.columns(min(6, len(totais))) if len(totais) > 0 else []
    for i, row in enumerate(totais.itertuples(index=False)):
        col = cols[i % len(cols)] if cols else st
        col.metric(str(row.pessoa), f"R$ {row.valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    # opcional: tabela por categoria
    st.subheader("Por categoria (resumo)")
    resumo_cat = (
        df.pivot_table(index="pessoa", columns="categoria", values="valor", aggfunc="sum", fill_value=0)
        .sort_index()
    )
    st.dataframe(resumo_cat, use_container_width=True)

    # pendências
    if mostrar_pendencias:
        pend = df[(df["pessoa"].str.lower() == "pendente") | (df["categoria"].str.lower() == "revisar")].copy()
        st.subheader("Pendências")
        if pend.empty:
            st.success("Nada pendente 🎯")
        else:
            st.dataframe(pend[["data","descricao","valor","pessoa","categoria","fonte_regra"]], use_container_width=True, height=260)

    # detalhes (se quiser)
    if mostrar_detalhes:
        st.subheader("Lançamentos (detalhe)")
        st.dataframe(df[["data","descricao","valor","pessoa","categoria","fonte_regra"]], use_container_width=True, height=420)

else:
    st.info("Suba uma fatura para ver os totais por pessoa.")
