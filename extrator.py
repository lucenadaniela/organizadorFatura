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


APP_TITLE = "Resumo da fatura (com regras + parcelamento)"
RULES_FILE = "regras_pagamentos.json"         # regras manuais: keyword + (valor opcional)
PARCELAS_FILE = "regras_parcelamento.json"    # regras automáticas: id_parcelamento


# =========================
# Helpers
# =========================
def norm(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", " ", s)

def norm_compact(s: str) -> str:
    """
    Normaliza agressivo para casar:
    'mercado livre' com 'MercadoLivre*...'
    """
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # remove acentos
    s = re.sub(r"[^a-z0-9]+", "", s)  # remove tudo que não é letra/número
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

    # Defaults (se não existir regras_pagamentos.json)
    return pd.DataFrame([
        # ===== FIXOS (do seu print) =====
        {"tipo": "fixo", "palavra_chave": "spotify",         "valor": "40,9",  "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "netflix",         "valor": "20,9",  "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "amazon prime",    "valor": "19,9",  "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "apple.com/bill",  "valor": "19,9",  "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "academia",        "valor": "149,9", "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "academia",        "valor": "133",   "pessoa": "Yves",   "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "nucel",           "valor": "30",    "pessoa": "Yves",   "categoria": "Fixos"},

        # ===== VARIÁVEIS (do seu print) =====
        {"tipo": "variavel", "palavra_chave": "uber",          "valor": "",       "pessoa": "Daiane", "categoria": "Uber"},
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
    base = f"{remover_texto_parcela(desc)}|{float(valor):.2f}".lower()
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:10]


# =========================
# Classificação
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

    # prioridade: regra com valor > keyword longa > ordem na tabela (estável)
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

        # casa se bater no texto normal OU compacto
        if (kw not in d) and (kw_comp not in d_comp):
            continue

        vr = r["valor_float"]
        if vr is not None and not valor_bate(valor_lanc, vr, tol=0.01):
            continue

        pessoa = r.get("pessoa", "") or default_person
        categoria = r.get("categoria", "") or default_cat
        return pessoa, categoria, "manual"

    return default_person, default_cat, "fallback"

def classify(desc: str, valor: float, id_parc: str, regras_parc: dict,
             rules_df: pd.DataFrame, default_person: str, default_cat: str):
    # 1) parcelamento automático tem prioridade
    if id_parc and id_parc in regras_parc and not regras_parc[id_parc].get("concluido", False):
        r = regras_parc[id_parc]
        return (
            r.get("pessoa", default_person),
            r.get("categoria", default_cat),
            "parcelamento:auto"
        )
    # 2) regras manuais
    return classify_manual(desc, valor, rules_df, default_person, default_cat)


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
# App
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

rules = load_rules()
regras_parc = load_parcelas_rules()

with st.expander("⚙️ Configurações", expanded=False):
    default_person = st.text_input("Pessoa padrão (se não casar regra)", value="Pendente")
    default_cat = st.text_input("Categoria padrão (se não casar regra)", value="Revisar")
    ano = st.number_input("Ano da fatura (PDF)", min_value=2020, max_value=2100, value=2026, step=1)

    colA, colB, colC = st.columns(3)
    with colA:
        mostrar_categoria = st.checkbox("Mostrar resumo por categoria", value=True)
    with colB:
        mostrar_pendencias = st.checkbox("Mostrar pendências", value=True)
    with colC:
        mostrar_detalhes = st.checkbox("Mostrar detalhes", value=False)

up = st.file_uploader("Upload PDF (Nubank texto selecionável) ou CSV (data, descricao, valor)", type=["pdf", "csv"])

# Expander: Regras
with st.expander("🧠 Regras (editar/cadastrar)", expanded=False):
    st.caption("Dica: deixe 'valor' vazio para valer para qualquer valor. Use valor só para diferenciar pessoas na mesma loja.")
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
        if st.button("💾 Salvar regras"):
            save_rules(rules_edited)
            st.success("Regras salvas.")
    with c2:
        if st.button("🗑️ Apagar regras (reset)"):
            Path(RULES_FILE).unlink(missing_ok=True)
            st.warning("Arquivo removido. Recarregue (F5).")

# Processamento
if up:
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

    parc = df["descricao"].astype(str).apply(extrair_parcela)
    df["parcela_txt"] = parc.apply(lambda x: x[0])
    df["parcela_atual"] = parc.apply(lambda x: x[1])
    df["parcela_total"] = parc.apply(lambda x: x[2])
    df["desc_base"] = df["descricao"].astype(str).apply(remover_texto_parcela)
    df["id_parcelamento"] = df.apply(lambda r: gerar_id_parcelamento(r["descricao"], r["valor"]), axis=1)

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

    # ---- RESUMO (tela principal) ----
    st.divider()
    total_geral = float(df["valor"].sum())
    st.metric("Total geral", brl(total_geral))

    totais = df.groupby("pessoa", as_index=False)["valor"].sum().sort_values("valor", ascending=False)
    st.subheader("Totais por pessoa")
    cols = st.columns(min(6, max(1, len(totais))))
    for i, row in enumerate(totais.itertuples(index=False)):
        cols[i % len(cols)].metric(str(row.pessoa), brl(float(row.valor)))

    if mostrar_categoria:
        st.subheader("Resumo por categoria")
        resumo_cat = (
            df.pivot_table(index="pessoa", columns="categoria", values="valor", aggfunc="sum", fill_value=0)
            .sort_index()
        )
        st.dataframe(resumo_cat, use_container_width=True)

    if mostrar_pendencias:
        st.subheader("Pendências")
        pend = df[(df["pessoa"].str.lower() == "pendente") | (df["categoria"].str.lower() == "revisar")].copy()
        if pend.empty:
            st.success("Nada pendente 🎯")
        else:
            st.dataframe(
                pend[["data", "descricao", "valor", "pessoa", "categoria", "fonte_regra"]],
                use_container_width=True,
                height=260
            )

    # ---- Ensinar parcelamentos ----
    with st.expander("📌 Ensinar parcelamentos (aplica até a última parcela)", expanded=False):
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
                if st.button("✅ Salvar para este parcelamento"):
                    regras_parc[sel_id] = {
                        "pessoa": pessoa_sel,
                        "categoria": cat_sel,
                        "parcelas_total": int(ex["parcela_total"]) if pd.notna(ex["parcela_total"]) else None,
                        "concluido": False,
                        "desc_base": str(ex["desc_base"]),
                    }
                    save_parcelas_rules(regras_parc)
                    st.success("Parcelamento salvo! Próximas parcelas vão cair automático.")
            with cB:
                if st.button("🧾 Marcar como concluído"):
                    if sel_id in regras_parc:
                        regras_parc[sel_id]["concluido"] = True
                        save_parcelas_rules(regras_parc)
                        st.success("Marcado como concluído.")
            with cC:
                if st.button("🗑️ Remover regra do parcelamento"):
                    if sel_id in regras_parc:
                        regras_parc.pop(sel_id, None)
                        save_parcelas_rules(regras_parc)
                        st.warning("Regra removida.")

    if mostrar_detalhes:
        with st.expander("🧾 Detalhes (lançamentos)", expanded=False):
            st.dataframe(
                df[["data","descricao","valor","pessoa","categoria","fonte_regra","parcela_txt","id_parcelamento"]],
                use_container_width=True,
                height=420
            )
else:
    st.info("Suba uma fatura para ver o resumo.")
