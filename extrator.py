import io
import json
import re
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

# PDF opcional (se não tiver, o app ainda funciona com CSV)
try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False


APP_TITLE = "Separador de Pagamentos (Regras + Parcelas Inteligentes)"
RULES_FILE = "regras_pagamentos.json"           # regras manuais: keyword + (valor opcional)
PARCELAS_FILE = "regras_parcelamento.json"      # regras automáticas: id_parcelamento -> pessoa/categoria até terminar


# =========================
# Normalização / conversões
# =========================
def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_valor_regra(x):
    """
    Converte valor da regra para float.
    Aceita: 136,5 | 136.5 | 'R$ 136,50' | vazio
    """
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

def valor_bate(valor_lanc: float, valor_regra: float, tol: float = 0.01) -> bool:
    try:
        return abs(float(valor_lanc) - float(valor_regra)) <= tol
    except Exception:
        return False


# =========================
# Regras manuais (editáveis) - 1 coluna extra: "valor"
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
# Regras automáticas de parcelamento (id_parcelamento)
# =========================
def load_parcelas_rules() -> dict:
    if Path(PARCELAS_FILE).exists():
        try:
            return json.loads(Path(PARCELAS_FILE).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_parcelas_rules(data: dict):
    Path(PARCELAS_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# Parcelas: extração + ID parcelamento
# =========================
RE_PARCELA = re.compile(
    r"(?:parcela\s*)?(?P<atual>\d{1,2})\s*(?:/|de)\s*(?P<total>\d{1,2})",
    re.IGNORECASE
)

def extrair_parcela(desc: str):
    m = RE_PARCELA.search(desc or "")
    if not m:
        return "", None, None
    atual = int(m.group("atual"))
    total = int(m.group("total"))
    return f"{atual}/{total}", atual, total

def remover_texto_parcela(desc: str) -> str:
    return RE_PARCELA.sub("", desc or "").strip()

def gerar_id_parcelamento(desc: str, valor: float):
    # ID estável (para o mesmo parcelamento)
    base = f"{remover_texto_parcela(desc)}|{float(valor):.2f}".lower()
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:10]


# =========================
# Classificação
# =========================
def classify_manual(desc: str, valor_lanc: float, rules_df: pd.DataFrame, default_person: str, default_cat: str):
    d = norm(desc)

    rules = rules_df.copy().fillna("")
    rules["kw_norm"] = rules["palavra_chave"].astype(str).apply(norm)
    rules["kw_len"] = rules["kw_norm"].apply(len)
    rules["valor_float"] = rules["valor"].apply(parse_valor_regra)
    rules["tem_valor"] = rules["valor_float"].apply(lambda v: 1 if v is not None else 0)

    # prioridade: tem valor > keyword longa
    rules = rules.sort_values(["tem_valor", "kw_len"], ascending=[False, False])

    for _, r in rules.iterrows():
        kw = r["kw_norm"]
        if not kw or kw not in d:
            continue

        vr = r["valor_float"]
        if vr is not None and not valor_bate(valor_lanc, vr, tol=0.01):
            continue

        pessoa = r.get("pessoa", "") or default_person
        categoria = r.get("categoria", "") or default_cat
        hit = r.get("palavra_chave", "")
        if vr is not None:
            hit = f"{hit} @ {vr:.2f}"
        return pessoa, categoria, hit

    return default_person, default_cat, ""

def classify_with_parcelas(desc: str, valor_lanc: float, id_parc: str, parc_atual, parc_total,
                          regras_parc: dict, rules_df: pd.DataFrame, default_person: str, default_cat: str):
    """
    1) Se existe regra automática por id_parcelamento e ainda não concluiu -> aplica
    2) Caso contrário -> usa regras manuais
    """
    # regra automática (id_parcelamento)
    if id_parc and id_parc in regras_parc:
        r = regras_parc[id_parc]
        # se já está concluído, não aplica mais (mas histórico no Excel fica)
        if not r.get("concluido", False):
            pessoa = r.get("pessoa", default_person)
            categoria = r.get("categoria", default_cat)
            return pessoa, categoria, "parcelamento:auto"

    # fallback manual
    return classify_manual(desc, valor_lanc, rules_df, default_person, default_cat)


# =========================
# Parser PDF Nubank (texto selecionável)
# =========================
LINHA_RE = re.compile(r"^(?P<dia>\d{2})\s(?P<mes>[A-Z]{3})\s(?P<desc>.+?)\sR\$\s(?P<valor>[\d\.,]+)$")
MESES = {"JAN":"01","FEV":"02","MAR":"03","ABR":"04","MAI":"05","JUN":"06","JUL":"07","AGO":"08","SET":"09","OUT":"10","NOV":"11","DEZ":"12"}

def parse_nubank_pdf(file_bytes: bytes, ano: int):
    if not PDF_OK:
        raise RuntimeError("pdfplumber não está instalado. Instale com: pip install pdfplumber")

    rows = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()
                m = LINHA_RE.match(line)
                if not m:
                    continue

                dia = m.group("dia")
                mes = MESES.get(m.group("mes"))
                if not mes:
                    continue

                desc = m.group("desc").strip()
                valor = float(m.group("valor").replace(".", "").replace(",", "."))

                rows.append({
                    "data": f"{ano}-{mes}-{dia}",
                    "descricao": desc,
                    "valor": valor
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date.astype(str)
        df = df.sort_values(["data", "valor"], ascending=[True, False], ignore_index=True)
    return df


# =========================
# Export Excel
# =========================
def to_excel_bytes(df_lanc: pd.DataFrame, regras_parc: dict):
    resumo_pessoa = (
        df_lanc.pivot_table(index="pessoa", values="valor", aggfunc="sum")
        .sort_values("valor", ascending=False)
    )
    resumo_cat = (
        df_lanc.pivot_table(index=["pessoa", "categoria"], values="valor", aggfunc="sum")
        .sort_values("valor", ascending=False)
    )

    # parcelados detalhados
    df_parc = df_lanc[df_lanc["parcela_total"].notna()].copy()

    # painel de parcelamentos (controle)
    if not df_parc.empty:
        df_parc["total_estimado_parcelamento"] = df_parc["valor"] * df_parc["parcela_total"].astype(float)
        parc_group = (
            df_parc.groupby(["id_parcelamento", "desc_base", "pessoa", "categoria"], as_index=False)
            .agg(
                valor_parcela=("valor", "mean"),
                parcelas_total=("parcela_total", "max"),
                maior_parcela_detectada=("parcela_atual", "max"),
                primeira_data=("data", "min"),
                ultima_data=("data", "max"),
                total_estimado=("total_estimado_parcelamento", "max"),
            )
        )
        # marca concluído se maior parcela detectada >= total
        parc_group["concluido_detectado"] = parc_group.apply(
            lambda r: True if pd.notna(r["parcelas_total"]) and pd.notna(r["maior_parcela_detectada"]) and
                            int(r["maior_parcela_detectada"]) >= int(r["parcelas_total"])
            else False,
            axis=1
        )
        # se existe regra salva, mostra status
        def status_regra(pid):
            rr = regras_parc.get(pid)
            if not rr:
                return "sem_regra"
            return "concluido" if rr.get("concluido", False) else "ativo"
        parc_group["status_regra"] = parc_group["id_parcelamento"].apply(status_regra)
        parc_group = parc_group.sort_values(["status_regra", "total_estimado"], ascending=[True, False])
    else:
        parc_group = pd.DataFrame(columns=[
            "id_parcelamento","desc_base","pessoa","categoria","valor_parcela","parcelas_total",
            "maior_parcela_detectada","primeira_data","ultima_data","total_estimado","concluido_detectado","status_regra"
        ])

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_lanc.to_excel(writer, index=False, sheet_name="lancamentos")
        resumo_pessoa.to_excel(writer, sheet_name="resumo_pessoa")
        resumo_cat.to_excel(writer, sheet_name="resumo_categoria")
        df_parc.to_excel(writer, index=False, sheet_name="parcelas_detalhe")
        parc_group.to_excel(writer, index=False, sheet_name="parcelamentos")
    out.seek(0)
    return out.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

rules = load_rules()
regras_parc = load_parcelas_rules()

tabs = st.tabs(["Regras", "Fatura", "Parcelamentos (auto)"])

# ---------- TAB 1: Regras ----------
with tabs[0]:
    st.subheader("Regras manuais (1 coluna extra: valor)")
    st.caption(
        "Preencha 'valor' apenas quando precisar diferenciar duas pessoas na mesma loja. "
        "Se deixar vazio, vale para qualquer valor daquela palavra-chave."
    )

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
        }
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Salvar regras manuais"):
            save_rules(rules_edited)
            st.success(f"Salvo em {RULES_FILE}")
    with c2:
        if st.button("🧹 Resetar regras manuais"):
            if Path(RULES_FILE).exists():
                Path(RULES_FILE).unlink()
            st.warning("Regras manuais resetadas. Recarregue a página (F5).")

# ---------- TAB 2: Fatura ----------
with tabs[1]:
    st.subheader("Classificar fatura + ensinar parcelamentos")

    default_person = st.text_input("Pessoa padrão (se não casar regra)", value="Pendente")
    default_cat = st.text_input("Categoria padrão (se não casar regra)", value="Revisar")
    ano = st.number_input("Ano da fatura (PDF)", min_value=2020, max_value=2100, value=2026, step=1)

    up = st.file_uploader("Upload PDF (Nubank texto selecionável) ou CSV (data, descricao, valor)", type=["pdf", "csv"])

    if up:
        # Carregar dados
        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up)
            cols = {c.lower().strip(): c for c in df.columns}
            if "descricao" not in cols and "descrição" in cols:
                cols["descricao"] = cols["descrição"]

            need = ["data", "descricao", "valor"]
            missing = [c for c in need if c not in cols]
            if missing:
                st.error(f"CSV inválido. Faltando colunas: {missing}. Necessário: data, descricao, valor.")
                st.stop()

            df = df.rename(columns={cols["data"]: "data", cols["descricao"]: "descricao", cols["valor"]: "valor"})
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
            df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date.astype(str)
            df = df.dropna(subset=["descricao", "valor"]).copy()
        else:
            try:
                df = parse_nubank_pdf(up.getvalue(), ano=ano)
            except Exception as e:
                st.error(f"Erro lendo PDF: {e}")
                st.stop()

        if df.empty:
            st.warning("Não consegui extrair lançamentos. Se o PDF for imagem/escaneado, precisa OCR.")
            st.stop()

        # Enriquecimentos: parcelas + id_parcelamento
        parcela_info = df["descricao"].astype(str).apply(extrair_parcela)
        df["parcela_txt"] = parcela_info.apply(lambda x: x[0])
        df["parcela_atual"] = parcela_info.apply(lambda x: x[1])
        df["parcela_total"] = parcela_info.apply(lambda x: x[2])
        df["desc_base"] = df["descricao"].astype(str).apply(remover_texto_parcela)
        df["id_parcelamento"] = df.apply(lambda r: gerar_id_parcelamento(r["descricao"], r["valor"]), axis=1)

        # Classificar usando: regra automática de parcelamento -> senão manual
        rules_live = rules_edited.fillna("")
        pessoas, cats, hits = [], [], []
        for _, row in df.iterrows():
            desc = str(row["descricao"])
            vl = float(row["valor"])
            pid = row.get("id_parcelamento", "")
            pa = row.get("parcela_atual")
            pt = row.get("parcela_total")
            p, c, hit = classify_with_parcelas(desc, vl, pid, pa, pt, regras_parc, rules_live, default_person, default_cat)
            pessoas.append(p)
            cats.append(c)
            hits.append(hit)

        df_out = df.copy()
        df_out["pessoa"] = pessoas
        df_out["categoria"] = cats
        df_out["regra"] = hits

        # Prévia
        st.subheader("Prévia")
        ordered_cols = [
            "data", "descricao", "desc_base", "valor",
            "parcela_txt", "parcela_atual", "parcela_total", "id_parcelamento",
            "pessoa", "categoria", "regra"
        ]
        df_out = df_out[ordered_cols]
        st.dataframe(df_out, use_container_width=True, height=380)

        # --- Ensinar parcelamento ---
        st.markdown("### Ensinar parcelamento (aplica até a última parcela)")
        st.caption("Escolha um parcelado na lista abaixo, defina pessoa/categoria e salve. Ele vai aplicar em todas as próximas parcelas automaticamente.")

        df_parc = df_out[df_out["parcela_total"].notna()].copy()
        if df_parc.empty:
            st.info("Nenhum parcelado detectado nessa fatura.")
        else:
            # lista de parcelados encontrados
            df_parc["label"] = df_parc.apply(
                lambda r: f"{r['id_parcelamento']} | {r['desc_base'][:45]} | R$ {r['valor']:.2f} | {r['parcela_txt']}",
                axis=1
            )
            options = df_parc["label"].unique().tolist()
            sel = st.selectbox("Escolha o parcelamento", options=options)

            sel_id = sel.split(" | ")[0].strip()
            # sugestão inicial: pega primeira linha daquele id
            ex = df_parc[df_parc["id_parcelamento"] == sel_id].iloc[0]

            pessoa_sel = st.text_input("Pessoa (para este parcelamento)", value=str(ex["pessoa"]))
            cat_sel = st.text_input("Categoria (para este parcelamento)", value=str(ex["categoria"]))

            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                if st.button("✅ Salvar para este parcelamento"):
                    # salva regra automática
                    regras_parc[sel_id] = {
                        "pessoa": pessoa_sel,
                        "categoria": cat_sel,
                        "parcelas_total": int(ex["parcela_total"]) if pd.notna(ex["parcela_total"]) else None,
                        "concluido": False,
                        "desc_base": str(ex["desc_base"]),
                    }
                    save_parcelas_rules(regras_parc)
                    st.success("Parcelamento salvo! A partir daqui, vai cair automático até acabar.")
            with colB:
                if st.button("🧾 Marcar como concluído (arquivar)"):
                    if sel_id in regras_parc:
                        regras_parc[sel_id]["concluido"] = True
                        save_parcelas_rules(regras_parc)
                        st.success("Marcado como concluído. Não será aplicado automaticamente mais.")
            with colC:
                if st.button("🗑️ Remover regra deste parcelamento"):
                    if sel_id in regras_parc:
                        regras_parc.pop(sel_id, None)
                        save_parcelas_rules(regras_parc)
                        st.warning("Regra removida.")

        # Export
        excel_bytes = to_excel_bytes(df_out, regras_parc)
        st.download_button(
            "⬇️ Baixar Excel separado (com parcelas + parcelamentos)",
            data=excel_bytes,
            file_name="fatura_separada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------- TAB 3: Parcelamentos (auto) ----------
with tabs[2]:
    st.subheader("Regras automáticas de parcelamento (id_parcelamento)")

    if not regras_parc:
        st.info("Ainda não existe nenhum parcelamento salvo.")
    else:
        dfp = pd.DataFrame([
            {
                "id_parcelamento": k,
                "desc_base": v.get("desc_base", ""),
                "pessoa": v.get("pessoa", ""),
                "categoria": v.get("categoria", ""),
                "parcelas_total": v.get("parcelas_total", ""),
                "concluido": v.get("concluido", False),
            }
            for k, v in regras_parc.items()
        ]).sort_values(["concluido", "pessoa", "categoria", "id_parcelamento"], ascending=[True, True, True, True])

        st.dataframe(dfp, use_container_width=True, height=420)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Exportar regras de parcelamento (backup JSON)"):
                st.download_button(
                    "Baixar regras_parcelamento.json",
                    data=json.dumps(regras_parc, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="regras_parcelamento.json",
                    mime="application/json"
                )
        with c2:
            if st.button("🧹 Limpar TODAS as regras de parcelamento"):
                Path(PARCELAS_FILE).unlink(missing_ok=True)
                st.warning("Regras de parcelamento apagadas. Recarregue a página (F5).")