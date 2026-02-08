import io
import re
import json
import hashlib
import unicodedata
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ========= PDF =========
try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False

# ========= SUPABASE =========
from supabase import create_client

APP_TITLE = "Organizador de fatura (com login + regras salvas)"

# =========================
# Helpers gerais
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
    s = f"{float(v):,.2f}"
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
# Supabase client + sessão
# =========================
@st.cache_resource
def get_supabase_base():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

def get_authed_supabase():
    sb = get_supabase_base()
    sess = st.session_state.get("sb_session")
    if sess and sess.get("access_token"):
        sb.postgrest.auth(sess["access_token"])
    return sb

def is_logged_in() -> bool:
    sess = st.session_state.get("sb_session")
    return bool(sess and sess.get("user_id") and sess.get("access_token"))

def auth_ui():
    st.subheader("🔐 Entrar / Criar conta")

    tab1, tab2 = st.tabs(["Entrar", "Criar conta"])

    sb = get_supabase_base()

    with tab1:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="seuemail@exemplo.com")
            password = st.text_input("Senha", type="password")
            ok = st.form_submit_button("Entrar")
        if ok:
            try:
                res = sb.auth.sign_in_with_password({"email": email, "password": password})
                # res.session / res.user
                session = res.session
                user = res.user
                st.session_state["sb_session"] = {
                    "access_token": session.access_token,
                    "refresh_token": session.refresh_token,
                    "user_id": user.id,
                    "email": user.email,
                }
                st.success("Logada ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Erro no login: {e}")

    with tab2:
        with st.form("signup_form", clear_on_submit=False):
            email = st.text_input("Email (cadastro)", placeholder="seuemail@exemplo.com")
            password = st.text_input("Senha (cadastro)", type="password")
            ok = st.form_submit_button("Criar conta")
        if ok:
            try:
                sb.auth.sign_up({"email": email, "password": password})
                st.success("Conta criada! Agora entra na aba **Entrar** ✅")
            except Exception as e:
                st.error(f"Erro no cadastro: {e}")

def logout_ui():
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Sair"):
            # não precisa chamar sign_out (token expira), mas pode:
            try:
                get_supabase_base().auth.sign_out()
            except Exception:
                pass
            st.session_state.pop("sb_session", None)
            st.rerun()

    with col2:
        sess = st.session_state.get("sb_session", {})
        st.caption(f"Logada como: {sess.get('email','')}")

# =========================
# Defaults de regras (seed)
# =========================
def default_rules_df() -> pd.DataFrame:
    # Uber e 99 ficam como regras sem valor (funciona pra qualquer valor)
    # A pessoa/categoria você ajusta no editor.
    return pd.DataFrame([
        # FIXOS
        {"tipo": "fixo", "palavra_chave": "spotify",        "valor": "40,9",  "pessoa": "Yves", "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "netflix",        "valor": "20,9",  "pessoa": "Yves", "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "amazon prime",   "valor": "19,9",  "pessoa": "Yves", "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "apple.com/bill", "valor": "19,9",  "pessoa": "Yves", "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "academia",       "valor": "149,9", "pessoa": "Yves", "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "academia",       "valor": "133",   "pessoa": "Yves", "categoria": "Fixos"},
        {"tipo": "fixo", "palavra_chave": "nucel",          "valor": "30",    "pessoa": "Yves", "categoria": "Fixos"},

        # TRANSPORTE (sem valor => pega tudo daquela palavra)
        {"tipo": "variavel", "palavra_chave": "uber",       "valor": "", "pessoa": "", "categoria": "Transporte"},
        {"tipo": "variavel", "palavra_chave": "99app",      "valor": "", "pessoa": "", "categoria": "Transporte"},
        {"tipo": "variavel", "palavra_chave": "99",         "valor": "", "pessoa": "", "categoria": "Transporte"},

        # EXEMPLOS (você pode remover/alterar)
        {"tipo": "variavel", "palavra_chave": "shein",         "valor": "136,50", "pessoa": "Maria", "categoria": "Roupas"},
        {"tipo": "variavel", "palavra_chave": "mercado livre", "valor": "55,87",  "pessoa": "Maria", "categoria": "Compras"},
    ]).fillna("")

# =========================
# DB: regras_pagamentos
# =========================
def load_rules_db(user_id: str) -> pd.DataFrame:
    sb = get_authed_supabase()
    resp = sb.table("regras_pagamentos") \
        .select("id,user_id,tipo,palavra_chave,valor,pessoa,categoria") \
        .eq("user_id", user_id) \
        .order("id") \
        .execute()
    data = resp.data or []
    if not data:
        # seed
        seed = default_rules_df()
        payload = []
        for r in seed.to_dict(orient="records"):
            r["user_id"] = user_id
            payload.append(r)
        sb.table("regras_pagamentos").insert(payload).execute()
        return seed
    df = pd.DataFrame(data)
    # remove colunas técnicas do editor (mantemos id interno só pra debug; pode ocultar)
    return df.fillna("")

def save_rules_db(user_id: str, df: pd.DataFrame):
    sb = get_authed_supabase()
    df = df.fillna("")
    # vamos salvar só as colunas de regra (id é gerado pelo banco)
    keep = ["tipo", "palavra_chave", "valor", "pessoa", "categoria"]
    out = df[keep].copy()

    # apaga todas do user e insere novamente (simples e funciona bem)
    sb.table("regras_pagamentos").delete().eq("user_id", user_id).execute()

    payload = []
    for r in out.to_dict(orient="records"):
        r["user_id"] = user_id
        payload.append(r)

    if payload:
        sb.table("regras_pagamentos").insert(payload).execute()

# =========================
# DB: regras_parcelamento
# =========================
def load_parcelas_db(user_id: str) -> dict:
    sb = get_authed_supabase()
    resp = sb.table("regras_parcelamento") \
        .select("id_parcelamento,user_id,pessoa,categoria,parcelas_total,desc_base,ativo") \
        .eq("user_id", user_id) \
        .execute()
    data = resp.data or []
    d = {}
    for r in data:
        d[r["id_parcelamento"]] = {
            "pessoa": r.get("pessoa"),
            "categoria": r.get("categoria"),
            "parcelas_total": r.get("parcelas_total"),
            "desc_base": r.get("desc_base"),
            "ativo": r.get("ativo", True),
        }
    return d

def upsert_parcela_db(user_id: str, id_parc: str, pessoa: str, categoria: str, parcelas_total: int, desc_base: str):
    sb = get_authed_supabase()
    record = {
        "id_parcelamento": id_parc,
        "user_id": user_id,
        "pessoa": pessoa,
        "categoria": categoria,
        "parcelas_total": parcelas_total,
        "desc_base": desc_base,
        "ativo": True,
    }
    sb.table("regras_parcelamento").upsert(record, on_conflict="id_parcelamento").execute()

def delete_parcela_db(user_id: str, id_parc: str):
    sb = get_authed_supabase()
    sb.table("regras_parcelamento").delete().eq("user_id", user_id).eq("id_parcelamento", id_parc).execute()

# =========================
# Parcelas: extração + id
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
    rules["kw_len"] = rules["kw_norm"].apply(len)
    rules["kw_comp"] = rules["palavra_chave"].astype(str).apply(norm_compact)
    rules["valor_float"] = rules["valor"].apply(parse_valor_regra)
    rules["tem_valor"] = rules["valor_float"].apply(lambda v: 1 if v is not None else 0)

    # prioridade: regra com valor > keyword maior > ordem
    rules = rules.reset_index().rename(columns={"index": "__ordem"})
    rules = rules.sort_values(
        ["tem_valor", "kw_len", "__ordem"],
        ascending=[False, False, True],
        kind="mergesort"
    )

    for _, r in rules.iterrows():
        kw = r["kw_norm"]
        kwc = r["kw_comp"]
        if not kw:
            continue

        if (kw not in d) and (kwc not in d_comp):
            continue

        vr = r["valor_float"]
        # se regra tem valor, exige bater
        if vr is not None and not valor_bate(valor_lanc, vr, tol=0.01):
            continue

        pessoa = (r.get("pessoa") or "").strip() or default_person
        categoria = (r.get("categoria") or "").strip() or default_cat
        return pessoa, categoria, "manual"

    return default_person, default_cat, "fallback"

def classify(desc: str, valor: float, id_parc: str, regras_parc: dict, rules_df: pd.DataFrame, default_person: str, default_cat: str):
    # 1) se existe regra de parcelamento salva, usa ela
    if id_parc and id_parc in regras_parc and regras_parc[id_parc].get("ativo", True):
        r = regras_parc[id_parc]
        pessoa = (r.get("pessoa") or "").strip() or default_person
        cat = (r.get("categoria") or "").strip() or default_cat
        return pessoa, cat, "parcelamento"

    # 2) senão, tenta regras manuais
    return (*classify_manual(desc, valor, rules_df, default_person, default_cat),)

# =========================
# Parser PDF Nubank (texto selecionável)
# =========================
LINHA_RE = re.compile(r"^(?P<dia>\d{2})\s(?P<mes>[A-Z]{3})\s(?P<desc>.+?)\sR\$\s(?P<valor>[\d\.,]+)$")
MESES = {"JAN":"01","FEV":"02","MAR":"03","ABR":"04","MAI":"05","JUN":"06","JUL":"07","AGO":"08","SET":"09","OUT":"10","NOV":"11","DEZ":"12"}

def parse_nubank_pdf(file_bytes: bytes, ano: int):
    if not PDF_OK:
        raise RuntimeError("pdfplumber não está instalado (ver requirements.txt).")

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
# App UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# sidebar: login
with st.sidebar:
    st.header("Conta")
    if not is_logged_in():
        auth_ui()
        st.stop()
    else:
        logout_ui()

sess = st.session_state["sb_session"]
USER_ID = sess["user_id"]

# carrega regras do usuário
rules_df = load_rules_db(USER_ID)
regras_parc = load_parcelas_db(USER_ID)

# configurações básicas (padrões)
with st.expander("⚙️ Configurações", expanded=False):
    default_person = st.text_input("Pessoa padrão (se não casar regra)", value="Pendente")
    default_cat = st.text_input("Categoria padrão (se não casar regra)", value="Revisar")
    ano = st.number_input("Ano da fatura (PDF)", min_value=2020, max_value=2100, value=int(datetime.now().year), step=1)

# editor de regras
with st.expander("🧠 Regras (salvas por usuário)", expanded=False):
    st.caption("Dica: deixe **valor vazio** quando quiser que a regra valha pra qualquer valor daquela palavra-chave.")
    edited = st.data_editor(
        rules_df[["tipo","palavra_chave","valor","pessoa","categoria"]].copy(),
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
            try:
                save_rules_db(USER_ID, edited)
                st.success("Regras salvas no Supabase ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Erro salvando regras: {e}")
    with c2:
        if st.button("↩️ Restaurar defaults"):
            try:
                save_rules_db(USER_ID, default_rules_df())
                st.success("Defaults restaurados ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Erro restaurando: {e}")

# upload
up = st.file_uploader("Envie PDF do Nubank (texto selecionável) ou CSV (data, descricao, valor)", type=["pdf", "csv"])

if not up:
    st.info("Suba uma fatura para ver o resumo.")
    st.stop()

# carrega dataframe
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

# enriquecer parcelas + id
parc = df["descricao"].astype(str).apply(extrair_parcela)
df["parcela_txt"] = parc.apply(lambda x: x[0])
df["parcela_atual"] = parc.apply(lambda x: x[1])
df["parcela_total"] = parc.apply(lambda x: x[2])
df["desc_base"] = df["descricao"].astype(str).apply(remover_texto_parcela)
df["id_parcelamento"] = df.apply(lambda r: gerar_id_parcelamento(r["descricao"], r["valor"]), axis=1)

# classificar
rules_live = edited.fillna("") if "edited" in locals() else rules_df.fillna("")
pessoas, cats, fonte = [], [], []
for _, r in df.iterrows():
    p, c, f = classify(str(r["descricao"]), float(r["valor"]), r["id_parcelamento"], regras_parc, rules_live, default_person, default_cat)
    pessoas.append(p); cats.append(c); fonte.append(f)

df["pessoa"] = pessoas
df["categoria"] = cats
df["fonte_regra"] = fonte

# =========================
# RESUMO
# =========================
st.divider()
total_geral = float(df["valor"].sum())
st.metric("Total geral", brl(total_geral))

# Totais por pessoa
totais = df.groupby("pessoa", as_index=False)["valor"].sum().sort_values("valor", ascending=False)
st.subheader("Totais por pessoa")
cols = st.columns(min(3, max(1, len(totais))))
for i, row in enumerate(totais.itertuples(index=False)):
    cols[i % len(cols)].metric(str(row.pessoa), brl(float(row.valor)))

# Cards Uber/99 (somatório + quantidade) — entra nos totais porque já vira pessoa via regra
st.subheader("Transporte (somatório e quantidade)")
def is_uber(desc): 
    d = norm_compact(desc)
    return ("uber" in d)

def is_99(desc):
    d = norm_compact(desc)
    return ("99app" in d) or (d.endswith("99")) or (" 99 " in (" " + norm(desc) + " "))

uber_df = df[df["descricao"].astype(str).apply(is_uber)]
n99_df = df[df["descricao"].astype(str).apply(is_99)]

c1, c2 = st.columns(2)
c1.metric("Uber", brl(uber_df["valor"].sum()), f"Qtd: {len(uber_df)}")
c2.metric("99", brl(n99_df["valor"].sum()), f"Qtd: {len(n99_df)}")

# Resumo por categoria
st.subheader("Resumo por categoria")
resumo_cat = (
    df.pivot_table(index="pessoa", columns="categoria", values="valor", aggfunc="sum", fill_value=0)
    .sort_index()
)
st.dataframe(resumo_cat, use_container_width=True)

# Pendências
st.subheader("Pendências")
pend = df[(df["pessoa"].str.lower() == "pendente") | (df["categoria"].str.lower() == "revisar")].copy()
if pend.empty:
    st.success("Nada pendente 🎯")
else:
    st.dataframe(pend[["data", "descricao", "valor", "pessoa", "categoria", "fonte_regra"]], use_container_width=True, height=280)

# =========================
# Ensinar parcelamentos (salva no Supabase por usuário)
# =========================
with st.expander("📌 Ensinar parcelamentos (fica salvo no seu usuário)", expanded=False):
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
        pessoa_sel = st.text_input("Pessoa", value=str(ex["pessoa"]))
        cat_sel = st.text_input("Categoria", value=str(ex["categoria"]))

        cA, cB = st.columns(2)
        with cA:
            if st.button("✅ Salvar parcelamento"):
                try:
                    upsert_parcela_db(
                        USER_ID,
                        sel_id,
                        pessoa_sel,
                        cat_sel,
                        int(ex["parcela_total"]) if pd.notna(ex["parcela_total"]) else None,
                        str(ex["desc_base"]),
                    )
                    st.success("Parcelamento salvo no Supabase ✅ (não some mais)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro salvando parcelamento: {e}")

        with cB:
            if st.button("🗑️ Remover regra do parcelamento"):
                try:
                    delete_parcela_db(USER_ID, sel_id)
                    st.warning("Regra removida.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro removendo: {e}")

with st.expander("🧾 Detalhes (lançamentos)", expanded=False):
    st.dataframe(
        df[["data","descricao","valor","pessoa","categoria","fonte_regra","parcela_txt","id_parcelamento"]],
        use_container_width=True,
        height=420
    )
