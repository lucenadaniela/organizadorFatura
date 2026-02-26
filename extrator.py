import io
import re
import hashlib
import unicodedata

import pandas as pd
import streamlit as st

# =========================
# PDF (Nubank texto selecionável)
# =========================
try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False

# =========================
# Supabase
# =========================
try:
    from supabase import create_client
    SUPABASE_OK = True
except Exception:
    SUPABASE_OK = False


APP_TITLE = "Organizador de fatura (com login + regras salvas)"

# Tabelas no Supabase (como estão hoje)
TBL_RULES = "regras_pagamentos"
TBL_PARC = "regras_parcelamento"


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

def categoria_from_tipo(tipo: str, default_cat: str) -> str:
    t = norm(tipo)
    if t == "fixo":
        return "Fixo"
    if t == "variavel":
        return "Variável"
    return default_cat


# =========================
# UI / CSS (dark estável)
# =========================
def inject_css():
    st.markdown("""
    <style>
      :root{
        --bg: #0b1220;
        --bg-soft: #111827;
        --panel: rgba(17, 24, 39, 0.88);
        --panel-2: rgba(31, 41, 55, 0.95);
        --text: #f9fafb;
        --muted: #cbd5e1;
        --border: rgba(255,255,255,0.10);
        --border-strong: rgba(255,255,255,0.16);
        --accent: #7c3aed;
        --accent-2: #6366f1;
        --shadow: 0 10px 30px rgba(0,0,0,0.35);
        --radius: 18px;
      }

      html, body, [data-testid="stAppViewContainer"]{
        background:
          radial-gradient(1200px 700px at 20% 10%, rgba(124,58,237,0.14), transparent 50%),
          radial-gradient(1000px 600px at 80% 30%, rgba(34,197,94,0.08), transparent 55%),
          var(--bg) !important;
        color: var(--text) !important;
      }

      [data-testid="stAppViewContainer"] *{
        color: inherit;
      }

      .block-container{
        max-width: 1200px;
        padding-top: 1.2rem !important;
        padding-bottom: 2.2rem !important;
      }

      h1,h2,h3,h4,h5,h6{
        color: var(--text) !important;
        letter-spacing: -0.02em;
      }

      p, label, span, small{
        color: var(--text) !important;
      }

      .muted{
        color: var(--muted) !important;
      }

      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(10,16,30,0.98)) !important;
        border-right: 1px solid var(--border) !important;
      }

      [data-testid="stSidebar"] *{
        color: var(--text) !important;
      }

      .card{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px;
        box-shadow: var(--shadow);
      }

      .stTextInput input,
      .stTextArea textarea,
      .stNumberInput input,
      input, textarea{
        background: var(--panel-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 12px !important;
      }

      .stTextInput input::placeholder,
      .stTextArea textarea::placeholder,
      .stNumberInput input::placeholder,
      input::placeholder,
      textarea::placeholder{
        color: #94a3b8 !important;
      }

      .stTextInput input:focus,
      .stTextArea textarea:focus,
      .stNumberInput input:focus,
      input:focus,
      textarea:focus{
        box-shadow: 0 0 0 3px rgba(124,58,237,0.22) !important;
        border-color: rgba(124,58,237,0.55) !important;
        outline: none !important;
      }

      [data-baseweb="select"] > div{
        background: var(--panel-2) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
      }

      [data-baseweb="popover"]{
        background: #111827 !important;
        color: var(--text) !important;
      }

      [data-testid="stFileUploaderDropzone"]{
        background: rgba(17,24,39,0.72) !important;
        border: 1px dashed rgba(255,255,255,0.18) !important;
        border-radius: var(--radius) !important;
        padding: 20px !important;
      }

      .stButton > button{
        background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 14px !important;
        padding: 0.72rem 1rem !important;
        width: 100%;
        box-shadow: var(--shadow);
      }

      .stButton > button:hover{
        filter: brightness(1.05);
      }

      button[kind="secondary"]{
        background: rgba(255,255,255,0.06) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        box-shadow: none !important;
      }

      [data-testid="stAlert"]{
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
      }

      details{
        background: rgba(17,24,39,0.42) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
      }

      [data-testid="stDataFrame"]{
        background: rgba(17,24,39,0.86) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
      }

      [data-testid="stDataFrame"] [role="grid"]{
        background: #111827 !important;
        color: var(--text) !important;
      }

      [data-testid="stDataFrame"] [role="row"]{
        background: #111827 !important;
        color: var(--text) !important;
      }

      [data-testid="stDataFrame"] [role="gridcell"],
      [data-testid="stDataFrame"] [role="columnheader"],
      [data-testid="stDataFrame"] [role="rowheader"]{
        background: #111827 !important;
        color: var(--text) !important;
        border-color: rgba(255,255,255,0.08) !important;
      }

      [data-testid="stDataFrame"] input,
      [data-testid="stDataFrame"] textarea{
        background: #1f2937 !important;
        color: var(--text) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
      }

      [data-testid="stDataFrame"] button{
        color: var(--text) !important;
      }

      table{
        background: #111827 !important;
        color: var(--text) !important;
      }

      th, td{
        border-color: rgba(255,255,255,0.08) !important;
      }

      .stCheckbox label, .stRadio label{
        color: var(--text) !important;
      }

      hr{
        border-color: rgba(255,255,255,0.08) !important;
      }

      @media (max-width: 900px){
        .block-container{
          padding-left: 1rem !important;
          padding-right: 1rem !important;
        }

        .stButton > button{
          padding: 0.85rem 1rem !important;
          border-radius: 16px !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input{
          font-size: 1rem !important;
        }
      }

      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# =========================
# Supabase helpers
# =========================
def supabase_client():
    if not SUPABASE_OK:
        st.error("Biblioteca do Supabase não instalada. Adicione `supabase` no requirements.txt.")
        st.stop()

    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")

    if not url or not key:
        st.error("Faltam SUPABASE_URL e SUPABASE_ANON_KEY em `.streamlit/secrets.toml` (ou Secrets do Streamlit Cloud).")
        st.stop()

    return create_client(url, key)

def get_uid():
    sess = st.session_state.get("auth_session")
    if not sess:
        return None

    user = getattr(sess, "user", None)
    if user and getattr(user, "id", None):
        return user.id

    try:
        return sess.get("user", {}).get("id")
    except Exception:
        return None


# =========================
# Login UI
# =========================
def login_ui(sb):
    if "auth_session" not in st.session_state:
        st.session_state["auth_session"] = None

    with st.sidebar:
        st.markdown("## 🔒 Conta")
        tab1, tab2 = st.tabs(["Entrar", "Criar conta"])

        with tab1:
            with st.form("login_form_sidebar"):
                email = st.text_input("Email", placeholder="seuemail@exemplo.com", key="login_email_sb")
                senha = st.text_input("Senha", type="password", key="login_pass_sb")
                ok = st.form_submit_button("Entrar")

            if ok:
                try:
                    res = sb.auth.sign_in_with_password({"email": email, "password": senha})
                    st.session_state["auth_session"] = res.session
                    st.success("Login feito ✅")
                    st.rerun()
                except Exception:
                    st.error("Falha no login. Confira email/senha.")

        with tab2:
            with st.form("signup_form_sidebar"):
                email2 = st.text_input("Email", placeholder="seuemail@exemplo.com", key="su_email_sb")
                senha2 = st.text_input("Senha", type="password", key="su_pass_sb")
                ok2 = st.form_submit_button("Criar conta")

            if ok2:
                try:
                    sb.auth.sign_up({"email": email2, "password": senha2})
                    st.success("Conta criada! Agora entre em 'Entrar' ✅")
                except Exception as e:
                    msg = str(e).lower()
                    if "rate limit" in msg:
                        st.error("Erro no cadastro: limite de emails atingido. Aguarde e tente de novo.")
                    else:
                        st.error("Erro no cadastro. Verifique email/senha.")

        if st.session_state["auth_session"]:
            if st.button("Sair", type="secondary"):
                try:
                    sb.auth.sign_out()
                except Exception:
                    pass
                st.session_state["auth_session"] = None
                st.rerun()

    if not st.session_state["auth_session"]:
        st.markdown(f"## {APP_TITLE}")
        st.markdown('<p class="muted">Entre para acessar suas regras e parcelamentos salvos.</p>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Entrar")
        with st.form("login_form_center"):
            email = st.text_input("Email", placeholder="seuemail@exemplo.com", key="login_email_ct")
            senha = st.text_input("Senha", type="password", key="login_pass_ct")
            ok = st.form_submit_button("Entrar")
        st.markdown("</div>", unsafe_allow_html=True)

        if ok:
            try:
                res = sb.auth.sign_in_with_password({"email": email, "password": senha})
                st.session_state["auth_session"] = res.session
                st.success("Login feito ✅")
                st.rerun()
            except Exception:
                st.error("Falha no login. Confira email/senha.")

        st.stop()


# =========================
# Regras (Supabase)
# Tabela: user_id, tipo, palavra_chave, valor
# =========================
def rules_default_df():
    return pd.DataFrame([
        {"tipo": "variavel", "palavra_chave": "uber", "valor": ""},
        {"tipo": "variavel", "palavra_chave": "99app", "valor": ""},
        {"tipo": "variavel", "palavra_chave": "99", "valor": ""},
    ])

def load_rules_sb(sb, uid: str) -> pd.DataFrame:
    try:
        res = sb.table(TBL_RULES).select("tipo,palavra_chave,valor").eq("user_id", uid).execute()
        data = res.data or []

        if not data:
            return rules_default_df()

        df = pd.DataFrame(data).fillna("")
        for c in ["tipo", "palavra_chave", "valor"]:
            if c not in df.columns:
                df[c] = ""

        return df[["tipo", "palavra_chave", "valor"]].fillna("")
    except Exception as e:
        st.error(f"Erro ao carregar regras do Supabase: {e}")
        st.stop()

def save_rules_sb(sb, uid: str, df: pd.DataFrame):
    df = df.fillna("")
    rows = df.to_dict(orient="records")

    try:
        sb.table(TBL_RULES).delete().eq("user_id", uid).execute()

        if rows:
            payload = []
            for r in rows:
                payload.append({
                    "user_id": uid,
                    "tipo": r.get("tipo", ""),
                    "palavra_chave": r.get("palavra_chave", ""),
                    "valor": r.get("valor", ""),
                })
            sb.table(TBL_RULES).insert(payload).execute()
    except Exception as e:
        st.error(f"Erro ao salvar regras no Supabase: {e}")
        st.stop()


# =========================
# Parcelamentos (Supabase)
# Tabela: user_id, id_parcelamento, pessoa, categoria
# =========================
def load_parc_sb(sb, uid: str) -> dict:
    try:
        res = sb.table(TBL_PARC).select("id_parcelamento,pessoa,categoria").eq("user_id", uid).execute()
        data = res.data or []

        out = {}
        for r in data:
            out[str(r.get("id_parcelamento"))] = {
                "pessoa": r.get("pessoa") or "",
                "categoria": r.get("categoria") or "",
            }
        return out
    except Exception as e:
        st.error(f"Erro ao carregar parcelamentos do Supabase: {e}")
        st.stop()

def upsert_parc_sb(sb, uid: str, pid: str, pessoa: str, categoria: str):
    payload = {
        "user_id": uid,
        "id_parcelamento": pid,
        "pessoa": pessoa or "",
        "categoria": categoria or "",
    }

    try:
        sb.table(TBL_PARC).delete().eq("user_id", uid).eq("id_parcelamento", pid).execute()
        sb.table(TBL_PARC).insert(payload).execute()
    except Exception as e:
        st.error(f"Erro ao salvar parcelamento: {e}")
        st.stop()

def delete_parc_sb(sb, uid: str, pid: str):
    try:
        sb.table(TBL_PARC).delete().eq("user_id", uid).eq("id_parcelamento", pid).execute()
    except Exception as e:
        st.error(f"Erro ao remover parcelamento: {e}")
        st.stop()


# =========================
# Parcelas (detecção)
# =========================
RE_PARCELA = re.compile(
    r"(?:parcela\s*)?(?P<atual>\d{1,2})\s*(?:/|de)\s*(?P<total>\d{1,2})",
    re.IGNORECASE
)

def extrair_parcela(desc: str):
    m = RE_PARCELA.search(desc or "")
    if not m:
        return "", None, None
    return (
        f"{int(m.group('atual'))}/{int(m.group('total'))}",
        int(m.group("atual")),
        int(m.group("total")),
    )

def remover_texto_parcela(desc: str) -> str:
    return RE_PARCELA.sub("", desc or "").strip()

def gerar_id_parcelamento(desc: str, valor: float):
    base = f"{remover_texto_parcela(desc)}|{float(valor):.2f}".lower()
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:10]


# =========================
# Classificação
# 1) parcelamento salvo
# 2) regra manual (tipo + palavra + valor)
# 3) fallback
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

    rules = rules.reset_index().rename(columns={"index": "__ordem"})
    rules = rules.sort_values(
        ["tem_valor", "kw_len", "__ordem"],
        ascending=[False, False, True],
        kind="mergesort"
    )

    for _, r in rules.iterrows():
        kw = r["kw_norm"]
        kw_comp = r["kw_comp"]

        if not kw and not kw_comp:
            continue

        if (kw and kw not in d) and (kw_comp and kw_comp not in d_comp):
            continue

        vr = r["valor_float"]
        if vr is not None and not valor_bate(valor_lanc, vr, tol=0.01):
            continue

        pessoa = default_person
        categoria = categoria_from_tipo(r.get("tipo", ""), default_cat)
        return pessoa, categoria, "manual"

    return default_person, default_cat, "fallback"

def classify(desc: str, valor_lanc: float, id_parc: str, parc_rules: dict, rules_df: pd.DataFrame, default_person: str, default_cat: str):
    if id_parc and id_parc in parc_rules:
        pr = parc_rules[id_parc]
        pessoa = (pr.get("pessoa", "") or "").strip() or default_person
        categoria = (pr.get("categoria", "") or "").strip() or default_cat
        return pessoa, categoria, "parcelamento"

    return classify_manual(desc, valor_lanc, rules_df, default_person, default_cat)


# =========================
# Parser PDF Nubank
# =========================
LINHA_RE = re.compile(r"^(?P<dia>\d{2})\s(?P<mes>[A-Z]{3})\s(?P<desc>.+?)\sR\$\s(?P<valor>[\d\.,]+)$")
MESES = {
    "JAN": "01", "FEV": "02", "MAR": "03", "ABR": "04",
    "MAI": "05", "JUN": "06", "JUL": "07", "AGO": "08",
    "SET": "09", "OUT": "10", "NOV": "11", "DEZ": "12"
}

def parse_nubank_pdf(file_bytes: bytes, ano: int):
    if not PDF_OK:
        raise RuntimeError("pdfplumber não está instalado (confira requirements.txt).")

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

                rows.append({
                    "data": data,
                    "descricao": desc,
                    "valor": valor
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date.astype(str)

    return df


# =========================
# APP
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_css()

sb = supabase_client()
login_ui(sb)

uid = get_uid()
if not uid:
    st.error("Não consegui pegar seu user_id (sessão). Tenta sair e entrar de novo.")
    st.stop()

rules_df = load_rules_sb(sb, uid)
parc_rules = load_parc_sb(sb, uid)

st.markdown(f"## {APP_TITLE}")

with st.expander("⚙️ Configurações", expanded=False):
    default_person = st.text_input("Pessoa padrão (se não casar regra)", value="Pendente")
    default_cat = st.text_input("Categoria padrão (se não casar regra)", value="Revisar")
    ano = st.number_input("Ano da fatura (PDF)", min_value=2020, max_value=2100, value=2026, step=1)

    colA, colB, colC = st.columns(3)
    with colA:
        mostrar_categoria = st.checkbox("Resumo por categoria", value=True)
    with colB:
        mostrar_pendencias = st.checkbox("Pendências (editável)", value=True)
    with colC:
        mostrar_detalhes = st.checkbox("Detalhes", value=False)

up = st.file_uploader(
    "Envie o PDF do Nubank (texto selecionável) ou CSV (data, descricao, valor).",
    type=["pdf", "csv"]
)

# =========================
# REGRAS editor
# =========================
with st.expander("🧠 Regras (editar/cadastrar) — fica salvo no seu login", expanded=False):
    st.caption("Dica: deixe o campo VALOR vazio pra regra valer pra qualquer valor.")
    st.caption("A categoria é definida pelo TIPO: fixo = Fixo | variavel = Variável | outros = categoria padrão.")

    rules_edited = st.data_editor(
        rules_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "tipo": st.column_config.SelectboxColumn("tipo", options=["fixo", "variavel", "outros"]),
            "palavra_chave": st.column_config.TextColumn("palavra_chave"),
            "valor": st.column_config.TextColumn("valor", help="Opcional. Ex.: 136,50"),
        },
        key="rules_editor"
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("💾 Salvar regras"):
            save_rules_sb(sb, uid, rules_edited)
            st.success("Regras salvas ✅")
            st.rerun()

    with c2:
        if st.button("♻️ Resetar para defaults", type="secondary"):
            save_rules_sb(sb, uid, rules_default_df())
            st.warning("Regras resetadas para defaults.")
            st.rerun()


# =========================
# PROCESSAMENTO
# =========================
if not up:
    st.info("Suba uma fatura para ver o resumo.")
    st.stop()

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

# Enriquecer parcelas
parc = df["descricao"].astype(str).apply(extrair_parcela)
df["parcela_txt"] = parc.apply(lambda x: x[0])
df["parcela_atual"] = parc.apply(lambda x: x[1])
df["parcela_total"] = parc.apply(lambda x: x[2])
df["desc_base"] = df["descricao"].astype(str).apply(remover_texto_parcela)
df["id_parcelamento"] = df.apply(lambda r: gerar_id_parcelamento(r["descricao"], r["valor"]), axis=1)

# Classificar
rules_live = rules_edited.fillna("") if "rules_edited" in locals() else rules_df.fillna("")

pessoas, cats, fontes = [], [], []

for _, r in df.iterrows():
    p, c, f = classify(
        str(r["descricao"]),
        float(r["valor"]),
        str(r["id_parcelamento"]),
        parc_rules,
        rules_live,
        default_person,
        default_cat
    )
    pessoas.append(p)
    cats.append(c)
    fontes.append(f)

df["pessoa"] = pessoas
df["categoria"] = cats
df["fonte_regra"] = fontes


# =========================
# RESUMO PRINCIPAL
# =========================
st.divider()
total_geral = float(df["valor"].sum())
st.metric("Total geral", brl(total_geral))

def render_totais_por_pessoa(df_):
    totais = (
        df_.groupby("pessoa", as_index=False)["valor"]
        .sum()
        .sort_values("valor", ascending=False)
    )

    st.subheader("Totais por pessoa")

    cols = st.columns(2 if len(totais) > 1 else 1)
    for i, row in enumerate(totais.itertuples(index=False)):
        cols[i % len(cols)].metric(str(row.pessoa), brl(float(row.valor)))

render_totais_por_pessoa(df)

if mostrar_categoria:
    st.subheader("Resumo por categoria")
    resumo_cat = (
        df.pivot_table(
            index="pessoa",
            columns="categoria",
            values="valor",
            aggfunc="sum",
            fill_value=0
        )
        .sort_index()
    )
    st.dataframe(resumo_cat, use_container_width=True)


# =========================
# PENDÊNCIAS EDITÁVEIS
# =========================
if mostrar_pendencias:
    st.subheader("Pendências (você pode editar pessoa/categoria aqui)")

    pend = df[
        (df["pessoa"].str.lower() == "pendente") |
        (df["categoria"].str.lower() == "revisar")
    ].copy()

    if pend.empty:
        st.success("Nada pendente 🎯")
    else:
        pend_view = pend[["data", "descricao", "valor", "pessoa", "categoria"]].copy()

        pend_edit = st.data_editor(
            pend_view,
            use_container_width=True,
            num_rows="fixed",
            key="pend_editor"
        )

        cA, cB = st.columns(2)

        with cA:
            if st.button("✅ Aplicar edições (só nesta tela)"):
                for _, row in pend_edit.iterrows():
                    mask = (
                        (df["data"] == row["data"]) &
                        (df["descricao"] == row["descricao"]) &
                        (df["valor"] == row["valor"])
                    )
                    df.loc[mask, "pessoa"] = row["pessoa"]
                    df.loc[mask, "categoria"] = row["categoria"]
                    df.loc[mask, "fonte_regra"] = "manual_tela"

                st.success("Aplicado! Totais atualizados abaixo ✅")
                render_totais_por_pessoa(df)

        with cB:
            st.caption("Se quiser automatizar pro futuro, crie uma regra em 🧠 Regras.")


# =========================
# PARCELAMENTOS
# =========================
with st.expander("Parcelamentos", expanded=False):
    df_parc = df[df["parcela_total"].notna()].copy()

    if df_parc.empty:
        st.info("Nenhum parcelado detectado nesta fatura.")
    else:
        df_parc["label"] = df_parc.apply(
            lambda r: f"{r['id_parcelamento']} | {str(r['desc_base'])[:45]} | {brl(float(r['valor']))} | {r['parcela_txt']}",
            axis=1
        )

        sel = st.selectbox("Escolha o parcelamento", options=df_parc["label"].unique().tolist())
        sel_id = sel.split(" | ")[0].strip()

        ex = df_parc[df_parc["id_parcelamento"] == sel_id].iloc[0]

        pessoa_sel = st.text_input("Pessoa", value=str(ex["pessoa"]))
        cat_sel = st.text_input("Categoria", value=str(ex["categoria"]))

        c1, c2 = st.columns(2)

        with c1:
            if st.button("💾 Salvar parcelamento"):
                upsert_parc_sb(
                    sb,
                    uid,
                    pid=sel_id,
                    pessoa=pessoa_sel,
                    categoria=cat_sel,
                )
                st.success("Parcelamento salvo ✅")
                st.rerun()

        with c2:
            if st.button("🗑️ Remover parcelamento", type="secondary"):
                delete_parc_sb(sb, uid, sel_id)
                st.warning("Parcelamento removido.")
                st.rerun()


# =========================
# DETALHES
# =========================
if mostrar_detalhes:
    with st.expander("🧾 Detalhes (lançamentos)", expanded=False):
        st.dataframe(
            df[[
                "data",
                "descricao",
                "valor",
                "pessoa",
                "categoria",
                "fonte_regra",
                "parcela_txt",
                "id_parcelamento"
            ]],
            use_container_width=True,
            height=420
        )
