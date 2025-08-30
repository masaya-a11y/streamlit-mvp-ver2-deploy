
import os, json, pandas as pd
import streamlit as st
from utils import (
    load_focus_keywords, load_loc_dict, enrich_reviews, kpi_summary, export_excel,
    DEFAULT_SENTIMENT_PROMPT, guess_mapping, apply_column_mapping
)

st.set_page_config(page_title="新穂高ロープウェイ 口コミ分析MVP", layout="wide")

st.sidebar.title("新穂高ロープウェイ 口コミ分析MVP")
st.sidebar.caption("MVP v0.2（Excel/CSV + マッピング + LLM感情）")
data_dir = st.sidebar.text_input("📁 データフォルダ", value="data")

translation_mode = st.sidebar.selectbox("🔤 翻訳モード", ["stub","pass"], help="本番はAPIに差し替え")
pos_th = st.sidebar.number_input("😀 ポジ判定しきい値（-10〜+10）", value=2.0, min_value=-10.0, max_value=10.0, step=0.5)
neg_th = st.sidebar.number_input("☹ ネガ判定しきい値（-10〜+10）", value=-2.0, min_value=-10.0, max_value=10.0, step=0.5)
low_thr = st.sidebar.number_input("⚠ 低評価（-1〜+1）", value=-0.2, min_value=-1.0, max_value=1.0, step=0.05)

engine = st.sidebar.selectbox("🧠 感情エンジン", ["ルール（デモ）","LLM（プロンプト）"])
llm_model = st.sidebar.text_input("LLM モデル（OpenAI）", value="gpt-4o-mini")
with st.sidebar.expander("LLM プロンプト（編集可）", expanded=False):
    llm_prompt = st.text_area("プロンプト", value=DEFAULT_SENTIMENT_PROMPT, height=220)
st.sidebar.caption("※ LLM利用には OPENAI_API_KEY を環境変数で設定してください。")

tabs = st.tabs(["① データ更新", "② ダッシュボード", "③ 辞書編集", "④ 出力"])

# ---------- ① データ更新 ----------
with tabs[0]:
    st.header("① データ更新")
    st.write("媒体 **ファイル**（**Excel推奨**・CSVも可）をアップロードし、**列マッピング**と辞書を使って前処理します。")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Excel（.xlsx/.xls）または CSV（.csv）をアップロード", type=["xlsx","xls","csv"])
    with col2:
        existing = []
        try:
            for fn in os.listdir(data_dir):
                lf = fn.lower()
                if lf.endswith((".xlsx",".xls",".csv")):
                    existing.append(fn)
        except FileNotFoundError:
            os.makedirs(data_dir, exist_ok=True)
        pick = st.selectbox("📄 既存ファイルを選択（Excel/CSV）", options=["(選択なし)"] + existing)

    fk_path = os.path.join(data_dir, "focus_keywords.csv")
    lc_path = os.path.join(data_dir, "location_to_country.csv")
    focus_df = load_focus_keywords(fk_path)
    loc_df = load_loc_dict(lc_path)

    df_raw = None
    selected_sheet = None
    excel_file_bytes = None
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith((".xlsx",".xls")):
            excel_file_bytes = uploaded.read()
            try:
                xls = pd.ExcelFile(excel_file_bytes)
                sheets = xls.sheet_names
                selected_sheet = st.selectbox("🧾 シートを選択（Excel選択時）", sheets, index=0)
                df_raw = pd.read_excel(pd.ExcelFile(excel_file_bytes), sheet_name=selected_sheet)
            except Exception as e:
                st.error(f"Excelを開けませんでした: {e}")
        else:
            df_raw = pd.read_csv(uploaded)
    elif pick != "(選択なし)":
        path = os.path.join(data_dir, pick)
        try:
            if path.lower().endswith(".csv"):
                df_raw = pd.read_csv(path)
            else:
                xls = pd.ExcelFile(path)
                sheets = xls.sheet_names
                default_sheet = 0
                for prefer in ["reviews", "examples", "template_blank"]:
                    if prefer in sheets:
                        default_sheet = sheets.index(prefer)
                        break
                selected_sheet = st.selectbox("🧾 シートを選択（Excel選択時）", sheets, index=default_sheet)
                df_raw = pd.read_excel(path, sheet_name=selected_sheet)
        except Exception as e:
            st.error(f"読み込みに失敗しました: {e}")

    st.subheader("🧭 列マッピング")
    if df_raw is None:
        st.info("ファイルを選択すると、ここにマッピングUIが表示されます。")
    else:
        file_cols = df_raw.columns.tolist()
        st.caption(f"ファイル内の列数: {len(file_cols)} 列")
        default_map = guess_mapping(file_cols)

        map_path = os.path.join(data_dir, "mappings.json")
        saved_maps = {}
        if os.path.exists(map_path):
            try:
                saved_maps = json.load(open(map_path, "r", encoding="utf-8"))
            except Exception:
                saved_maps = {}
        map_key = (pick if pick != "(選択なし)" else (uploaded.name if uploaded else "default"))
        if selected_sheet:
            map_key = f"{map_key}::{selected_sheet}"
        current_map = saved_maps.get(map_key, default_map)

        def map_row(label, key):
            cols = st.columns([2,2,1])
            with cols[0]:
                mode = st.selectbox(f"{label} の取得方法", ["列から","固定値"], index=0 if current_map.get(key,{}).get("mode","column")=="column" else 1, key=f"mode_{key}")
            with cols[1]:
                if mode == "列から":
                    init = current_map.get(key,{}).get("value","")
                    choices = ["(なし)"] + file_cols
                    idx = choices.index(init) if init in file_cols else 0
                    val = st.selectbox(f"{label} に使う列", choices, index=idx, key=f"col_{key}")
                else:
                    init = current_map.get(key,{}).get("value","")
                    val = st.text_input(f"{label} の固定値", value=init, key=f"fix_{key}")
            if mode == "列から":
                current_map[key] = {"mode":"column","value": (val if val!="(なし)" else "")}
            else:
                current_map[key] = {"mode":"fixed","value": val}

        st.markdown("**必須列**")
        map_row("媒体（source）", "source")
        map_row("レビューID（review_id）", "review_id")
        map_row("投稿日（posted_date, YYYY-MM-DD）", "posted_date")
        map_row("本文（body）", "body")

        with st.expander("任意列（ある場合はマッピング）"):
            for key, label in [
                ("visit_date","訪問日（visit_date）"),
                ("rating_original","媒体の星（rating_original）"),
                ("title","タイトル（title）"),
                ("language","原文言語（language）"),
                ("url","URL（url）"),
                ("author_name","投稿者名（author_name）"),
                ("author_location_raw","所在地テキスト（author_location_raw）"),
                ("author_country_code","出身国コード（author_country_code）"),
                ("companions","同行者（companions）"),
                ("tags","タグ（tags）"),
                ("media_count","写真/動画枚数（media_count）"),
            ]:
                map_row(label, key)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 マッピングを保存"):
                saved_maps[map_key] = current_map
                with open(map_path, "w", encoding="utf-8") as f:
                    json.dump(saved_maps, f, ensure_ascii=False, indent=2)
                st.success(f"保存しました: {map_key}")

        if st.button("🔄 前処理を実行", type="primary"):
            try:
                df_mapped = apply_column_mapping(df_raw, current_map)
            except Exception as e:
                st.error(f"マッピングエラー: {e}")
            else:
                enriched = enrich_reviews(df_mapped, focus_df, loc_df,
                                          translation_mode=translation_mode,
                                          pos_th=pos_th, neg_th=neg_th,
                                          engine=("llm" if engine=="LLM（プロンプト）" else "rule"),
                                          llm_prompt=llm_prompt, llm_model=llm_model, data_dir=data_dir)
                st.session_state["enriched"] = enriched
                sheet_info = f"（シート: {selected_sheet}）" if selected_sheet else ""
                st.success(f"完了: {len(enriched)}件を処理しました。{sheet_info}")
                st.dataframe(enriched.head(100), use_container_width=True)

# ---------- ② ダッシュボード ----------
with tabs[1]:
    st.header("② ダッシュボード")
    enriched = st.session_state.get("enriched")
    if enriched is None or len(enriched)==0:
        st.info("先に①で前処理を実行してください。（data/reviews_examples.xlsx も同梱）")
    else:
        df = enriched.copy()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sources = ["(すべて)"] + sorted(df["source"].dropna().unique().tolist())
            src = st.selectbox("媒体", sources)
        with c2:
            countries = ["(すべて)"] + sorted(df["author_country_code"].dropna().unique().tolist())
            cc = st.selectbox("国", countries)
        with c3:
            labels = ["(すべて)"] + ["pos","neu","neg"]
            lbl = st.selectbox("感情ラベル", labels)
        with c4:
            df["posted_month"] = df["posted_date"].astype(str).str[:7]
            months = ["(すべて)"] + sorted(df["posted_month"].dropna().unique().tolist())
            m = st.selectbox("投稿日（YYYY-MM）", months)

        if src != "(すべて)": df = df[df["source"] == src]
        if cc != "(すべて)": df = df[df["author_country_code"] == cc]
        if lbl != "(すべて)": df = df[df["sentiment_label"] == lbl]
        if m != "(すべて)": df = df[df["posted_month"] == m]

        kpis = kpi_summary(df, low_thr=low_thr)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("平均感情スコア（-1〜+1）", f"{kpis['overall']['avg_sentiment'].iloc[0]:.3f}")
        with k2:
            st.metric("件数", int(kpis['overall']['count'].iloc[0]))
        with k3:
            st.metric("低評価率", f"{kpis['low_rate']['low_rate'].iloc[0]*100:.1f}%")
        with k4:
            st.metric("しきい値", kpis['low_rate']['threshold'].iloc[0])

        ctab1, ctab2, ctab3 = st.tabs(["国別", "キーワード", "明細"])
        with ctab1:
            st.subheader("国別 件数 / 平均感情")
            st.dataframe(kpis["by_country"], use_container_width=True)
        with ctab2:
            st.subheader("フォーカスキーワード言及率")
            st.dataframe(kpis["focus_keywords"], use_container_width=True)
        with ctab3:
            st.subheader("レビュー明細（フィルタ後）")
            st.dataframe(df[["source","review_id","posted_date","author_country_code","sentiment_label","sentiment_score","focus_kw_hits","body","text_ja","sentiment_reason"]].head(800),
                         use_container_width=True)

# ---------- ③ 辞書編集 ----------
with tabs[2]:
    st.header("③ 辞書編集")
    st.caption("キーワード辞書（group/term/label_ja）とロケーション辞書（token/country_code）を編集できます。")

    st.subheader("フォーカスキーワード")
    fk_path = os.path.join(data_dir, "focus_keywords.csv")
    fk_df = load_focus_keywords(fk_path)[["group","term","label_ja"]]
    fk_edit = st.data_editor(fk_df, num_rows="dynamic", use_container_width=True, key="fk_edit")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 フォーカスキーワードを保存"):
            fk_edit.to_csv(fk_path, index=False, encoding="utf-8-sig")
            st.success("保存しました。")
    with c2:
        st.download_button("⬇ CSVとしてダウンロード", data=fk_edit.to_csv(index=False).encode("utf-8-sig"),
                           file_name="focus_keywords.csv", mime="text/csv")

    st.subheader("ロケーション→国コード")
    lc_path = os.path.join(data_dir, "location_to_country.csv")
    lc_df = load_loc_dict(lc_path)[["token","country_code"]]
    lc_edit = st.data_editor(lc_df, num_rows="dynamic", use_container_width=True, key="lc_edit")
    c3, c4 = st.columns(2)
    with c3:
        if st.button("💾 ロケーション辞書を保存"):
            lc_edit.to_csv(lc_path, index=False, encoding="utf-8-sig")
            st.success("保存しました。")
    with c4:
        st.download_button("⬇ CSVとしてダウンロード", data=lc_edit.to_csv(index=False).encode("utf-8-sig"),
                           file_name="location_to_country.csv", mime="text/csv")

# ---------- ④ 出力 ----------
with tabs[3]:
    st.header("④ 出力")
    enriched = st.session_state.get("enriched")
    if enriched is None or len(enriched)==0:
        st.info("②まで実行してから出力してください。")
    else:
        df = enriched.copy()
        kpis = kpi_summary(df, low_thr=low_thr)
        xlsx_bytes = export_excel(df, kpis)
        st.download_button("📥 Excel（KPI＋明細）をダウンロード", data=xlsx_bytes,
                           file_name="summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("📥 CSV（enriched_reviews）をダウンロード", data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="enriched_reviews.csv", mime="text/csv")

st.caption("© Okuhida Kanko Development × Research MVP")
