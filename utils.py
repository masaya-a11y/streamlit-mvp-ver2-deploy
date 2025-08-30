
import re, os, io, hashlib, json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from io import BytesIO

# ================= Normalization =================
def nk_normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_for_match(s: str) -> str:
    s = nk_normalize(s).lower()
    s = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ffー\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ================= Dictionaries =================
def load_focus_keywords(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["group","term","label_ja"])
    if "term_norm" not in df.columns and "term" in df.columns:
        df["term_norm"] = df["term"].apply(normalize_for_match)
    return df

def load_loc_dict(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["token","country_code"])
    return df

# ================= Translation (pluggable) =================
def translate_to_ja(text: str, lang: str, mode: str="stub") -> str:
    """
    Translation stub.
    mode="stub": return original text (no translation).
    mode="pass": return empty string (to fill later).
    mode="api" : TODO - integrate a real translation API.
    """
    if not isinstance(text, str):
        return ""
    if mode == "pass":
        return ""
    return text

# ================= Sentiment (rule-based) =================
POS_WORDS = ["最高", "絶景", "素晴らしい", "良い", "楽しい", "満足", "美味", "おすすめ", "breathtaking", "unforgettable", "amazing"]
NEG_WORDS = ["最悪", "高い", "残念", "混雑", "行列", "待ち", "汚い", "寒すぎ", "ひどい", "bad", "terrible"]

def rule_sentiment_score_10(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = normalize_for_match(text)
    pos = sum(1 for w in POS_WORDS if w.lower() in t)
    neg = sum(1 for w in NEG_WORDS if w.lower() in t)
    score = (pos - neg) * 2.0
    return float(max(-10.0, min(10.0, score)))

def label_from_score10(score10: float, pos_th=2.0, neg_th=-2.0) -> str:
    if score10 >= pos_th: return "pos"
    if score10 <= neg_th: return "neg"
    return "neu"

# ================= LLM Sentiment =================
DEFAULT_SENTIMENT_PROMPT = (
    "このGPTは、日本語で書かれた観光客の口コミを分析し、その感情的なトーンに基づいて数値化されたスコアを付けます。"
    "スコアの範囲は-10（非常にネガティブ）から+10（非常にポジティブ）であり、レビューの内容に含まれる感情的な"
    "キーワードやフレーズを特定し、それに基づいてスコアを決定します。必要であれば、そのスコアの理由も簡潔に説明します。"
    "レビューの内容が曖昧または中立的である場合は、0に近いスコアを付け、曖昧さについても言及します。翻訳や要約は、特に指示がない限り行いません。"
    "応答は簡潔、明瞭かつ専門的な文体で提供されます.\n\n"
    "このGPTは、直接入力された個別の口コミだけでなく、Excelなどの表形式データも処理可能です。表が提供された場合、"
    "「レビュー内容」または同義の日本語列名から内容を抽出し、各レビューに対して感情分析を行い、それぞれにスコアを付与します。\n\n"
    "一貫性を保つため、同一の口コミ文に対しては、異なるリクエストや異なるブラウザ・セッション間でも常に同じ感情スコアを返します。"
    "デバイスや環境の違いが結果に影響しないよう、決定論的な評価を行います。"
)

LLM_SYSTEM_PREFIX = (
    "あなたは日本語口コミの感情評価器です。"
    "出力は厳密なJSONのみ：{\"score\": <数値>, \"label\": \"pos|neu|neg\", \"reason\": \"<80字以内>\"}。"
    "媒体の星評価は参照せず本文の感情だけで判断。"
)

def _sentiment_cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, "sentiment_cache.json")

def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(path: str, cache: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _hash_text(s: str) -> str:
    import hashlib
    norm = normalize_for_match(s)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def llm_sentiment_score_10(text: str, prompt: str = DEFAULT_SENTIMENT_PROMPT,
                           model: str = "gpt-4o-mini", temperature: float = 0.0,
                           seed: int|None = 42, data_dir: str = "data") -> tuple[float,str,str]:
    """
    Returns: (score_10, label, reason). Uses OpenAI if available, else fallback to rule.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, "neu", "空テキスト"
    try:
        from openai import OpenAI
    except Exception as e:
        s10 = rule_sentiment_score_10(text)
        return s10, label_from_score10(s10), "fallback: rule"

    client = OpenAI()  # needs OPENAI_API_KEY

    p_hash = _hash_text(prompt)[:16]
    key = f"{_hash_text(text)}::{model}::{p_hash}"
    cache_path = _sentiment_cache_path(data_dir)
    cache = _load_cache(cache_path)
    if key in cache:
        v = cache[key]
        return float(v["score_10"]), v["label"], v.get("reason","")

    sys_msg = LLM_SYSTEM_PREFIX
    user_msg = (
        "対象テキスト:\n" + text + "\n\n"
        "評価ルール:\n"
        "- 範囲: -10 〜 +10（本文の感情で判断）\n"
        "- 閾値:\n"
        "  * score >= +2.0 → label=\"pos\"\n"
        "  * -2.0 < score < +2.0 → label=\"neu\"\n"
        "  * score <= -2.0 → label=\"neg\"\n"
        "- 一貫性: 同一本文には常に同じ評価\n\n"
        + prompt + "\n必ずJSONのみで返答。"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": sys_msg},
                {"role":"user","content": user_msg},
            ],
            temperature=temperature,
            seed=seed
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        score_10 = float(data.get("score", 0.0))
        label = str(data.get("label","neu"))
        reason = str(data.get("reason",""))
    except Exception as e:
        score_10 = rule_sentiment_score_10(text)
        label = label_from_score10(score_10)
        reason = f"fallback: rule ({type(e).__name__})"

    score_10 = max(-10.0, min(10.0, score_10))
    if label not in ["pos","neu","neg"]:
        label = label_from_score10(score_10)

    cache[key] = {"score_10": score_10, "label": label, "reason": reason}
    _save_cache(cache_path, cache)

    return score_10, label, reason

# ================= Country inference =================
def build_token_map(loc_df: pd.DataFrame) -> Dict[str,str]:
    mp = {}
    for _, r in loc_df.iterrows():
        token = normalize_for_match(str(r.get("token","")))
        cc = str(r.get("country_code","")).strip()
        if token:
            mp[token] = cc
    return mp

def infer_country(raw: str, lang_hint: str, token_map: Dict[str,str]) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    n = normalize_for_match(raw)
    if n in token_map:
        return token_map[n]
    for w in n.split():
        if w in token_map:
            return token_map[w]
    if lang_hint in ["zh-TW"] and any(x in n for x in ["台北","taipei","高雄","kaohsiung","台中","taichung","台灣","台湾"]):
        return "TW"
    return "Unknown"

# ================= Focus keyword =================
def build_focus_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    if "term_norm" not in df.columns and "term" in df.columns:
        df["term_norm"] = df["term"].apply(normalize_for_match)
    return df.groupby("group")["term_norm"].apply(list).to_dict() if len(df) else {}

def match_focus_groups(text: str, group_to_terms: Dict[str, List[str]]) -> List[str]:
    if not isinstance(text, str):
        return []
    t = normalize_for_match(text)
    hits = []
    for g, terms in group_to_terms.items():
        if any(term and term in t for term in terms):
            hits.append(g)
    return hits

# ================= Column Mapping =================
STD_COLUMNS = [
    "source","review_id","posted_date","body",  # required
    "visit_date","rating_original","title","language","url",
    "author_name","author_location_raw","author_country_code",
    "companions","tags","media_count"
]

def guess_mapping(file_cols: list[str]) -> dict:
    lc = {c.lower(): c for c in file_cols}
    def pick(*cands):
        for x in cands:
            if x in lc: return lc[x]
        return ""
    return {
        "source": {"mode":"column","value": pick("source","媒体","platform")},
        "review_id": {"mode":"column","value": pick("review_id","id","レビューid","reviewid")},
        "posted_date": {"mode":"column","value": pick("posted_date","date","投稿日","published_date","create_time","update_time")},
        "body": {"mode":"column","value": pick("body","text","review_text","comment","本文")},
        "visit_date": {"mode":"column","value": pick("visit_date","travel_date","stay_date","行った時期")},
        "rating_original": {"mode":"column","value": pick("rating_original","rating","評価")},
        "title": {"mode":"column","value": pick("title","タイトル")},
        "language": {"mode":"column","value": pick("language","lang","言語")},
        "url": {"mode":"column","value": pick("url","review_url","リンク")},
        "author_name": {"mode":"column","value": pick("author_name","user_name","reviewer_display_name","ユーザー名")},
        "author_location_raw": {"mode":"column","value": pick("author_location_raw","user_location","reviewer_location","所在地")},
        "author_country_code": {"mode":"column","value": pick("author_country_code","country","国コード")},
        "companions": {"mode":"column","value": pick("companions","travel_type","同行者")},
        "tags": {"mode":"column","value": pick("tags","タグ")},
        "media_count": {"mode":"column","value": pick("media_count","photos","images","写真枚数")},
    }

def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame()
    file_cols = set(df.columns)
    for std in STD_COLUMNS:
        cfg = mapping.get(std, {"mode":"column","value":""})
        mode = cfg.get("mode","column")
        val  = cfg.get("value","")
        if mode == "column" and val and val in file_cols:
            out[std] = df[val]
        elif mode == "fixed" and val != "":
            out[std] = [val]*len(df)
        else:
            out[std] = ""
    missing = [c for c in ["source","review_id","posted_date","body"] if out[c].isna().all() or (out[c]== "").all()]
    if missing:
        raise ValueError(f"マッピング後に必須列が空です: {missing}")
    return out

# ================= Pipeline =================
def enrich_reviews(df: pd.DataFrame, focus_df: pd.DataFrame, loc_df: pd.DataFrame,
                   translation_mode: str="stub", pos_th=2.0, neg_th=-2.0,
                   engine: str="rule", llm_prompt: str=DEFAULT_SENTIMENT_PROMPT,
                   llm_model: str="gpt-4o-mini", data_dir: str="data") -> pd.DataFrame:
    for c in ["text_ja","sentiment_label","sentiment_score","sentiment_score_10","sentiment_reason","focus_kw_hits","author_country_code"]:
        if c not in df.columns:
            df[c] = ""

    token_map = build_token_map(loc_df)
    group_to_terms = build_focus_groups(focus_df)

    enriched = []
    for _, row in df.iterrows():
        body = row.get("body","")
        lang = str(row.get("language","")).strip() if pd.notna(row.get("language","")) else ""
        text_ja = row.get("text_ja","")
        if not isinstance(text_ja, str) or not text_ja.strip():
            text_ja = translate_to_ja(body, lang, mode=translation_mode)

        if engine == "llm":
            s10, label, reason = llm_sentiment_score_10(text_ja if text_ja else body,
                                                        prompt=llm_prompt, model=llm_model,
                                                        temperature=0.0, seed=42, data_dir=data_dir)
        else:
            s10 = rule_sentiment_score_10(text_ja if text_ja else body)
            label = label_from_score10(s10, pos_th=pos_th, neg_th=neg_th)
            reason = "rule"

        s = round(s10/10.0, 3)
        cc = row.get("author_country_code","")
        if not isinstance(cc, str) or not cc:
            cc = infer_country(row.get("author_location_raw",""), lang, token_map)
        hits = match_focus_groups(f"{text_ja} {body}", group_to_terms)

        new_row = dict(row)
        new_row["text_ja"] = text_ja
        new_row["sentiment_score_10"] = round(s10,1)
        new_row["sentiment_score"] = s
        new_row["sentiment_label"] = label
        new_row["sentiment_reason"] = reason
        new_row["author_country_code"] = cc if cc else "Unknown"
        new_row["focus_kw_hits"] = ";".join(hits)
        enriched.append(new_row)

    return pd.DataFrame(enriched)

# ================= KPIs & Export =================
def kpi_summary(df: pd.DataFrame, low_thr: float=-0.2) -> dict:
    out = {}
    overall = pd.DataFrame([{
        "avg_sentiment": float(df["sentiment_score"].mean() if len(df) else 0.0),
        "count": int(len(df))
    }])
    out["overall"] = overall

    tmp = df.copy()
    tmp["__hits"] = tmp["focus_kw_hits"].apply(lambda x: [h for h in str(x).split(";") if h] if isinstance(x, str) else [])
    exploded = tmp.explode("__hits")
    exploded["hit"] = ((exploded["__hits"].notna()) & (exploded["__hits"] != "")).astype(int)
    by_group = (exploded.groupby("__hits", dropna=True)
                        .agg(hits=("hit","sum"), total=("hit","count"))
                        .reset_index()
                        .rename(columns={"__hits":"group"}))
    by_group = by_group[by_group["group"].notna() & (by_group["group"]!="")]
    by_group["mention_rate"] = (by_group["hits"] / by_group["total"]).fillna(0.0)
    out["focus_keywords"] = by_group.sort_values(["mention_rate","hits"], ascending=[False,False])

    by_country = (df.assign(n=1)
                    .groupby("author_country_code")
                    .agg(count=("n","sum"), avg_sentiment=("sentiment_score","mean"))
                    .reset_index()
                    .sort_values(["count","avg_sentiment"], ascending=[False,False]))
    out["by_country"] = by_country

    low_count = int((df["sentiment_score"] <= low_thr).sum())
    total = int(len(df))
    low_rate = pd.DataFrame([{"threshold": low_thr, "low_rate": (low_count / total) if total else 0.0,
                              "low_count": low_count, "total": total}])
    out["low_rate"] = low_rate
    return out

def export_excel(enriched: pd.DataFrame, kpis: dict) -> BytesIO:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        enriched.to_excel(w, sheet_name="enriched", index=False)
        for name, d in kpis.items():
            d.to_excel(w, sheet_name=name[:31], index=False)
    bio.seek(0)
    return bio
