"""Màn hình Data Assistant: hỏi bằng tiếng Việt, AI viết SQL, chạy trên dữ liệu Olist của app.

Luồng xử lý một câu hỏi:
  1. ask_sql()    : gửi câu hỏi + mô tả bảng don_hang cho AI (Gemini, dự phòng Groq) -> nhận về câu SQL
  2. check_sql()  : chỉ cho phép một câu SELECT đọc bảng don_hang
  3. run_sql()    : chạy bằng DuckDB trên chính dữ liệu app đang dùng (MySQL hoặc snapshot)
     (nếu lỗi, gửi lỗi lại cho AI sửa 1 lần)
  4. ask_answer() : AI đọc kết quả và trả lời ngắn gọn bằng tiếng Việt

Cần khai báo trong Secrets của Streamlit (hoặc .streamlit/secrets.toml khi chạy máy):
  GEMINI_API_KEY = "..."      # bắt buộc (hoặc GROQ_API_KEY)
  GROQ_API_KEY = "..."        # không bắt buộc, dùng khi Gemini quá tải
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import streamlit as st

TABLE = "don_hang"
MAX_ROWS = 500
TIMEOUT = 30                        # mỗi lần gọi AI chờ tối đa 30 giây
MAX_WAIT = 75                       # tổng thời gian tối đa cho một lần gọi AI (kể cả mô hình dự phòng)
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]
GROQ_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
BUSY_WORDS = ("503", "unavailable", "overloaded", "high demand", "429", "resource_exhausted", "rate",
              "timed out", "timeout", "504", "deadline", "500", "internal")

REGIONS = {
    "Norte": {"AC", "AP", "AM", "PA", "RO", "RR", "TO"},
    "Nordeste": {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"},
    "Centro-Oeste": {"DF", "GO", "MS", "MT"},
    "Sudeste": {"ES", "MG", "RJ", "SP"},
    "Sul": {"PR", "RS", "SC"},
}
STATE_REGION = {s: r for r, states in REGIONS.items() for s in states}

SAMPLES = [
    "5 tuyến có tỷ lệ giao trễ cao nhất (ít nhất 100 đơn)",
    "OTD theo từng năm",
    "Bang người bán nào bàn giao trễ cho hãng vận chuyển nhiều nhất?",
    "Điểm đánh giá trung bình của đơn trễ so với đơn đúng hạn",
    "Vùng nào khách nhận hàng lâu nhất?",
    "OTD được tính như thế nào?",
]

# ------------------------------------------------------------------ mô tả dữ liệu cho AI
SCHEMA = f"""
Bạn làm việc với MỘT bảng trong DuckDB tên `{TABLE}` (dữ liệu Olist - sàn thương mại điện tử Brazil,
đơn đặt từ 09/2016 đến 10/2018, tiền tệ BRL). Mỗi dòng là MỘT ĐƠN HÀNG (khoảng 99 nghìn đơn).

Cột:
- order_id, order_status ('delivered','shipped','canceled','invoiced','processing','unavailable','approved','created')
- purchase_at (thời điểm đặt), approved_at, carrier_at (người bán bàn giao cho hãng vận chuyển),
  delivered_at (khách nhận hàng), estimated_at (ngày hẹn giao cho khách), shipping_limit_at (hạn bàn giao của người bán),
  review_at (TIMESTAMP, có thể NULL)
- purchase_year (INTEGER), purchase_month (chuỗi 'YYYY-MM')
- review_score (1-5, có thể NULL)
- seller_id (NULL nếu đơn có nhiều người bán), seller_count
- gmv (DOUBLE): tổng giá sản phẩm của đơn, BRL, không gồm phí vận chuyển
- seller_state, customer_state: mã bang 2 chữ, ví dụ 'SP','RJ','MG','RS','BA'
- seller_region, customer_region: vùng của Brazil: 'Norte','Nordeste','Centro-Oeste','Sudeste','Sul'
- route_id: tuyến "bang người bán → bang khách", ví dụ 'SP → RJ' (ký tự mũi tên →, có dấu cách hai bên)
- route_eligible (BOOLEAN): TRUE nếu đơn chỉ có 1 người bán và đủ bang hai đầu (chỉ những đơn này được tính KPI tuyến)
- is_canceled (BOOLEAN), is_delivered (BOOLEAN: đã giao, không hủy, có delivered_at)
- is_late (BOOLEAN, NULL nếu chưa giao): ngày nhận hàng > ngày hẹn (so theo NGÀY)
- late_days (INTEGER): số ngày giao thực tế trừ ngày hẹn (dương là trễ, âm là sớm), chỉ đơn đã giao
- delivery_days (DOUBLE): số ngày từ lúc đặt đến lúc khách nhận
- promised_days (DOUBLE): số ngày từ lúc đặt đến ngày hẹn
- handover_late (BOOLEAN, NULL nếu thiếu dữ liệu): người bán bàn giao cho hãng vận chuyển sau hạn shipping_limit_at
"""

GLOSSARY = """
Thuật ngữ của app RouteOps (dùng để trả lời câu hỏi khái niệm, không cần SQL):
- OTD (On-Time Delivery): tỷ lệ đơn giao đúng hạn = số đơn giao không trễ / số đơn đã giao. Đơn trễ khi ngày khách nhận
  lớn hơn ngày hẹn (so theo ngày). Đơn bị hủy không tính.
- Tuyến: bang người bán → bang khách hàng, ví dụ SP → RJ. Chỉ tính đơn có đúng một người bán.
- Trạng thái tuyến trong app (OTD 30 ngày gần nhất, cần ít nhất 30 đơn đã giao): GREEN nếu OTD từ 90%,
  YELLOW từ 85% đến dưới 90%, RED dưới 85%; GRAY nếu chưa đủ 30 đơn.
- GMV: tổng giá sản phẩm của đơn (BRL), không gồm phí vận chuyển. GMV at Risk: GMV của các đơn giao trễ,
  hoặc của đơn chưa giao đã quá hạn từ 3 ngày (cần ưu tiên cứu).
- Bàn giao trễ: người bán giao hàng cho hãng vận chuyển sau hạn shipping_limit_at.
- App phát lại dữ liệu lịch sử theo ngày mô phỏng; trợ lý trả lời trên TOÀN BỘ dữ liệu lịch sử 09/2016 - 10/2018.
"""

RULES = f"""
Quy tắc viết SQL (DuckDB):
- Chỉ viết MỘT câu SELECT (được dùng WITH), chỉ đọc bảng `{TABLE}`. Không sửa, xóa, tạo bảng, không đọc file.
- Mỗi dòng đã là một đơn: đếm đơn dùng COUNT(*) hoặc COUNT(DISTINCT order_id).
- Tỷ lệ trễ = AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END) trên các đơn is_delivered. OTD = 1 - tỷ lệ trễ.
  Mẫu: SELECT route_id, COUNT(*) AS so_don,
         ROUND(AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END), 4) AS ty_le_tre
       FROM {TABLE} WHERE is_delivered AND route_eligible GROUP BY route_id HAVING COUNT(*) >= 100
- Câu hỏi về tuyến: thêm điều kiện route_eligible. Khi xếp hạng tuyến hoặc bang, chỉ lấy nhóm có ít nhất 30 đơn
  (hoặc theo số người dùng nêu) để tránh nhóm quá nhỏ.
- Năm/tháng: dùng purchase_year, purchase_month.
- Đặt tên cột kết quả bằng tiếng Việt không dấu, ngắn gọn (vd: tuyen, bang, vung, nam, so_don, ty_le_tre, otd, gmv).
- Tỷ lệ để ở dạng 0-1, làm tròn 4 chữ số. Tiền làm tròn 0 chữ số. Số ngày làm tròn 1 chữ số.
- Luôn có ORDER BY hợp lý và LIMIT tối đa 50.

Trả về DUY NHẤT một JSON:
- Nếu câu hỏi cần dữ liệu: {{"sql": "<câu SQL>"}}
- Nếu là câu hỏi khái niệm (vd "OTD là gì"): trả lời dựa trên phần Thuật ngữ, 2-3 câu ngắn: {{"sql": null, "reply": "<câu trả lời>"}}
- Nếu là lời chào hoặc dữ liệu không có thông tin để trả lời: {{"sql": null, "reply": "<trả lời ngắn, gợi ý câu hỏi phù hợp>"}}
- Câu trả lời trong "reply" viết bình tĩnh, không dùng dấu gạch dài.
"""

ANSWER_RULES = """
Bạn là trợ lý phân tích giao hàng của Olist. Dựa vào câu hỏi và bảng kết quả, trả lời bằng tiếng Việt:
- Tối đa 3 câu ngắn, đi thẳng vào con số chính và một nhận xét có ích cho người vận hành.
- Số viết kiểu Việt Nam: 96.478 đơn, 91,9%. Tỷ lệ 0-1 thì đổi ra % và làm tròn 1 chữ số. Tiền ghi BRL.
- Tuyến ghi dạng "SP → RJ". Giọng bình tĩnh, khách quan. Không dùng từ cảm xúc như "báo động", "khẩn cấp".
- Chỉ dùng số có trong bảng kết quả, không bịa thêm. Nếu bảng rỗng thì nói không tìm thấy dữ liệu phù hợp.
- Không dùng tiêu đề, không gạch đầu dòng, không dùng dấu gạch dài, chỉ dùng gạch ngắn "-" nếu cần.
"""

PAGE_CSS = """
<style>
[data-testid="stChatMessage"] { background:#FFFFFF; border:1px solid #e1eaf2; border-radius:14px;
    padding:12px 16px; margin-bottom:10px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#EAF5FC; }
.st-key-assistant_pills button { background:#FFFFFF; border:1px solid #cbdceb; }
.st-key-assistant_pills button:hover { border-color:#2789E8; color:#2789E8; }
</style>
"""


# ------------------------------------------------------------------ 0. DỮ LIỆU CHO DUCKDB
def build_table(history: pd.DataFrame) -> pd.DataFrame:
    """Chuyển dữ liệu app (cột PascalCase) thành bảng don_hang, tên cột dễ hiểu cho AI."""
    h = history
    status = h.OrderStatus.astype("string").str.lower()
    canceled = status.eq("canceled")
    delivered = ~canceled & h.DeliveredAt.notna()
    late_days = (h.DeliveredAt.dt.normalize() - h.EstimatedAt.dt.normalize()).dt.days
    is_late = (late_days > 0).where(delivered & h.EstimatedAt.notna())
    handover = (h.CarrierAt > h.ShippingLimitAt).where(h.CarrierAt.notna() & h.ShippingLimitAt.notna())
    days = lambda a, b: (a - b).dt.total_seconds() / 86400
    return pd.DataFrame({
        "order_id": h.OrderId, "order_status": status,
        "purchase_at": h.PurchaseAt, "approved_at": h.ApprovedAt, "carrier_at": h.CarrierAt,
        "delivered_at": h.DeliveredAt, "estimated_at": h.EstimatedAt, "shipping_limit_at": h.ShippingLimitAt,
        "review_at": h.ReviewAt,
        "purchase_year": h.PurchaseAt.dt.year.astype("Int64"),
        "purchase_month": h.PurchaseAt.dt.strftime("%Y-%m"),
        "review_score": pd.to_numeric(h.ReviewScore, errors="coerce"),
        "seller_id": h.SellerId, "seller_count": h.SellerCount, "gmv": h.GMV.astype(float),
        "seller_state": h.SellerState, "customer_state": h.CustomerState,
        "seller_region": h.SellerState.map(STATE_REGION), "customer_region": h.CustomerState.map(STATE_REGION),
        "route_id": h.RouteId.where(h.RouteEligible), "route_eligible": h.RouteEligible.astype(bool),
        "is_canceled": canceled.astype(bool), "is_delivered": delivered.astype(bool),
        "is_late": is_late.astype("boolean"),
        "late_days": late_days.where(delivered).astype("Int64"),
        "delivery_days": days(h.DeliveredAt, h.PurchaseAt).where(delivered),
        "promised_days": days(h.EstimatedAt, h.PurchaseAt),
        "handover_late": handover.astype("boolean"),
    })


@st.cache_resource(show_spinner=False)
def _database(_history: pd.DataFrame, token: str):
    """Một cơ sở dữ liệu DuckDB trong bộ nhớ, chỉ có bảng don_hang, tắt quyền đọc file/mạng."""
    import duckdb
    con = duckdb.connect(":memory:")
    con.register("_src", build_table(_history))
    con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM _src")
    con.unregister("_src")
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


# ------------------------------------------------------------------ 1. GỌI AI
def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def providers():
    """Thứ tự thử: Gemini trước, Groq dự phòng (chỉ những cái có key)."""
    order = [name for name, key in (("Gemini", "GEMINI_API_KEY"), ("Groq", "GROQ_API_KEY")) if _secret(key)]
    if _secret("AI_PRIMARY") == "Groq" and "Groq" in order:
        order = ["Groq"] + [p for p in order if p != "Groq"]
    return order


@st.cache_resource
def _groq():
    from groq import Groq
    return Groq(api_key=_secret("GROQ_API_KEY"), timeout=TIMEOUT, max_retries=1)


@st.cache_resource
def _gemini():
    from google import genai
    from google.genai import types
    return genai.Client(api_key=_secret("GEMINI_API_KEY"), http_options=types.HttpOptions(timeout=TIMEOUT * 1000))


def _is_busy(e):
    return any(w in str(e).lower() for w in BUSY_WORDS)


def _friendly(e):
    if _is_busy(e):
        return "AI đang quá tải hoặc hết lượt miễn phí. Bạn đợi khoảng 1 phút rồi hỏi lại nhé."
    return f"Không gọi được AI: {str(e)[:300]}"


def _gemini_config(model, system, json_mode, fast):
    from google.genai import types
    kw = dict(system_instruction=system, temperature=0.1,
              response_mime_type="application/json" if json_mode else "text/plain")
    if fast:
        try:
            kw["thinking_config"] = (types.ThinkingConfig(thinking_budget=0) if "2.5" in model
                                     else types.ThinkingConfig(thinking_level="low"))
        except Exception:
            pass
    return types.GenerateContentConfig(**kw)


def _call_gemini(system, prompt, json_mode, deadline):
    first = _secret("GEMINI_MODEL")
    models = ([first] if first else []) + [m for m in GEMINI_MODELS if m != first]
    last = None
    for model in models:
        if time.time() > deadline:
            break
        for fast in (True, False):              # lỗi 400 do cấu hình "suy nghĩ" -> thử lại không kèm cấu hình
            try:
                cfg = _gemini_config(model, system, json_mode, fast)
                return _gemini().models.generate_content(model=model, contents=prompt, config=cfg).text
            except Exception as e:
                last = e
                m = str(e).lower()
                if fast and ("400" in m or "invalid" in m or "thinking" in m):
                    continue
                break
        m = str(last).lower()
        if not (_is_busy(last) or "not found" in m or "404" in m or "400" in m or "invalid" in m):
            break
    raise last or TimeoutError("timeout: AI phản hồi quá lâu")


def _call_groq(system, prompt, json_mode, deadline):
    models = [_secret("GROQ_MODEL")] if _secret("GROQ_MODEL") else GROQ_MODELS
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    last = None
    for model in models:
        if time.time() > deadline:
            break
        try:
            r = _groq().chat.completions.create(
                model=model, temperature=0.1, max_tokens=1200,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], **extra)
            return r.choices[0].message.content
        except Exception as e:
            last = e
            if not (_is_busy(e) or "model" in str(e).lower()):
                break
    raise last or TimeoutError("timeout: AI phản hồi quá lâu")


def _generate(system, prompt, json_mode):
    order = providers()
    if not order:
        raise RuntimeError("Chưa có API key của AI trong Secrets.")
    deadline = time.time() + MAX_WAIT
    last = None
    for name in order:
        try:
            return (_call_gemini if name == "Gemini" else _call_groq)(system, prompt, json_mode, deadline)
        except Exception as e:
            last = e
    raise RuntimeError(_friendly(last))


@st.cache_data(ttl=3600, show_spinner=False)
def _ask_sql_cached(question, ctx, error, bad_sql):
    prompt = (f"Ngữ cảnh hội thoại gần đây (dùng nếu câu hỏi mới nói tiếp ý trước):\n{ctx or '(không có)'}\n\n"
              f"Câu hỏi mới: {question}")
    if error:
        prompt += f"\n\nCâu SQL trước bị lỗi:\n{bad_sql}\nLỗi: {error}\nHãy viết lại câu SQL đúng."
    return _generate(SCHEMA + GLOSSARY + RULES, prompt, json_mode=True)


def ask_sql(question, history, error=None, bad_sql=None):
    ctx = "".join(f"- Câu hỏi trước: {h['q']}\n  SQL đã dùng: {h['sql']}\n" for h in history[-2:] if h.get("sql"))
    text = _ask_sql_cached(question.strip(), ctx, error, bad_sql)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else {"sql": None, "reply": text}
    return out if isinstance(out, dict) else {"sql": None, "reply": str(out)}


@st.cache_data(ttl=3600, show_spinner=False)
def ask_answer(question, sql, df):
    table = df.head(30).to_csv(index=False)
    prompt = f"Câu hỏi: {question}\n\nSQL đã chạy:\n{sql}\n\nKết quả ({len(df)} dòng, tối đa 30 dòng đầu):\n{table}"
    return _generate(ANSWER_RULES + GLOSSARY, prompt, json_mode=False).strip().replace("—", "-").replace("–", "-")


# ------------------------------------------------------------------ 2. KIỂM TRA SQL
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|COPY|EXPORT|IMPORT|"
                       r"LOAD|INSTALL|ATTACH|DETACH|PRAGMA|CALL|SET|RESET|BEGIN|COMMIT|ROLLBACK|EXECUTE|"
                       r"VACUUM|CHECKPOINT|USE)\b", re.I)
FILE_FUNCS = re.compile(r"\b(read_\w+|glob|sniff_csv|parquet_\w+|duckdb_\w+|pragma_\w+|getenv|"
                        r"current_setting|which_secret)\s*\(", re.I)


def check_sql(sql):
    """Chỉ cho phép một câu SELECT đọc bảng don_hang."""
    s = re.sub(r"^```(?:sql)?|```$", "", (sql or "").strip(), flags=re.I).strip().rstrip(";").strip()
    no_str = re.sub(r"'(?:[^'\\]|\\.|'')*'", "''", s)
    no_str = re.sub(r"--[^\n]*|/\*.*?\*/", " ", no_str, flags=re.S)
    if not s:
        raise ValueError("SQL rỗng.")
    if ";" in no_str:
        raise ValueError("Chỉ được chạy một câu lệnh.")
    if not re.match(r"(?is)^\s*(WITH|SELECT)\b", no_str):
        raise ValueError("Chỉ được dùng câu SELECT.")
    if FORBIDDEN.search(no_str) or FILE_FUNCS.search(no_str):
        raise ValueError("Câu SQL chứa lệnh không được phép.")
    scan = re.sub(r"(?is)\b(EXTRACT|TRIM|SUBSTRING|OVERLAY)\s*\([^()]*\)", " ", no_str)  # FROM bên trong hàm
    ctes = {c.lower() for c in re.findall(r"(?i)(?:\bWITH|,)\s*([A-Za-z_]\w*)\s+AS\s*\(", scan)}
    refs = re.findall(r"(?i)\b(?:FROM|JOIN)\s+(\"[^\"]+\"|[A-Za-z_][\w.]*|\()", scan)
    for ref in refs:
        name = ref.strip('"').lower()
        if ref == "(" or name in ctes or name == TABLE:
            continue
        raise ValueError(f"Chỉ được đọc bảng {TABLE} (gặp: {ref}).")
    if not any(r.strip('"').lower() == TABLE for r in refs):
        raise ValueError(f"Câu SQL phải đọc từ bảng {TABLE}.")
    return s


# ------------------------------------------------------------------ 3. CHẠY BẰNG DUCKDB
def run_sql(con, sql):
    cur = con.cursor()
    try:
        return cur.execute(sql).fetchdf().head(MAX_ROWS)
    finally:
        cur.close()


def answer(con, question, history, step=lambda text: None):
    step("Đang viết truy vấn SQL...")
    plan = ask_sql(question, history)
    if not plan.get("sql"):
        reply = plan.get("reply") or "Mình chưa hiểu câu hỏi, bạn thử hỏi lại nhé."
        return {"q": question, "text": reply.replace("—", "-").replace("–", "-")}
    sql, error = plan["sql"], None
    for attempt in range(2):                    # lần đầu + 1 lần nhờ AI sửa lỗi
        try:
            sql = check_sql(sql)
            step("Đang chạy truy vấn trên dữ liệu Olist...")
            df = run_sql(con, sql)
            step("Đang tóm tắt kết quả...")
            return {"q": question, "sql": sql, "text": ask_answer(question, sql, df)}
        except RuntimeError:                    # lỗi gọi AI -> báo luôn
            raise
        except Exception as e:
            error = str(e)[:500]
            if attempt == 1:
                break
            step("Truy vấn bị lỗi, đang nhờ AI sửa...")
            plan = ask_sql(question, history, error=error, bad_sql=sql)
            if not plan.get("sql"):
                break
            sql = plan["sql"]
    return {"q": question, "sql": sql, "text": f"Mình chưa chạy được truy vấn cho câu này. Lỗi: {error}"}


# ------------------------------------------------------------------ 4. GIAO DIỆN
def screen_assistant(history_df: pd.DataFrame) -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown("""<section class="hero"><div class="eyebrow">Olist · data assistant</div>
    <h1>Trợ lý dữ liệu</h1>
    <p>Hỏi bằng tiếng Việt về giao hàng, tuyến, người bán và đánh giá của khách.</p></section>""",
                unsafe_allow_html=True)

    if not providers():
        st.warning("Chưa có API key của AI. Vào Settings → Secrets của app trên Streamlit Cloud "
                   "(hoặc file .streamlit/secrets.toml khi chạy trên máy) và thêm dòng "
                   '`GEMINI_API_KEY = "..."`.')
        return

    token = f"{len(history_df)}|{history_df.PurchaseAt.max()}|{history_df.attrs.get('data_source')}"
    con = _database(history_df, token)

    chat = st.session_state.setdefault("assistant_chat", [])
    with st.sidebar:
        if st.button("Xóa cuộc trò chuyện", width="stretch", key="assistant_clear"):
            chat.clear()
            st.rerun()

    question = st.chat_input("Ví dụ: Tỷ lệ giao trễ của tuyến SP → RJ là bao nhiêu?") \
        or st.session_state.pop("assistant_picked", None)

    def _pick():
        st.session_state["assistant_picked"] = st.session_state["assistant_pills"]
        st.session_state["assistant_pills"] = None

    st.pills("Gợi ý", SAMPLES, selection_mode="single", key="assistant_pills",
             label_visibility="collapsed", on_change=_pick)

    for msg in chat:
        with st.chat_message("user"):
            st.markdown(msg["q"])
        with st.chat_message("assistant"):
            st.markdown(msg["text"])

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Đang xử lý câu hỏi...", expanded=False) as status:
                try:
                    msg = answer(con, question, chat, step=lambda t: status.update(label=t))
                    status.update(label="Xong", state="complete")
                except Exception as e:
                    msg = {"q": question, "text": str(e) if isinstance(e, RuntimeError) else f"Có lỗi khi xử lý: {e}"}
                    status.update(label="Không hoàn thành", state="error")
            st.markdown(msg["text"])
        chat.append(msg)
