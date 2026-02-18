"""
metrics.py
==========
HOW EACH METRIC IS CALCULATED
──────────────────────────────
① Precision@10 improvement over BM25
   Run BM25-only → top-10 chunks → cross-encoder score each (relevance proxy)
   Same for hybrid top-10. precision = relevant_count/10.
   improvement% = (hybrid_p - bm25_p) / bm25_p * 100

② Hallucination rate
   Split LLM response into sentences.
   Embed sentences + retrieved chunks with SentenceTransformer.
   max cosine-similarity(sentence, chunks) < 0.35 → not grounded → hallucinated.
   rate = hallucinated/total * 100

③ Response time
   Wall-clock timer around /chat pipeline. Rolling stats over last 20 calls.

④ Language detection accuracy
   langdetect tested against 30 labelled sentences (3 × 10 languages).
   accuracy = correct/30 * 100. Cached after first run.
"""

import re, logging
from typing import List, Dict, Optional
import numpy as np
from langdetect import detect
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD     = 0.35  
HALLUCINATION_THRESHOLD = 0.35
ROLLING_WINDOW          = 20

_cross_encoder : Optional[CrossEncoder]        = None
_sentence_model: Optional[SentenceTransformer] = None
_lang_cache    : Optional[float]               = None
_response_times: List[float]                   = []
_metric_history: List[Dict]                    = []

def _ce():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def _st():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

# ─────────────────────────────────────────────────────────
# ① Precision@10
# ─────────────────────────────────────────────────────────
def compute_precision_improvement(query, all_chunks, hybrid_top, k=10):
    ce = _ce()

    # BM25 retrieval
    bm25 = BM25Okapi([c.split() for c in all_chunks])
    raw = bm25.get_scores(query.split())
    top_idx = np.argsort(raw)[-k:][::-1]
    bm25_cands = [all_chunks[i] for i in top_idx]

    # Cross-encoder scores for BM25
    bm25_ce = ce.predict([(query, c) for c in bm25_cands]).tolist()

    # Dynamic relevance using median (relative ranking)
    if bm25_ce:
        bm25_threshold = np.median(bm25_ce)
        bm25_rel = sum(1 for s in bm25_ce if s >= bm25_threshold)
    else:
        bm25_rel = 0

    bm25_p = bm25_rel / max(len(bm25_cands), 1)

    # Hybrid candidates
    h_cands = hybrid_top[:k]
    h_ce = ce.predict([(query, c) for c in h_cands]).tolist() if h_cands else []

    if h_ce:
        h_threshold = np.median(h_ce)
        h_rel = sum(1 for s in h_ce if s >= h_threshold)
    else:
        h_rel = 0

    h_p = h_rel / max(len(h_cands), 1)

    # Improvement calculation
    improvement = ((h_p - bm25_p) / max(bm25_p, 0.001)) * 100

    return {
        "bm25_precision": round(bm25_p * 100, 1),
        "hybrid_precision": round(h_p * 100, 1),
        "improvement_pct": round(improvement, 1),
        "target_pct": 30.0,
        "meets_target": improvement >= 30.0,
        "bm25_scores": [round(s, 3) for s in bm25_ce],
        "hybrid_scores": [round(s, 3) for s in h_ce],
        "bm25_relevant": bm25_rel,
        "hybrid_relevant": h_rel,
        "k": k,
        "formula": "(hybrid_precision − bm25_precision) / bm25_precision × 100",
    }

# ─────────────────────────────────────────────────────────
# ② Hallucination rate
# ─────────────────────────────────────────────────────────
def _split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 10]

def compute_hallucination_rate(response_text, retrieved_chunks):
    st    = _st()
    sents = _split_sentences(response_text)
    empty = {"hallucination_rate":0.0,"target_pct":5.0,"meets_target":True,
             "total_sentences":0,"hallucinated":0,"grounded":0,"details":[],
             "threshold":HALLUCINATION_THRESHOLD,
             "formula":"hallucinated_sentences / total_sentences × 100"}
    if not sents or not retrieved_chunks:
        return empty

    s_embs = st.encode(sents,            convert_to_numpy=True)
    c_embs = st.encode(retrieved_chunks, convert_to_numpy=True)

    details, hallucinated = [], 0
    for sent, emb in zip(sents, s_embs):
        sims    = cosine_similarity([emb], c_embs)[0]
        max_sim = float(np.max(sims))
        ok      = max_sim >= HALLUCINATION_THRESHOLD
        if not ok: hallucinated += 1
        details.append({
            "sentence": sent[:120] + ("…" if len(sent) > 120 else ""),
            "max_sim" : round(max_sim, 3),
            "grounded": ok,
        })

    rate = hallucinated / len(sents) * 100
    return {
        "hallucination_rate": round(rate, 1),
        "target_pct"        : 5.0,
        "meets_target"      : rate <= 5.0,
        "total_sentences"   : len(sents),
        "hallucinated"      : hallucinated,
        "grounded"          : len(sents) - hallucinated,
        "details"           : details,
        "threshold"         : HALLUCINATION_THRESHOLD,
        "formula"           : "hallucinated_sentences / total_sentences × 100",
    }

# ─────────────────────────────────────────────────────────
# ③ Response time
# ─────────────────────────────────────────────────────────
def record_response_time(s):
    _response_times.append(round(s, 3))
    if len(_response_times) > ROLLING_WINDOW:
        _response_times.pop(0)

def compute_response_time_stats():
    if not _response_times:
        return {"last_s":0,"avg_s":0,"min_s":0,"max_s":0,"target_s":30.0,
                "meets_target":True,"samples":0,"history":[],
                "formula":"wall-clock time: pipeline start → response ready"}
    arr  = np.array(_response_times)
    last = _response_times[-1]
    return {
        "last_s"      : round(last,              3),
        "avg_s"       : round(float(np.mean(arr)),3),
        "min_s"       : round(float(np.min(arr)), 3),
        "max_s"       : round(float(np.max(arr)), 3),
        "target_s"    : 30.0,
        "meets_target": last <= 30.0,
        "samples"     : len(_response_times),
        "history"     : _response_times[-10:],
        "formula"     : "wall-clock time: pipeline start → response ready",
    }

# ─────────────────────────────────────────────────────────
# ④ Language detection accuracy
# ─────────────────────────────────────────────────────────
_LANG_TESTS = [
    ("The quick brown fox jumps over the lazy dog.", "en"),
    ("Machine learning is transforming every industry.", "en"),
    ("Please submit your report before the deadline.", "en"),
    ("El aprendizaje automático está cambiando la industria.", "es"),
    ("Por favor envíe su informe antes de la fecha límite.", "es"),
    ("Buenos días, ¿cómo puedo ayudarle hoy?", "es"),
    ("L'intelligence artificielle révolutionne le monde.", "fr"),
    ("Veuillez soumettre votre rapport avant la date limite.", "fr"),
    ("Bonjour, comment puis-je vous aider aujourd'hui?", "fr"),
    ("Maschinelles Lernen verändert jede Branche.", "de"),
    ("Bitte reichen Sie Ihren Bericht vor der Deadline ein.", "de"),
    ("Guten Morgen, wie kann ich Ihnen helfen?", "de"),
    ("L'apprendimento automatico sta trasformando ogni settore.", "it"),
    ("Si prega di presentare il rapporto prima della scadenza.", "it"),
    ("Buongiorno, come posso aiutarla oggi?", "it"),
    ("O aprendizado de máquina está transformando cada setor.", "pt"),
    ("Por favor, envie seu relatório antes do prazo.", "pt"),
    ("Bom dia, como posso ajudá-lo hoje?", "pt"),
    ("Машинное обучение меняет каждую отрасль.", "ru"),
    ("Пожалуйста, представьте свой отчёт до срока.", "ru"),
    ("Доброе утро, чем могу помочь?", "ru"),
    ("机器学习正在改变每个行业。", "zh-cn"),
    ("请在截止日期前提交您的报告。", "zh-cn"),
    ("早上好，我今天能帮您什么？", "zh-cn"),
    ("機械学習はすべての産業を変革しています。", "ja"),
    ("締め切り前にレポートを提出してください。", "ja"),
    ("おはようございます、今日はどのようにお手伝いできますか？", "ja"),
    ("التعلم الآلي يغير كل صناعة.", "ar"),
    ("يرجى تقديم تقريرك قبل الموعد النهائي.", "ar"),
    ("صباح الخير، كيف يمكنني مساعدتك اليوم؟", "ar"),
]

def compute_language_accuracy():
    global _lang_cache
    rows, correct = [], 0
    for text, expected in _LANG_TESTS:
        try:
            got = detect(text)
            ok  = got.split('-')[0] == expected.split('-')[0]
        except Exception:
            got, ok = "error", False
        if ok: correct += 1
        rows.append({"text": text[:55]+"…", "expected": expected, "detected": got, "correct": ok})
    acc = correct / len(_LANG_TESTS) * 100
    _lang_cache = acc
    return {"accuracy_pct": round(acc,1), "target_pct": 95.0,
            "meets_target": acc >= 95.0, "correct": correct,
            "total": len(_LANG_TESTS), "rows": rows, "cached": False,
            "formula": "correct_detections / 30 labelled sentences × 100"}

def get_language_accuracy():
    if _lang_cache is not None:
        return {"accuracy_pct": round(_lang_cache,1), "target_pct": 95.0,
                "meets_target": _lang_cache >= 95.0, "total": len(_LANG_TESTS),
                "cached": True,
                "formula": "correct_detections / 30 labelled sentences × 100"}
    return compute_language_accuracy()

# ─────────────────────────────────────────────────────────
# Master evaluator — call once per /chat request
# ─────────────────────────────────────────────────────────
def evaluate_request(query, all_chunks, hybrid_top, response_text, elapsed_s):
    record_response_time(elapsed_s)
    p  = compute_precision_improvement(query, all_chunks, hybrid_top)
    h  = compute_hallucination_rate(response_text, hybrid_top)
    rt = compute_response_time_stats()
    la = get_language_accuracy()
    _metric_history.append({
        "precision_pct"    : p["improvement_pct"],
        "hallucination_pct": h["hallucination_rate"],
        "response_s"       : rt["last_s"],
        "lang_acc_pct"     : la["accuracy_pct"],
    })
    if len(_metric_history) > 50:
        _metric_history.pop(0)
    return {"precision": p, "hallucination": h, "response_time": rt, "language": la}

def get_history():
    return _metric_history