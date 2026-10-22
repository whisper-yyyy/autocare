"""Dependency-free BM25 retrieval tuned for Chinese automotive questions."""

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Chunk, RetrievalHit


SYNONYMS = {
    "刹车油": "制动液",
    "打不着火": "启动困难",
    "启动不了": "启动困难",
    "无法发动": "启动困难",
    "电瓶": "蓄电池",
    "刹车": "制动",
    "冷气": "空调",
    "修车": "维修",
    "胎压灯": "胎压警告灯",
}

# Generic request words must not be enough to make an unsupported answer look
# grounded. They remain in the document text, but are ignored as query anchors.
QUERY_STOP_TOKENS = {
    "车",
    "车辆",
    "汽车",
    "车型",
    "这个",
    "这辆",
    "我的",
    "怎么",
    "如何",
    "什么",
    "哪个",
    "哪些",
    "多少",
    "多久",
    "一次",
    "一下",
    "问题",
    "相关",
    "可以",
    "能否",
    "是否",
    "应该",
    "需要",
    "更换",
    "检查",
    "维修",
    "保养",
    "使用",
    "处理",
    "办法",
    "情况",
    "请问",
}

# These phrases act as explainable semantic anchors without requiring a heavy
# tokenizer. Unknown terms still work through matching Chinese bigrams.
DOMAIN_CONCEPTS = tuple(
    sorted(
        {
            "安全气囊",
            "制动踏板",
            "制动距离",
            "制动液",
            "蓄电池",
            "充电系统",
            "启动困难",
            "冷却液",
            "散热器",
            "发动机",
            "水温",
            "空调",
            "制冷剂",
            "压缩机",
            "胎压警告灯",
            "胎压",
            "轮胎",
            "机油",
            "火花塞",
            "变速箱油",
            "质保",
            "召回",
            "故障码",
            "异响",
            "漏油",
            "漏水",
            "雨刮器",
            "保险丝",
            "传感器",
            "制动",
            "转向",
            "启动",
            "冒烟",
            "着火",
            "爆胎",
        },
        key=len,
        reverse=True,
    )
)


def canonicalize(text: str) -> str:
    normalized = text.lower()
    for original in sorted(SYNONYMS, key=len, reverse=True):
        replacement = SYNONYMS[original]
        normalized = normalized.replace(original, replacement)
    return normalized


def tokenize(text: str) -> List[str]:
    """Tokenize Latin words, diagnostic codes, and Chinese character bigrams."""

    text = canonicalize(text)
    tokens = re.findall(r"[a-z0-9][a-z0-9._-]*", text)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def query_anchors(text: str) -> List[str]:
    """Return non-generic concepts used to verify that evidence is grounded."""

    normalized = canonicalize(text)
    anchors: List[str] = [term for term in DOMAIN_CONCEPTS if term in normalized]
    anchors.extend(token for token in tokenize(normalized) if token not in QUERY_STOP_TOKENS)
    return list(dict.fromkeys(anchors))


def _matched_anchors(query: str, chunk_text: str, frequencies: Counter) -> Tuple[str, ...]:
    normalized_chunk = canonicalize(chunk_text)
    matched = []
    for anchor in query_anchors(query):
        if anchor in DOMAIN_CONCEPTS:
            is_match = anchor in normalized_chunk
        else:
            is_match = frequencies.get(anchor, 0) > 0
        if is_match:
            matched.append(anchor)
    return tuple(matched)


class BM25Retriever:
    """Small in-memory BM25 index with interpretable coverage scores."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self._tokens: List[List[str]] = []
        self._frequencies: List[Counter] = []
        self._document_frequency: Dict[str, int] = {}
        self._average_length = 0.0

    def build(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )
        document_frequency: Counter = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        self._document_frequency = dict(document_frequency)

    def add(self, chunks: Iterable[Chunk]) -> None:
        new_chunks = list(chunks)
        if not new_chunks:
            return

        previous_count = len(self._tokens)
        previous_total_length = self._average_length * previous_count
        new_tokens = [tokenize(chunk.text) for chunk in new_chunks]
        self.chunks.extend(new_chunks)
        self._tokens.extend(new_tokens)
        self._frequencies.extend(Counter(tokens) for tokens in new_tokens)

        document_frequency = Counter(self._document_frequency)
        for tokens in new_tokens:
            document_frequency.update(set(tokens))
        self._document_frequency = dict(document_frequency)
        self._average_length = (
            previous_total_length + sum(len(tokens) for tokens in new_tokens)
        ) / len(self._tokens)

    def _idf(self, token: str) -> float:
        total = len(self.chunks)
        frequency = self._document_frequency.get(token, 0)
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[RetrievalHit]:
        query_tokens = [
            token
            for token in dict.fromkeys(tokenize(query))
            if token not in QUERY_STOP_TOKENS
        ]
        if not query_tokens or not self.chunks:
            return []

        hits: List[RetrievalHit] = []
        query_set = set(query_tokens)
        anchors = query_anchors(query)
        allowed_documents = set(document_ids) if document_ids is not None else None
        for chunk, tokens, frequencies in zip(self.chunks, self._tokens, self._frequencies):
            if allowed_documents is not None and chunk.document_id not in allowed_documents:
                continue
            document_length = max(len(tokens), 1)
            bm25_score = 0.0
            for token in query_tokens:
                term_frequency = frequencies.get(token, 0)
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * document_length / max(self._average_length, 1.0)
                )
                bm25_score += self._idf(token) * (term_frequency * (self.k1 + 1)) / denominator

            matched = query_set.intersection(frequencies.keys())
            coverage = len(matched) / len(query_set)
            matched_anchors = _matched_anchors(query, chunk.text, frequencies)
            anchor_coverage = len(matched_anchors) / max(len(anchors), 1)
            scaled_bm25 = bm25_score / (bm25_score + 3.0) if bm25_score > 0 else 0.0
            concept_bonus = 0.12 if any(anchor in DOMAIN_CONCEPTS for anchor in matched_anchors) else 0.0
            combined_score = 0.55 * coverage + 0.33 * scaled_bm25 + concept_bonus
            if combined_score > 0 and matched_anchors:
                hits.append(
                    RetrievalHit(
                        chunk=chunk,
                        score=combined_score,
                        bm25_score=bm25_score,
                        query_coverage=coverage,
                        anchor_coverage=anchor_coverage,
                        matched_anchors=matched_anchors,
                    )
                )

        hits.sort(key=lambda hit: (hit.score, hit.bm25_score), reverse=True)
        return hits[: max(top_k, 0)]
