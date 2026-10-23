
import math
import os
import re

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import ChatOpenAI

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "my",
    "of", "on", "or", "the", "to", "we", "what", "when", "where",
    "which", "who", "with", "you", "your",
}
_RRF_K = 60
_SEMANTIC_WEIGHT = 0.6
_LEXICAL_WEIGHT = 0.4
_PROPERTY_BOOST = 0.15
_SEMANTIC_POOL_MIN = 50

try:
    from nltk.stem import PorterStemmer
except ImportError:
    _stemmer = None
else:
    _stemmer = PorterStemmer()


def _tokenize(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", text.casefold())
    if _stemmer is None:
        def stem(term: str) -> str:
            for suffix in ("ing", "ed", "es", "s"):
                if term.endswith(suffix) and len(term) - len(suffix) >= 3:
                    return term[:-len(suffix)]
            return term
    else:
        stem = _stemmer.stem
    return [stem(term) for term in terms if term not in _STOP_WORDS]


def _corpus_stats(vectorstore: FAISS) -> dict:
    cached = getattr(vectorstore, "_corpus_cache", None)
    if cached is not None and cached["ntotal"] == vectorstore.index.ntotal:
        return cached
    docs = list(vectorstore.docstore._dict.values())
    tokens = [_tokenize(doc.page_content) for doc in docs]
    frequency = {}
    for doc_tokens in tokens:
        for term in set(doc_tokens):
            frequency[term] = frequency.get(term, 0) + 1
    n_docs = len(docs)
    idf = {
        term: math.log(1 + (n_docs - count + 0.5) / (count + 0.5))
        for term, count in frequency.items()
    }
    cached = {
        "ntotal": vectorstore.index.ntotal,
        "docs": docs,
        "tokens": tokens,
        "idf": idf,
    }
    vectorstore._corpus_cache = cached
    return cached


def _property_code(query: str) -> str | None:
    match = re.search(r"\bASH-[A-Z]{3}\b", query.upper())
    return match.group(0) if match else None


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _rrf(rank: int) -> float:
    return (_RRF_K + 1) / (_RRF_K + 1 + rank)


def _deduplicate(docs: list, k: int, threshold: float = 0.9) -> list:
    kept, kept_tokens = [], []
    for doc in docs:
        tokens = set(_tokenize(doc.page_content))
        if any(
            len(tokens & other) / max(len(tokens | other), 1) >= threshold
            for other in kept_tokens
        ):
            continue
        kept.append(doc)
        kept_tokens.append(tokens)
        if len(kept) == k:
            break
    return kept


def _retrieve_single(
    query: str,
    vectorstore: FAISS,
    k: int,
    property_code: str | None = None,
) -> list:
    if vectorstore.index.ntotal == 0:
        raise ValueError("The vector store is empty.")
    stats = _corpus_stats(vectorstore)
    idf = stats["idf"]
    default_idf = max(idf.values(), default=1.0)
    query_tokens = _tokenize(query)
    query_terms = set(query_tokens)
    bigrams = set(zip(query_tokens, query_tokens[1:]))
    total_weight = sum(idf.get(term, default_idf) for term in query_terms) or 1.0
    property_code = property_code or _property_code(query)

    pool_size = min(max(k * 10, _SEMANTIC_POOL_MIN), vectorstore.index.ntotal)
    semantic = vectorstore.similarity_search_with_score(query, k=pool_size)
    higher_is_better = vectorstore.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT
    semantic.sort(key=lambda hit: hit[1], reverse=higher_is_better)
    semantic_rank = {}
    for rank, (doc, _) in enumerate(semantic):
        semantic_rank.setdefault(_normalize(doc.page_content), rank)

    lexical = []
    for doc, tokens in zip(stats["docs"], stats["tokens"]):
        overlap = sum(idf.get(term, default_idf) for term in query_terms & set(tokens)) / total_weight
        phrase = len(bigrams & set(zip(tokens, tokens[1:]))) / max(len(bigrams), 1)
        lexical.append((0.8 * overlap + 0.2 * phrase, doc))
    lexical.sort(key=lambda item: -item[0])
    lexical_rank = {}
    for rank, (score, doc) in enumerate(lexical):
        if score > 0:
            lexical_rank.setdefault(_normalize(doc.page_content), rank)

    pool = {}
    for doc, _ in semantic:
        pool.setdefault(_normalize(doc.page_content), doc)
    for score, doc in lexical[:pool_size]:
        if score > 0:
            pool.setdefault(_normalize(doc.page_content), doc)

    if property_code:
        code = property_code.casefold()
        property_matches = [
            doc for doc in pool.values() if code in doc.page_content.casefold()
        ]
        if property_matches:
            pool = {
                key: doc for key, doc in pool.items()
                if not (
                    re.findall(r"ASH-[A-Z]{3}", doc.page_content.upper())
                    and code not in doc.page_content.casefold()
                )
            }

    missing_rank = len(stats["docs"])
    ranked = []
    for order, (key, doc) in enumerate(pool.items()):
        score = (
            _SEMANTIC_WEIGHT * _rrf(semantic_rank.get(key, missing_rank))
            + _LEXICAL_WEIGHT * _rrf(lexical_rank.get(key, missing_rank))
        )
        if property_code and property_code.casefold() in doc.page_content.casefold():
            score += _PROPERTY_BOOST
        ranked.append((score, order, doc))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return _deduplicate([doc for _, _, doc in ranked], k)


def _split_query(query: str) -> list[str]:
    parts = re.split(
        r"\?\s+|;\s+|,\s+and\s+(?=(?:are|is|do|does|can|what|how|when|where|which|who|will)\b)",
        query.strip(),
        flags=re.IGNORECASE,
    )
    parts = [part.strip(" ?") for part in parts if len(part.strip(" ?").split()) >= 3]
    return parts or [query]


def retrieve_documents(query: str, vectorstore: FAISS, k: int = 5) -> list:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    parts = _split_query(query)
    code = _property_code(query)
    if len(parts) == 1:
        return _retrieve_single(query, vectorstore, k, code)
    ranked_lists = [_retrieve_single(query, vectorstore, k, code)]
    ranked_lists.extend(_retrieve_single(part, vectorstore, k) for part in parts)
    merged, seen = [], set()
    for rank in range(k):
        for results in ranked_lists:
            if rank < len(results):
                key = _normalize(results[rank].page_content)
                if key not in seen:
                    seen.add(key)
                    merged.append(results[rank])
    return _deduplicate(merged, k)


def retrieve(query: str, vectorstore: FAISS, k: int = 5) -> list[str]:
    """Return hybrid-ranked AuroraStays policy chunks with source metadata."""
    docs = retrieve_documents(query, vectorstore, k)
    return [
        f"Content:\n{doc.page_content}\n\nMetadata:\n{doc.metadata}"
        for doc in docs
    ]


_SUPPORTED_PROPERTIES = {
    "ash-atl midtown", "atl midtown", "atlanta midtown",
    "ash-bos harbor", "bos harbor", "boston harbor",
    "ash-chi loop", "chi loop", "chicago loop",
    "ash-den union", "den union", "denver union",
    "ash-mia brickell", "mia brickell", "miami brickell",
    "ash-sea pike", "sea pike", "seattle pike",
}
_PROPERTY_PATTERNS = (
    re.compile(r"\bASH-[A-Z]{3}(?:\s+[A-Z][A-Za-z-]*)?\b", re.IGNORECASE),
    re.compile(
        r"\b[A-Z][a-z]+\s+(?:North|Midtown|Harbor|Loop|Union|Brickell|Pike)\b",
        re.IGNORECASE,
    ),
)


def has_unsupported_named_property(query: str) -> bool:
    for pattern in _PROPERTY_PATTERNS:
        match = pattern.search(query)
        if match:
            name = " ".join(match.group(0).casefold().split())
            return name not in _SUPPORTED_PROPERTIES
    return False


def generate(query: str, retrieved_chunks: list[str], model: ChatOpenAI) -> str:
    prompt = f"""
    You are an AI assistant helping AuroraStays customer-service staff answer questions about
    hotel booking, property, guest-conduct, accessibility, refund, and additional-charge policies.
    Answer using only facts supported by the retrieved policy information. Do not use outside
    knowledge, assumptions, or invented fees, dates, deadlines, room features, eligibility, or
    conditions. Answer directly and concisely, including relevant limits, exceptions, deadlines,
    and fees. Use only information for the named property; never substitute another property's
    policy. If a property-specific answer is requested but the property is unclear, ask which
    AuroraStays property is meant. Explain conflicts instead of choosing silently. If information
    is missing, identify what is supported and what is not. If no relevant information is found,
    say exactly: "I could not find this information in the AuroraStays policy documents. Please
    look it up manually or escalate to the back-office helpdesk." Do not mention retrieved chunks
    or context. Use professional guest-service language.

    User Query:
    {query}

    Retrieved AuroraStays Policy Information:
    {retrieved_chunks}

    Answer:
    """
    response = model.invoke(prompt)
    if not isinstance(response.content, str):
        raise TypeError("The chat model returned non-text content; expected a text answer.")
    return response.content


def rag(
    query: str,
    vectorstore: FAISS,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
) -> tuple[str, list[str]]:
    """Return the generated answer text and retrieved policy chunks."""
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    if has_unsupported_named_property(query):
        return (
            "I could not find property-specific information for the named hotel in the AuroraStays policy documents. "
            "Please look it up manually or escalate to the back-office helpdesk.",
            retrieved_chunks,
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=os.getenv("OPENAI_API_BASE"),
    )
    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)
    return answer, retrieved_chunks
