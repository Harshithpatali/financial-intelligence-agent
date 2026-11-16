"""Evidence graph with explicit mention, co-occurrence and temporal semantics.

CO_MENTIONED_WITHIN_CHUNK is not causality. No causal edges are inferred.
"""
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import networkx as nx

INDEX_PATH = Path(__file__).resolve().parents[1] / "indexes" / "tcs_knowledge_graph.json"
ENTITY_ALIASES = {
    "Tata Consultancy Services": ["tata consultancy services", "tcs"],
    "Revenue": ["total revenue", "operating revenue", "revenue", "total income"],
    "Net Income": ["net income", "net profit", "profit after tax", "pat"],
    "Profit Before Tax": ["profit before tax", "profit before taxes", "pbt"],
    "Earnings Per Share": ["earnings per share", "basic eps", "diluted eps", "eps"],
    "Total Assets": ["total assets"],
    "Total Equity": ["total equity"],
    "Cash and Cash Equivalents": ["cash and cash equivalents"],
    "Trade Receivables": ["trade receivables"],
    "Current Liabilities": ["current liabilities"],
    "Operating Income": ["operating income", "operating profit", "ebit"],
    "Operating Margin": ["operating margin", "ebit margin"],
    "Gross Profit": ["gross profit"],
    "Cash Flow": ["operating cash flow", "free cash flow", "cash flow"],
    "Operating Cash Flow": ["net cash flows generated from operating activities"],
    "Investing Cash Flow": ["net cash flows used in investing activities"],
    "Foreign Currency Sensitivity": ["foreign currency movement", "functional currency"],
    "Capital Expenditure": ["capital expenditure", "capex"],
    "Debt": ["total debt", "borrowings", "debt"],
    "Artificial Intelligence": ["artificial intelligence", "generative ai", "genai"],
    "Cloud": ["cloud services", "cloud"],
    "Banking": ["banking", "financial services", "bsfi"],
    "Employees": ["employees", "workforce", "associates", "employee headcount"],
    "Consultancy Services": ["consultancy services"],
    "Contract Value": ["total contract value", "tcv"],
    "Dividend": ["dividends", "dividend"],
    "Digital Services": ["digital services", "digital transformation"],
    "North America": ["north america", "united states"],
    "Europe": ["europe", "uk market"],
}
YEAR_RE = re.compile(r"\b(?:FY\s*)?(20\d{2})(?:[-–/](\d{2,4}))?\b", re.I)
TREND_PATTERNS = {
    "INCREASED": re.compile(r"\b(increased|grew|growth|rose|higher|improved)\b", re.I),
    "DECREASED": re.compile(r"\b(decreased|declined|fell|lower|reduced|contracted)\b", re.I),
}


def _contains_alias(text, alias):
    return re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text) is not None


def extract_entities(text):
    normalized = re.sub(r"\s+", " ", str(text).lower())
    found = {entity for entity, aliases in ENTITY_ALIASES.items()
             if any(_contains_alias(normalized, alias) for alias in aliases)}
    for match in YEAR_RE.finditer(str(text)):
        start, suffix = match.groups()
        found.add("FY" + start)
        if suffix:
            found.add("FY" + (suffix if len(suffix) == 4 else start[:2] + suffix))
    return sorted(found)


def _add_edge(graph, left, right, relation):
    if graph.has_edge(left, right):
        relations = graph[left][right].setdefault("relations", [])
        if relation not in relations:
            relations.append(relation)
    else:
        graph.add_edge(left, right, relations=[relation])


def build_graph(chunks, output_path=INDEX_PATH):
    graph = nx.Graph()
    periods = set()
    for chunk in chunks:
        chunk_id = str(chunk["id"])
        text = str(chunk["text"])
        entities = extract_entities(text)
        period_entities = [entity for entity in entities if entity.startswith("FY")]
        concepts = [entity for entity in entities if not entity.startswith("FY")]
        evidence = "CHUNK:" + chunk_id
        ref = {"chunk_id": chunk_id, "source": chunk.get("source", "Unknown"),
               "page": chunk.get("page"), "text": text[:1200]}
        graph.add_node(evidence, node_type="evidence_chunk", **ref)
        for entity in concepts:
            if not graph.has_node(entity):
                graph.add_node(entity, node_type="entity", entity_type="financial_concept", chunks=[])
            graph.nodes[entity]["chunks"].append(ref)
            _add_edge(graph, entity, evidence, "MENTIONS")
            for period in period_entities:
                periods.add(period)
                if not graph.has_node(period):
                    graph.add_node(period, node_type="reporting_period", chunks=[])
                graph.nodes[period]["chunks"].append(ref)
                _add_edge(graph, entity, period, "HAS_EVIDENCE_IN_PERIOD")
        for i, left in enumerate(concepts):
            for right in concepts[i + 1:]:
                _add_edge(graph, left, right, "CO_MENTIONED_WITHIN_CHUNK")
        for trend, pattern in TREND_PATTERNS.items():
            if pattern.search(text):
                trend_node = f"TREND:{trend}:{chunk_id}"
                graph.add_node(trend_node, node_type="qualitative_trend", direction=trend, verified=False)
                _add_edge(graph, trend_node, evidence, "SUPPORTED_BY")
    ordered = sorted(periods, key=lambda value: int(value[2:]))
    for previous, current in zip(ordered, ordered[1:]):
        _add_edge(graph, previous, current, "PRECEDES")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nx.node_link_data(graph, edges="edges"),
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    _load_graph_cached.cache_clear()
    return graph


@lru_cache(maxsize=4)
def _load_graph_cached(path_string, modified_ns):
    path = Path(path_string)
    return nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")), edges="edges") if path.exists() else nx.Graph()


def load_graph(path=INDEX_PATH):
    path = Path(path)
    if not path.exists():
        return nx.Graph()
    return _load_graph_cached(str(path.resolve()), path.stat().st_mtime_ns)


def graph_retrieve(query, top_k=5, path=INDEX_PATH):
    graph = load_graph(path)
    candidates = defaultdict(float)
    for entity in extract_entities(query):
        if entity not in graph:
            continue
        for ref in graph.nodes[entity].get("chunks", []):
            candidates[ref["chunk_id"]] += 1.0
        for neighbor in graph.neighbors(entity):
            node = graph.nodes[neighbor]
            relations = graph[entity][neighbor].get("relations", [])
            if node.get("node_type") == "evidence_chunk" and "MENTIONS" in relations:
                candidates[node["chunk_id"]] += 1.5
            elif node.get("node_type") == "entity" and "CO_MENTIONED_WITHIN_CHUNK" in relations:
                for ref in node.get("chunks", []):
                    candidates[ref["chunk_id"]] += 0.35
    return [{"id": key, "score": score, "method": "graph"}
            for key, score in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
