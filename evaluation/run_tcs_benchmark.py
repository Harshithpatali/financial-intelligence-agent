"""Compare production FTS5, graph-only and RRF retrieval; emit an answer audit packet."""
from __future__ import annotations
import hashlib,json,logging,os,re,time
from decimal import Decimal,InvalidOperation
from pathlib import Path
from backend.rag.chunker import load_documents,chunk_documents
from backend.rag.document_store import get_index
from backend.rag.graph_index import graph_retrieve
from backend.rag.rag_pipeline import answer_question
from evaluation.benchmark import evaluate_records
from evaluation.metrics import citation_coverage
from evaluation.traceability import page_for_excerpt,page_for_anchor

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"evaluation"/"tcs_benchmark.json"; OUT=ROOT/"evaluation"/"results"
TOP_K=5; CANDIDATE_K=40; LOG=logging.getLogger(__name__)
NUMBER_RE=re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?")
CITATION_RE=re.compile(r"(?:\[S(\d+)\]|【S(\d+)】)")
def normalize(value): return re.sub(r"\s+"," ",str(value)).strip().casefold()
def stable_id(item):
    return hashlib.sha1(f'{item["source"]}|{item.get("page")}|{item["chunk_id"]}|{item["text"]}'.encode()).hexdigest()
def resolve_rows(rows,chunks):
    resolved=[]
    for row in rows:
        anchor=normalize(row["anchor"])
        hits=[x for x in chunks if x["source"]==row["source"] and anchor in normalize(x["text"])]
        if not hits: raise RuntimeError(f"Reviewed anchor did not resolve for {row['id']}: {row['source']} :: {row['anchor']}")
        resolved.append({**row,"relevant_chunk_ids":[stable_id(x) for x in hits]})
    return resolved
def rrf(*rankings,limit=TOP_K):
    scores={}
    for ranking in rankings:
        for rank,key in enumerate(ranking,1): scores[key]=scores.get(key,0)+1/(60+rank)
    return [key for key,_ in sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:limit]]
def numbers(value):
    found=[]
    for token in NUMBER_RE.findall(str(value)):
        try: found.append(Decimal(token.replace(",","")))
        except InvalidOperation: pass
    return found
def expected_number_present(answer,expected,unit):
    target=numbers(expected); actual=numbers(answer)
    if not target: return False
    if target[0] in actual: return True
    if "crore" in unit.casefold() and re.search(r"\bmillions?\b",answer,re.I):
        return any(x/Decimal(10)==target[0] for x in actual)
    if "employees" in unit.casefold() and re.search(r"\b\d+(?:\.\d+)?\s*k\+?\b",answer,re.I):
        return any(x*1000==target[0] for x in actual)
    return False
def cited_indices(answer,n):
    result=[]
    for match in CITATION_RE.finditer(answer or ""):
        idx=int(match.group(1) or match.group(2))
        if 1<=idx<=n and idx not in result: result.append(idx)
    return result
def main():
    rows=json.loads(DATA.read_text(encoding="utf-8"))
    chunks=chunk_documents(load_documents(ROOT/"knowledge"))
    rows=resolve_rows(rows,chunks)
    index=get_index(); total=index.count()
    if total<=0: raise RuntimeError("Downloaded SQLite FTS5 index is empty.")
    methods={"fts5_only":[],"graph_only":[],"fts5_graph_rrf":[]}
    for n,row in enumerate(rows,1):
        fts=[x["id"] for x in index.search(row["question"],CANDIDATE_K)]
        graph=[x["id"] for x in graph_retrieve(row["question"],top_k=CANDIDATE_K)]
        rankings={"fts5_only":fts[:TOP_K],"graph_only":graph[:TOP_K],"fts5_graph_rrf":rrf(fts,graph)}
        for name,ids in rankings.items(): methods[name].append({**row,"retrieved_chunk_ids":ids})
        print(f"Retrieved {n}/{len(rows)}: {row['id']}")
    metrics={}
    for name,records in methods.items():
        result=evaluate_records([{"question":x["question"],"relevant_chunk_ids":x["relevant_chunk_ids"],"retrieved_chunk_ids":x["retrieved_chunk_ids"]} for x in records],k=TOP_K)
        metrics[name]={"mean_recall_at_5":result["mean_recall_at_k"],"mean_mrr_at_5":result["mean_mrr_at_k"],"per_question":result["per_question"]}
    OUT.mkdir(parents=True,exist_ok=True)
    summary={"benchmark":"TCS financial disclosures, source-anchored benchmark","question_count":len(rows),"k":TOP_K,
      "index":"SQLite FTS5 BM25 + NetworkX evidence graph","index_chunk_count":total,"methods":metrics,
      "limitations":["No dense retrieval is included: Groq embedding endpoint is unavailable and no third-party/local embedding model is used.","Metrics are retrieval relevance scores, not answer correctness or faithfulness.","Overlapping chunks containing the same anchor are all labelled relevant; alternative valid evidence not explicitly labelled is not counted."]}
    (OUT/"retrieval_comparison.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    audits=[]
    if os.getenv("GROQ_API_KEY"):
        for n,row in enumerate(rows,1):
            if n>1 and (n-1)%2==0: time.sleep(65)
            result=None; error=None
            for attempt in range(3):
                try: result=answer_question(row["question"],top_k=5); break
                except Exception as exc:
                    error=exc
                    if attempt<2:
                        delay=max(1,int(getattr(exc,"retry_after",0) or 20*(attempt+1)))
                        LOG.warning("Answer retry %s attempt %s: %s",row["id"],attempt+2,exc); time.sleep(delay)
            source_text=(ROOT/row["source"]).read_text(encoding="utf-8")
            if result is None:
                audits.append({"id":row["id"],"question":row["question"],"expected_answer":row["expected_value"],"expected_unit":row["unit"],"expected_source":row["source"],"expected_anchor":row["anchor"],"expected_page":page_for_anchor(source_text,row["anchor"]),"answer":None,"generation_error":f"{type(error).__name__}: {error}","sources":[],"cited_evidence":[],"citation_coverage":0.0,"expected_numeric_value_present":False,"human_answer_relevance_review":"not_reviewable_generation_failed","human_faithfulness_review":"not_reviewable_generation_failed","human_numerical_review":"not_reviewable_generation_failed","human_citation_correctness_review":"not_reviewable_generation_failed","human_page_traceability_review":"not_reviewable_generation_failed"})
                continue
            answer=result["answer"]; sources=result["sources"]; evidence=[]
            for idx in cited_indices(answer,len(sources)):
                source=sources[idx-1]; found=index.get([source["chunk_id"]]); excerpt=(found["documents"][0] if found["documents"] else "")[:1600]
                file_text=(ROOT/source.get("source","")).read_text(encoding="utf-8") if (ROOT/source.get("source","")).is_file() else ""
                page=source.get("page") or page_for_excerpt(file_text,excerpt)
                evidence.append({**source,"page":page,"excerpt":excerpt,"page_traceable":page is not None})
            audits.append({"id":row["id"],"question":row["question"],"expected_answer":row["expected_value"],"expected_unit":row["unit"],"expected_source":row["source"],"expected_anchor":row["anchor"],"expected_page":page_for_anchor(source_text,row["anchor"]),"answer":answer,"sources":sources,"cited_evidence":evidence,"citation_coverage":citation_coverage(answer,len(sources)),"cited_gold_evidence_present":any(x.get("chunk_id") in row["relevant_chunk_ids"] for x in evidence),"expected_numeric_value_present":expected_number_present(answer,row["expected_value"],row["unit"]),"expected_value_present_in_cited_evidence":any(numbers(row["expected_value"]) and numbers(row["expected_value"])[0] in numbers(x["excerpt"]) for x in evidence),"human_answer_relevance_review":"pending","human_faithfulness_review":"pending","human_numerical_review":"pending","human_citation_correctness_review":"pending","human_page_traceability_review":"pending"})
    else: print("GROQ_API_KEY missing; answer-generation audit skipped.")
    packet={"status":"machine_checks_complete_human_review_pending" if audits and not any("generation_error" in x for x in audits) else ("generation_completed_with_errors_human_review_pending" if audits else "skipped_groq_api_key_missing"),"answer_count":len(audits),"successful_answer_count":sum("generation_error" not in x for x in audits),"generation_error_count":sum("generation_error" in x for x in audits),"benchmark_question_count":len(rows),"answer_context_top_k":5,"review_rubric":{"answer_relevance":"pass / partial / fail: entity, metric, period and comparison match.","faithfulness":"pass / partial / fail: every factual claim is supported by its cited excerpt.","numerical_accuracy":"pass / partial / fail: value, currency, unit, sign, period and accounting basis match.","citation_correctness":"pass / partial / fail: each citation supports the adjacent claim.","page_traceability":"pass / partial / fail: source and page resolve to original disclosure."},"checks":"Automated citation and numeric checks are screening signals only. Human review remains pending for every generated answer.","answers":audits}
    (OUT/"answer_audit_packet.json").write_text(json.dumps(packet,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"question_count":len(rows),"retrieval_metrics":{k:{"recall_at_5":v["mean_recall_at_5"],"mrr_at_5":v["mean_mrr_at_5"]} for k,v in metrics.items()},"answer_audit_count":len(audits),"audit_status":packet["status"]},indent=2))
if __name__=="__main__": main()
