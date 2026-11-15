"""Local NetworkX entity graph; edges represent co-mentions in evidence chunks."""
import json,re
from collections import defaultdict
from pathlib import Path
import networkx as nx
INDEX_PATH=Path(__file__).resolve().parents[1]/"indexes"/"tcs_knowledge_graph.json"
ENTITY_ALIASES={"Tata Consultancy Services":["tata consultancy services","tcs"],"Revenue":["revenue","total income","operating revenue"],"Net Income":["net income","net profit","profit after tax","pat"],"Operating Margin":["operating margin","ebit margin"],"Cash Flow":["cash flow","operating cash flow","free cash flow"],"Capital Expenditure":["capital expenditure","capex"],"Debt":["total debt","borrowings","debt"],"Artificial Intelligence":["artificial intelligence","generative ai","genai"],"Cloud":["cloud","cloud services"],"Banking":["banking","financial services","bsfi"],"Employees":["employees","workforce","associates"],"Dividend":["dividend","dividends"],"Digital Services":["digital services","digital transformation"],"North America":["north america","united states"],"Europe":["europe","uk market"]}
YEAR_RE=re.compile(r"\b(?:FY\s*)?(20\d{2})(?:[-–](\d{2,4}))?\b",re.I)
def extract_entities(text):
 normalized=" "+re.sub(r"\s+"," ",text.lower())+" "
 found={entity for entity,aliases in ENTITY_ALIASES.items() if any((" "+alias+" ") in normalized for alias in aliases)}
 for match in YEAR_RE.finditer(text):
  found.add("FY"+match.group(1))
  if match.group(2):
   suffix=match.group(2);found.add("FY"+(suffix if len(suffix)==4 else match.group(1)[:2]+suffix))
 return sorted(found)
def build_graph(chunks,output_path=INDEX_PATH):
 graph=nx.Graph()
 for chunk in chunks:
  entities=extract_entities(chunk["text"]);ref={"chunk_id":str(chunk["id"]),"source":chunk["source"],"page":chunk.get("page"),"text":chunk["text"][:1200]}
  for entity in entities:
   if not graph.has_node(entity):graph.add_node(entity,entity_type="financial_concept" if entity in ENTITY_ALIASES else "reporting_period",chunks=[])
   graph.nodes[entity]["chunks"].append(ref)
  for i,left in enumerate(entities):
   for right in entities[i+1:]:
    if graph.has_edge(left,right):graph[left][right]["weight"]+=1
    else:graph.add_edge(left,right,relation="co_mentioned_in_source_chunk",weight=1)
 output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
 output_path.write_text(json.dumps(nx.node_link_data(graph),ensure_ascii=False,indent=2),encoding="utf-8")
 return graph
def load_graph(path=INDEX_PATH):
 path=Path(path)
 return nx.node_link_graph(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else nx.Graph()
def graph_retrieve(query,top_k=5,path=INDEX_PATH):
 graph=load_graph(path);candidates=defaultdict(float)
 for entity in extract_entities(query):
  if entity not in graph:continue
  for ref in graph.nodes[entity].get("chunks",[]):candidates[ref["chunk_id"]]+=1.0
  for neighbor in graph.neighbors(entity):
   for ref in graph.nodes[neighbor].get("chunks",[]):candidates[ref["chunk_id"]]+=0.35
 return [{"id":key,"score":score,"method":"graph"} for key,score in sorted(candidates.items(),key=lambda x:(-x[1],x[0]))[:top_k]]
