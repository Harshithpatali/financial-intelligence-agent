from backend.rag.graph_index import extract_entities,build_graph,graph_retrieve
def test_entity_extraction():
 entities=extract_entities("TCS revenue and operating margin in FY2024")
 assert {"Revenue","Operating Margin","FY2024"}.issubset(set(entities))
def test_graph_build_and_query(tmp_path):
 path=tmp_path/"graph.json"
 chunks=[{"id":"c1","source":"annual.md","page":3,"text":"TCS revenue grew in FY2024 with cloud services."},{"id":"c2","source":"annual.md","page":4,"text":"TCS reported operating margin in FY2024."}]
 graph=build_graph(chunks,path)
 assert graph.number_of_nodes()>0 and graph.number_of_edges()>0
 assert any(data.get("relation")=="MENTIONS" for _,_,data in graph.edges(data=True))
 assert graph_retrieve("TCS revenue FY2024",path=path)
