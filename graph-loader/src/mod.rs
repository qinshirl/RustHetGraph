use std::collections::HashMap;
use neo4rs::*;
use petgraph::graph::{DiGraph, NodeIndex};

pub async fn load_graph() -> DiGraph<BoltNode, BoltRelation> {
    let uri = "127.0.0.1:7687";
    let user = std::env::var("NEO4J_USER").expect("NEO4J_USER env var not set");
    let pass = std::env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD env var not set");
    // neo4j graph
    let graph = Graph::new(uri, user, pass).unwrap();
    // petgraph
    let mut pet_graph = DiGraph::<BoltNode, BoltRelation>::new();
    let mut id2idx: HashMap<i64, NodeIndex> = HashMap::new();

    let mut r = graph.execute(query("MATCH (n:Node) RETURN n, id(n) as id")).await.unwrap();
    while let Ok(Some(row)) = r.next().await{
        let node: BoltNode = row.get("n").unwrap();
        let id: i64 = row.get("id").unwrap();
        let idx = pet_graph.add_node(node);
        id2idx.insert(id, idx);
    }
    let mut r = graph.execute(query("MATCH p=()-[r:LINK]->() RETURN r")).await.unwrap();
    while let Ok(Some(row)) = r.next().await{
        let rel: BoltRelation = row.get("r").unwrap();
        let src_id = rel.start_node_id.value;
        let dst_id = rel.end_node_id.value;
        if let (Some(&src_idx), Some(&dst_idx)) = (id2idx.get(&src_id), id2idx.get(&dst_id)) {
            pet_graph.add_edge(src_idx, dst_idx, rel);
        }
    }
    pet_graph
}

pub fn convert_graph_2_index_weight(graph: DiGraph<BoltNode, BoltRelation>) -> DiGraph<i64, i64> {
    let mut result = DiGraph::<i64, i64>::new();
    let idx_map: Vec<_> = graph
        .node_indices()
        .map(|node_index| (node_index, result.add_node(graph[node_index].id.value)))
        .collect();

    for edge_index in graph.edge_indices() {
        let (s, t) = graph.edge_endpoints(edge_index).unwrap();
        let weight = graph[edge_index]
            .properties
            .get("weight").unwrap();
        result.add_edge(idx_map[s.index()].1, idx_map[t.index()].1, weight);
    }
    result
}
