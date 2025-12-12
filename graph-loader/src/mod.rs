use neo4rs::*;
use petgraph::csr::Csr;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Directed;
use std::collections::HashMap;

/// load a graph from neo4j database into memory,
/// returning a directed graph with native neo4j representation.
pub async fn load_graph_from_neo4j() -> DiGraph<BoltNode, BoltRelation> {
    let graph = get_neo4j_graph();
    // petgraph
    let mut pet_graph = DiGraph::<BoltNode, BoltRelation>::new();
    let mut id2idx: HashMap<i64, NodeIndex> = HashMap::new();

    let mut r = graph.execute(query("MATCH (n:Node) RETURN n, n.id as id")).await.unwrap();
    while let Ok(Some(row)) = r.next().await{
        let node: BoltNode = row.get("n").unwrap();
        let id: i64 = row.get("id").unwrap();
        let idx = pet_graph.add_node(node);
        id2idx.insert(id, idx);
    }
    let mut r = graph.execute(query("MATCH p=(a)-[r:LINK]->(b) RETURN r,a.id as sid,b.id as tid")).await.unwrap();
    while let Ok(Some(row)) = r.next().await{
        let rel: BoltRelation = row.get("r").unwrap();
        let src_id = row.get("sid").unwrap();
        let dst_id = row.get("tid").unwrap();
        if let (Some(&src_idx), Some(&dst_idx)) = (id2idx.get(&src_id), id2idx.get(&dst_id)) {
            pet_graph.add_edge(src_idx, dst_idx, rel);
        }
    }
    pet_graph
}

/// establish a connection to neo4j and return a graph.
fn get_neo4j_graph() -> Graph {
    let uri = "127.0.0.1:7687";
    let user = std::env::var("NEO4J_USER").expect("NEO4J_USER env var not set");
    let pass = std::env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD env var not set");
    // neo4j graph
    let graph = Graph::new(uri, user, pass).unwrap();
    graph
}

/// convert a native neo4j graph to number formatted direct petgraph.
pub fn convert_graph_2_index_weight(graph: DiGraph<BoltNode, BoltRelation>) -> DiGraph<i64, i64> {
    let mut result = DiGraph::<i64, i64>::new();
    let idx_map: Vec<_> = graph
        .node_indices()
        .map(|node_index| (node_index, result
            .add_node(graph[node_index].get("id").unwrap())))
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

/// convert a number formatted adjacent list to csr representation.
pub fn convert_adjacent_list_2_csr(graph: DiGraph<i64,i64>) -> Csr<i64, i64, Directed, NodeIndex> {
    let mut result = Csr::new();
    for node_idx in graph.node_indices() {
        result.add_node(graph[node_idx]);
    }
    for edge_idx in graph.edge_indices() {
        let (s, t) = graph.edge_endpoints(edge_idx).unwrap();
        let weight = graph[edge_idx];
        result.add_edge(s,t,weight);
    }
    result

}

/// delete all data in neo4j, use with cautious.
pub async fn delete_graph(){
    let graph = get_neo4j_graph();
    let mut txn = graph.start_txn().await.unwrap();
    txn.run(query("MATCH (n) DETACH DELETE n")).await.unwrap();
    txn.commit().await.unwrap();
}

/// load csv into neo4j database, the file should under import folder of neo4j home directory.
pub async fn load_graph_to_neo4j(node_filename: &str, edge_filename: &str){
    let graph = get_neo4j_graph();
    let mut txn = graph.start_txn().await.unwrap();
    let cypher = format!(
        "LOAD CSV WITH HEADERS FROM 'file:///{}' AS row \
     CREATE (n:Node {{id: toInteger(row.`nodeId:ID`)}})",
        node_filename
    );
    txn.run(query(&cypher)).await.expect("load nodes failed");
    let cypher = format!(
        "LOAD CSV WITH HEADERS FROM 'file:///{}' AS row
         MATCH (a:Node {{id: toInteger(row.`:START_ID`)}})
         MATCH (b:Node {{id: toInteger(row.`:END_ID`)}})
         CREATE (a)-[r:LINK {{weight: toInteger(row.`weight:int`)}}]->(b)",
        edge_filename
    );
    txn.run(query(&cypher)).await.expect("load edges failed");
    txn.commit().await.unwrap();
}