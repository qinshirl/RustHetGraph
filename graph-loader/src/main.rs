use std::collections::HashMap;
use neo4rs::*;

struct MemNode {
    inner: BoltNode,
    relation_list: Vec<BoltRelation>,
}

impl MemNode {
    fn new(bolt_node: BoltNode) -> Self {
        MemNode { inner: bolt_node, relation_list: vec![] }
    }
    fn add_relation(&mut self, relation: BoltRelation) {
        self.relation_list.push(relation);
    }
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let uri = "127.0.0.1:7687";
    let user = std::env::var("NEO4J_USER").expect("NEO4J_USER env var not set");
    let pass = std::env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD env var not set");

    let graph = Graph::new(uri, user, pass).await?;
    let mut r = graph.execute(query("MATCH (n:Node) RETURN n, id(n) as id")).await?;
    let mut node_map:HashMap<i64, MemNode> = HashMap::new();
    while let Ok(Some(row)) = r.next().await{
        let node: BoltNode = row.get("n").unwrap();
        let id: i64 = row.get("id").unwrap();
        node_map.insert(id, MemNode::new(node));
    }
    let mut r = graph.execute(query("MATCH p=()-[r:Distance]->() RETURN r")).await?;
    let mut relation_map:HashMap<i64, BoltRelation> = HashMap::new();
    while let Ok(Some(row)) = r.next().await{
        let relation: BoltRelation = row.get("r").unwrap();
        let r_id:i64 = i64::from(relation.id.clone());
        relation_map.insert(r_id, relation);
    }

    for (_, relation) in relation_map{
        let start_id:i64 = i64::from(relation.start_node_id.clone());
        node_map.get_mut(&start_id).unwrap().add_relation(relation);
    }

    println!("load finish");
    Ok(())
}