use rust_het_graph::load_graph;

#[tokio::main]
async fn main() {
    let (node_map, relation_map) = load_graph().await;
    for (_, node) in node_map {
        println!("{:?}",node.get_properties());
    }
    println!("load finish");
}
