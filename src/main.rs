use petgraph::dot::Dot;
use rust_het_graph::{convert_adjacent_list_2_csr, convert_graph_2_index_weight, delete_graph, load_graph_from_neo4j, load_graph_to_neo4j};

#[tokio::main]
async fn main() {
    delete_graph().await;
    load_graph_to_neo4j("nodes-simple.csv","edges-simple.csv").await;
    let graph = load_graph_from_neo4j().await;
    println!("neo4j graph loaded");
    println!("{:?}", graph);

    let graph = convert_graph_2_index_weight(graph);
    println!("petgraph adjacent list loaded");
    println!("{:?}", graph);
    println!("Dot representations");
    println!("{}",Dot::new(&graph));
    
    let graph = convert_adjacent_list_2_csr(graph);
    println!("csr representations");
    println!("{:?}", graph);
    println!("Dot representations");
    println!("{}",Dot::new(&graph));
    
}
