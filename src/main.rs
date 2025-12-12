use rust_het_graph::{convert_adjacent_list_2_csr, convert_graph_2_index_weight, load_graph};
use petgraph::dot::Dot;
use petgraph::graph;

#[tokio::main]
async fn main() {
    let graph = load_graph().await;
    println!("{:?}", graph);
    println!("load finish");
    let graph = convert_graph_2_index_weight(graph);
    println!("{:?}", graph);
    println!("{}",Dot::new(&graph));
    let graph = convert_adjacent_list_2_csr(graph);
    println!("{:?}", graph);
    println!("{}",Dot::new(&graph));
}
