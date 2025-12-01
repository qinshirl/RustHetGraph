use rust_het_graph::{convert_graph_2_index_weight, load_graph};

#[tokio::main]
async fn main() {
    let graph = load_graph().await;
    println!("{:?}", graph);
    println!("load finish");
    let graph = convert_graph_2_index_weight(graph);
    println!("{:?}", graph);
}
