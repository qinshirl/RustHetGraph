use petgraph::dot::Dot;
use petgraph::algo::dijkstra;
use rust_het_graph::{convert_adjacent_list_2_csr, convert_graph_2_index_weight, delete_graph, load_graph_from_neo4j, load_graph_to_neo4j, reorder_graph};
use clap::{Parser, Subcommand};

#[derive(Subcommand, Debug)]
enum Commands {
    /// load graph to neo4j
    Load(LoadArgs),
    /// delete all data in neo4j
    Delete,
    /// display graph in neo4j
    Display,
    /// run sssp
    Sssp(SsspArgs),
    /// reorder a graph
    Reorder,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args{
    /// Command
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(clap::Args, Debug)]
struct LoadArgs{
    /// node file in csv format in the neo4j import folder
    #[arg(short='n', long="nodefile")]
    node_file: String,

    /// edge file in csv format in the neo4j import folder
    #[arg(short='e', long="edgefile")]
    edge_file: String,
}

#[derive(clap::Args, Debug)]
struct SsspArgs{
    /// start node index
    #[arg(short='s', long="start")]
    start: usize,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let cmd = args.cmd;
    match cmd {
        Commands::Load(load_args) => {
            let node_file = load_args.node_file.as_str();
            let edge_file = load_args.edge_file.as_str();
            load_graph_to_neo4j(node_file,edge_file).await;
        },
        Commands::Delete => {delete_graph().await;},
        Commands::Display => {
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
        Commands::Sssp(sssp_args) => {
            let graph = load_graph_from_neo4j().await;
            let graph = convert_graph_2_index_weight(graph);
            let idx = graph.node_indices().nth(sssp_args.start).unwrap();
            let cost_map = dijkstra(&graph, idx, None, |e| *e.weight());
            println!("{:?}", &cost_map);
        },
        Commands::Reorder => {
            let graph = load_graph_from_neo4j().await;
            let graph = convert_graph_2_index_weight(graph);
            let new_order = reorder_graph(&graph).await;
            println!("{:?}", new_order);
        }
    }
}
