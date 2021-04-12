use std::collections::LinkedList;
use std::io::{self, Write};

pub mod graph;

use graph::*;

fn read_input() -> String {
  io::stdout().flush().expect("flush failed");

  let mut buffer = String::new();

  io::stdin()
    .read_line(&mut buffer)
    .expect("unable to read input");

  buffer
}

fn main() {
  let commands = r#"
    help => shows list of commands
    graph.new => creates a new graph
    vertex.add <vertex> => adds a vertex to the graph
    vertex.remove <vertex> => removes a vertex from the graph
    edge.directed.add <from vertex> <to vertex> <identifier> <weight> => adds a directed edge
    edge.undirected.add <from vertex> <to vertex> <identifier> <weight> => adds an undirected edge
    edge.remove <vertex> <vertex> => removes edge between vertices
    prim <vertex> => shows primm's minimum spanning tree starting from vertex
    dfs <vertex> => shows depth first search path starting from vertex
    bfs <vertex> => shows breadth first search path starting from vertex
    roy <vertex> => shows graph strongly connected components starting from vertex
  "#;

  let mut current_graph: Graph<String> = graph::Graph::new();

  println!("{:#?}", &current_graph);

  loop {
    print!("> ");

    let buffer = read_input().replace("\n", "");
    let mut input: LinkedList<&str> = buffer.split(" ").collect();

    match input.pop_front().unwrap() {
      "graph.new" => {
        current_graph = graph::Graph::new();
        println!("{:#?}", &current_graph);
      }
      "vertex.add" => match input.pop_back() {
        None => println!("a vertex must be informed"),
        Some(vertex) => match current_graph.add_vertex(vertex.to_owned()) {
          Err(DuplicatedVertexError(vertex)) => {
            println!("vertex {} is already in the graph", vertex)
          }
          Ok(()) => {
            println!("{:#?}", &current_graph);
          }
        },
      },
      "vertex.remove" => match input.pop_back() {
        None => println!("a vertex must be informed"),
        Some(vertex) => match current_graph.remove_vertex(&vertex.to_owned()) {
          Err(VertexNotFoundError(vertex)) => println!("vertex {} is not in the graph", vertex),
          Ok(()) => println!("{:#?}", &current_graph),
        },
      },
      "edge.directed.add" => match (
        input.pop_back(),
        input.pop_back(),
        input.pop_back(),
        input.pop_back(),
      ) {
        (Some(weight), Some(identifier), Some(to), Some(from)) => match weight.parse::<i32>() {
          Err(_) => println!("weight {} must be a number", weight),
          Ok(weight) => {
            match current_graph.add_directed_edge(
              from.to_owned(),
              to.to_owned(),
              identifier.to_owned(),
              weight,
            ) {
              Err(VertexNotFoundError(vertex)) => println!("vertex {} is not in the graph", vertex),
              Ok(()) => println!("{:#?}", &current_graph),
            }
          }
        },
        _ => println!("missing arguments, type help to see a list of commands"),
      },
      "edge.undirected.add" => match (
        input.pop_back(),
        input.pop_back(),
        input.pop_back(),
        input.pop_back(),
      ) {
        (Some(weight), Some(identifier), Some(to), Some(from)) => match weight.parse::<i32>() {
          Err(_) => println!("weight {} must be a number", weight),
          Ok(weight) => {
            match current_graph.add_undirected_edge(
              from.to_owned(),
              to.to_owned(),
              identifier.to_owned(),
              weight,
            ) {
              Err(VertexNotFoundError(vertex)) => println!("vertex {} is not in the graph", vertex),
              Ok(()) => println!("{:#?}", &current_graph),
            }
          }
        },
        _ => println!("missing arguments, type help to see a list of commands"),
      },
      "edge.remove" => match (input.pop_back(), input.pop_back()) {
        (Some(vertex_a), Some(vertex_b)) => {
          current_graph.remove_edge(&vertex_a.to_owned(), &vertex_b.to_owned());
          println!("{:#?}", &current_graph);
        }
        _ => println!("missing arguments, type help to see a list of commands"),
      },
      "prim" => match input.pop_back() {
        None => println!("a starting vertex must be informed"),
        Some(vertex) => {
          match current_graph.minimum_spanning_tree_starting_from_vertex(&vertex.to_owned()) {
            Err(VertexNotFoundError(vertex)) => {
              println!("vertex {} is not in the graph", vertex)
            }
            Ok(tree) => println!("{:#?}", tree),
          }
        }
      },
      "dfs" => match input.pop_back() {
        None => println!("a starting vertex must be informed"),
        Some(vertex) => match current_graph.dfs_path(&vertex.to_owned()) {
          Err(VertexNotFoundError(vertex)) => {
            println!("vertex {} is not in the graph", vertex)
          }
          Ok(path) => println!("{:#?}", path),
        },
      },
      "bfs" => match input.pop_back() {
        None => println!("a starting vertex must be informed"),
        Some(vertex) => match current_graph.bfs_path(&vertex.to_owned()) {
          Err(VertexNotFoundError(vertex)) => {
            println!("vertex {} is not in the graph", vertex)
          }
          Ok(path) => println!("{:#?}", path),
        },
      },
      "roy" => match input.pop_back() {
        None => println!("a starting vertex must be informed"),
        Some(vertex) => match current_graph.strongly_connected_components(&vertex.to_owned()) {
          Err(VertexNotFoundError(vertex)) => {
            println!("vertex {} is not in the graph", vertex)
          }
          Ok(components) => println!("{:#?}", components),
        },
      },
      "help" => println!("{}", commands),
      command => println!(
        "unknown command {}\nhint: type help to see a list of commands",
        command
      ),
    }
  }
}
