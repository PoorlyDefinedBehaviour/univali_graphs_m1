use std::cmp::{Eq, Ordering};
use std::collections::{BinaryHeap, HashMap, HashSet, LinkedList};
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Debug, PartialEq, Clone)]
pub enum Edge<T> {
  Directed {
    vertex: T,
    weight: i32,
    identifier: String,
  },
  Undirected {
    vertex: T,
    weight: i32,
    identifier: String,
  },
}

fn edge_vertex<T: PartialEq>(edge: &Edge<T>) -> &T {
  match edge {
    Edge::Directed { vertex, .. } => vertex,
    Edge::Undirected { vertex, .. } => vertex,
  }
}

fn edge_weight<T: PartialEq>(edge: &Edge<T>) -> i32 {
  match edge {
    Edge::Undirected { weight, .. } => *weight,
    Edge::Directed { weight, .. } => *weight,
  }
}

pub type AdjencyList<KEY, VALUE> = HashMap<KEY, VALUE>;

#[derive(Debug, Default)]
pub struct Graph<T: PartialEq + Debug> {
  pub adjacency_list: AdjencyList<T, Vec<Edge<T>>>,
}

#[derive(Debug, PartialEq)]
pub struct DuplicatedVertexError<T>(pub T);

#[derive(Debug, PartialEq)]
pub struct VertexNotFoundError<T>(pub T);

#[derive(Debug, Eq, PartialEq)]
pub struct MinimumSpanningTreeEdge<T> {
  from: T,
  to: T,
  cost: i32,
}

impl<T: PartialEq + Eq> Ord for MinimumSpanningTreeEdge<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    other.cost.cmp(&self.cost)
  }
}

impl<T: PartialEq + Eq> PartialOrd for MinimumSpanningTreeEdge<T> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl<T: Eq + Hash + Clone + Debug> Graph<T> {
  pub fn new() -> Self {
    Graph {
      adjacency_list: HashMap::new(),
    }
  }

  pub fn add_vertex(&mut self, vertex: T) -> Result<(), DuplicatedVertexError<T>> {
    match self.adjacency_list.get(&vertex) {
      Some(_) => Err(DuplicatedVertexError(vertex)),
      None => {
        self.adjacency_list.insert(vertex, vec![]);

        Ok(())
      }
    }
  }

  pub fn remove_vertex(&mut self, vertex: &T) -> Result<(), VertexNotFoundError<T>> {
    match self.adjacency_list.get(vertex) {
      None => Err(VertexNotFoundError(vertex.clone())),
      Some(_) => {
        self.adjacency_list.remove(&vertex);

        Ok(())
      }
    }
  }

  fn add_edge_to_vertex(&mut self, vertex: T, edge: Edge<T>) -> Result<(), VertexNotFoundError<T>> {
    match self.adjacency_list.get_mut(&vertex) {
      None => Err(VertexNotFoundError(vertex)),
      Some(neighbors) => {
        neighbors.push(edge);

        Ok(())
      }
    }
  }

  pub fn add_undirected_edge(
    &mut self,
    vertex_a: T,
    vertex_b: T,
    identifier: String,
    weight: i32,
  ) -> Result<(), VertexNotFoundError<T>> {
    self.add_edge_to_vertex(
      vertex_a.clone(),
      Edge::Undirected {
        vertex: vertex_b.clone(),
        identifier: identifier.clone(),
        weight,
      },
    )?;

    if vertex_a != vertex_b {
      self.add_edge_to_vertex(
        vertex_b,
        Edge::Undirected {
          vertex: vertex_a,
          identifier,
          weight,
        },
      )?;
    }

    Ok(())
  }

  pub fn add_directed_edge(
    &mut self,
    vertex_a: T,
    vertex_b: T,
    identifier: String,
    weight: i32,
  ) -> Result<(), VertexNotFoundError<T>> {
    if !self.adjacency_list.contains_key(&vertex_b) {
      Err(VertexNotFoundError(vertex_b))
    } else {
      self.add_edge_to_vertex(
        vertex_a,
        Edge::Directed {
          vertex: vertex_b,
          identifier,
          weight,
        },
      )
    }
  }

  pub fn remove_edge(&mut self, vertex_a: &T, vertex_b: &T) {
    if let Some(neighbors) = self.adjacency_list.remove(vertex_a) {
      let new_neighbors: Vec<Edge<T>> = neighbors
        .into_iter()
        .filter(|neighbor| edge_vertex(neighbor) != vertex_b)
        .collect();
      self.adjacency_list.insert(vertex_a.clone(), new_neighbors);
    }

    if let Some(neighbors) = self.adjacency_list.remove(vertex_b) {
      let new_neighbors: Vec<Edge<T>> = neighbors
        .into_iter()
        .filter(|neighbor| edge_vertex(neighbor) != vertex_a)
        .collect();
      self.adjacency_list.insert(vertex_b.clone(), new_neighbors);
    }
  }

  fn add_neighbors_to_priority_queue(
    &self,
    vertex: &T,
    visited_vertices: &HashSet<T>,
    priority_queue: &mut BinaryHeap<MinimumSpanningTreeEdge<T>>,
  ) {
    for neighbor in self.adjacency_list.get(vertex).unwrap() {
      if visited_vertices.contains(edge_vertex(neighbor)) {
        continue;
      }

      priority_queue.push(MinimumSpanningTreeEdge {
        from: vertex.clone(),
        to: edge_vertex(neighbor).clone(),
        cost: edge_weight(neighbor),
      });
    }
  }

  pub fn minimum_spanning_tree_starting_from_vertex(
    &self,
    starting_vertex: &T,
  ) -> Result<Vec<MinimumSpanningTreeEdge<T>>, VertexNotFoundError<T>> {
    if !self.adjacency_list.contains_key(starting_vertex) {
      return Err(VertexNotFoundError(starting_vertex.clone()));
    }

    let mut priority_queue = BinaryHeap::new();

    let mut visited_vertices = HashSet::new();

    visited_vertices.insert(starting_vertex.clone());

    self.add_neighbors_to_priority_queue(starting_vertex, &visited_vertices, &mut priority_queue);

    let mut minimum_spanning_tree = Vec::new();
    let mut visited_edges = 0;

    while visited_edges != self.adjacency_list.len() - 1 && !priority_queue.is_empty() {
      let cheapest_path = priority_queue.pop().unwrap();

      if visited_vertices.contains(&cheapest_path.to) {
        continue;
      }

      visited_vertices.insert(cheapest_path.to.clone());

      self.add_neighbors_to_priority_queue(
        &cheapest_path.to,
        &visited_vertices,
        &mut priority_queue,
      );

      visited_edges += 1;

      minimum_spanning_tree.push(cheapest_path);
    }

    Ok(minimum_spanning_tree)
  }

  pub fn dfs_path(&self, starting_vertex: &T) -> Result<Vec<T>, VertexNotFoundError<T>> {
    if !self.adjacency_list.contains_key(starting_vertex) {
      return Err(VertexNotFoundError(starting_vertex.clone()));
    }

    let mut stack = LinkedList::new();
    let mut path = Vec::new();
    let mut visited_nodes = HashSet::new();

    // O(1)
    stack.push_back(starting_vertex.clone());

    while !stack.is_empty() {
      // O(1)
      let current_vertex = stack.pop_back().unwrap();

      // O(1)
      if !visited_nodes.contains(&current_vertex) {
        path.push(current_vertex.clone());
      }

      // O(1)~*
      visited_nodes.insert(current_vertex.clone());

      for neighbor in self.adjacency_list.get(&current_vertex).unwrap() {
        // O(1)
        if !visited_nodes.contains(edge_vertex(neighbor)) {
          // O(1)
          stack.push_back(edge_vertex(neighbor).clone());
        }
      }
    }

    Ok(path)
  }

  pub fn bfs_path(&self, starting_vertex: &T) -> Result<Vec<T>, VertexNotFoundError<T>> {
    if !self.adjacency_list.contains_key(starting_vertex) {
      return Err(VertexNotFoundError(starting_vertex.clone()));
    }

    let mut stack = LinkedList::new();
    let mut path = Vec::new();
    let mut visited_nodes = HashSet::new();

    // O(1)
    stack.push_back(starting_vertex.clone());

    while !stack.is_empty() {
      // O(1)
      let current_vertex = stack.pop_front().unwrap();

      // O(1)
      if !visited_nodes.contains(&current_vertex) {
        path.push(current_vertex.clone());
      }

      // O(1)~*
      visited_nodes.insert(current_vertex.clone());

      for neighbor in self.adjacency_list.get(&current_vertex).unwrap() {
        // O(1)
        if !visited_nodes.contains(edge_vertex(neighbor)) {
          // O(1)
          stack.push_back(edge_vertex(neighbor).clone());
        }
      }
    }

    Ok(path)
  }

  fn is_successor(&self, vertex_a: &T, vertex_b: &T) -> bool {
    if vertex_a == vertex_b {
      return false;
    }

    match self.adjacency_list.get(vertex_a) {
      None => false,
      Some(neighbors) => neighbors
        .iter()
        .any(|neighbor| edge_vertex(neighbor) == vertex_b),
    }
  }

  pub fn successors(&self, current_vertex: &T) -> Vec<T> {
    self
      .adjacency_list
      .get(current_vertex)
      .cloned()
      .unwrap()
      .iter()
      .map(|edge| edge_vertex(edge).clone())
      .collect()
  }

  pub fn predecessors(&self, current_vertex: &T) -> Vec<T> {
    self
      .adjacency_list
      .keys()
      .filter_map(|vertex| {
        if self.is_successor(vertex, current_vertex) {
          Some(vertex.clone())
        } else {
          None
        }
      })
      .collect()
  }

  fn mark_plus_vertices(&self, vertex: &T, pluses: &mut HashSet<T>) {
    pluses.insert(vertex.clone());

    let mut visited_vertices: HashSet<T> = HashSet::new();

    fn go<T: Eq + Hash + Clone + Debug>(
      graph: &Graph<T>,
      vertex: &T,
      pluses: &mut HashSet<T>,
      visited_vertices: &mut HashSet<T>,
    ) {
      if visited_vertices.contains(vertex) {
        return;
      }

      visited_vertices.insert(vertex.clone());

      for successor in graph.successors(vertex) {
        go(graph, &successor, pluses, visited_vertices);
      }

      if graph
        .successors(vertex)
        .iter()
        .any(|successor| pluses.contains(successor))
      {
        pluses.insert(vertex.clone());
      }
    }

    go(&self, vertex, pluses, &mut visited_vertices);
  }

  fn mark_minus_vertices(&self, vertex: &T, minuses: &mut HashSet<T>) {
    minuses.insert(vertex.clone());

    let mut visited_vertices: HashSet<T> = HashSet::new();

    fn go<T: Eq + Hash + Clone + Debug>(
      graph: &Graph<T>,
      vertex: &T,
      minuses: &mut HashSet<T>,
      visited_vertices: &mut HashSet<T>,
    ) {
      if visited_vertices.contains(vertex) {
        return;
      }

      visited_vertices.insert(vertex.clone());

      for predecessor in graph.predecessors(vertex) {
        go(graph, &predecessor, minuses, visited_vertices);
      }

      if graph
        .predecessors(vertex)
        .iter()
        .any(|predecessor| minuses.contains(predecessor))
      {
        minuses.insert(vertex.clone());
      }
    }

    go(&self, vertex, minuses, &mut visited_vertices);
  }

  pub fn strongly_connected_components(
    &self,
    starting_vertex: &T,
  ) -> Result<Vec<Vec<T>>, VertexNotFoundError<T>> {
    if !self.adjacency_list.contains_key(starting_vertex) {
      return Err(VertexNotFoundError(starting_vertex.clone()));
    }

    let mut components = vec![];

    let mut vertices: Vec<T> = self.adjacency_list.keys().cloned().collect();
    let mut pluses = HashSet::new();
    let mut minuses = HashSet::new();

    let mut current_vertex = starting_vertex.clone();

    loop {
      self.mark_plus_vertices(&current_vertex, &mut pluses);
      self.mark_minus_vertices(&current_vertex, &mut minuses);

      let (marked_vertices, unmarked_vertices) = vertices
        .into_iter()
        .partition(|vertex| pluses.contains(vertex) && minuses.contains(vertex));

      components.push(marked_vertices);

      pluses.clear();

      minuses.clear();

      vertices = unmarked_vertices;

      if !vertices.is_empty() {
        current_vertex = vertices.last().cloned().unwrap();
      } else {
        return Ok(components);
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  macro_rules! hashmap {
    ($($key: expr => $value: expr), *) => {{
        let mut map = HashMap::new();
        $(map.insert($key, $value);)*
        map
    }};
  }

  #[test]
  fn adds_vertex_to_graph() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();

    let expected = hashmap! {
      1 => vec![]
    };

    assert_eq!(graph.adjacency_list, expected);
  }

  #[test]
  fn add_vertex_returns_error_when_vertex_is_already_in_the_graph() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();

    let expected = Err(DuplicatedVertexError(1));

    let actual = graph.add_vertex(1);

    assert_eq!(actual, expected);
  }

  #[test]
  fn removes_vertex_from_graph() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();
    graph.add_vertex(3).unwrap();

    graph.remove_vertex(&2).unwrap();

    let expected = hashmap! {
      1 => vec![],
      3 => vec![]
    };

    assert_eq!(graph.adjacency_list, expected);
  }

  #[test]
  fn remove_vertex_returns_error_when_vertex_doesnt_exist() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();
    graph.add_vertex(3).unwrap();

    let expected = Err(VertexNotFoundError(5));

    let actual = graph.remove_vertex(&5);

    assert_eq!(actual, expected);
  }

  #[test]
  fn adds_undirected_edge_to_graph() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();

    graph.add_undirected_edge(1, 2, "a".to_owned(), 2).unwrap();

    let expected = hashmap! {
      1 => vec![Edge::Undirected{
        vertex: 2,
        weight: 2,
        identifier: "a".to_owned(),
      }],
      2 => vec![Edge::Undirected{
        vertex: 1,
        weight: 2,
        identifier: "a".to_owned(),
      }]
    };

    assert_eq!(graph.adjacency_list, expected);
  }

  #[test]
  fn adds_undirected_edge_to_graph_without_duplication_when_edge_is_a_cycle() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();

    graph.add_undirected_edge(1, 1, "a".to_owned(), 2).unwrap();

    let expected = hashmap! {
      1 => vec![Edge::Undirected{
        vertex: 1,
        weight: 2,
        identifier: "a".to_owned(),
      }]
    };

    assert_eq!(graph.adjacency_list, expected);
  }

  #[test]
  fn add_undirected_edge_returns_error_when_a_vertex_is_not_in_the_graph() {
    let mut graph = Graph::new();

    graph.add_vertex(2).unwrap();

    let expected = Err(VertexNotFoundError(1));

    let actual = graph.add_undirected_edge(1, 2, "b".to_owned(), 0);

    assert_eq!(actual, expected);
  }

  #[test]
  fn adds_directed_edge_to_graph() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();

    graph.add_directed_edge(1, 2, "a".to_owned(), 2).unwrap();

    let expected = hashmap! {
      1 => vec![Edge::Directed{
        vertex: 2,
        weight: 2,
        identifier: "a".to_owned(),
      }],
      2 => vec![]
    };

    assert_eq!(graph.adjacency_list, expected);
  }

  #[test]
  fn adds_directed_edge_to_graph_without_duplication_when_edge_is_a_cycle() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();

    graph.add_directed_edge(1, 1, "a".to_owned(), 2).unwrap();
    graph.add_directed_edge(2, 2, "b".to_owned(), 2).unwrap();

    let expected = hashmap! {
      1 => vec![Edge::Directed{
        vertex: 1,
        weight: 2,
        identifier: "a".to_owned(),
      }],
      2 => vec![Edge::Directed{
        vertex: 2,
        weight: 2,
        identifier: "b".to_owned(),
      }]
    };

    assert_eq!(graph.adjacency_list, expected);
  }

  #[test]
  fn add_directed_edge_returns_error_when_a_vertex_is_not_in_the_graph() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();

    let expected = Err(VertexNotFoundError(3));

    let actual = graph.add_directed_edge(1, 3, "a".to_owned(), 2);

    assert_eq!(actual, expected);
  }

  #[test]
  fn removes_edge_from_graph() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();

    graph.add_directed_edge(1, 2, "a".to_owned(), 2).unwrap();

    let expected = hashmap! {
      1 => vec![],
      2 => vec![]
    };

    graph.remove_edge(&1, &2);

    assert_eq!(graph.adjacency_list, expected);
  }

  #[test]
  fn returns_primms_minimum_spanning_tree() {
    let mut graph = Graph::new();

    graph.add_vertex(0).unwrap();
    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();
    graph.add_vertex(3).unwrap();
    graph.add_vertex(4).unwrap();

    graph.add_undirected_edge(0, 1, "a".to_owned(), 2).unwrap();
    graph.add_undirected_edge(0, 3, "b".to_owned(), 6).unwrap();

    graph.add_undirected_edge(1, 2, "c".to_owned(), 3).unwrap();
    graph.add_undirected_edge(1, 3, "d".to_owned(), 8).unwrap();
    graph.add_undirected_edge(1, 4, "e".to_owned(), 5).unwrap();

    graph.add_undirected_edge(2, 4, "f".to_owned(), 7).unwrap();

    graph.add_undirected_edge(3, 4, "g".to_owned(), 9).unwrap();

    let expected = Ok(vec![
      MinimumSpanningTreeEdge {
        from: 0,
        to: 1,
        cost: 2,
      },
      MinimumSpanningTreeEdge {
        from: 1,
        to: 2,
        cost: 3,
      },
      MinimumSpanningTreeEdge {
        from: 1,
        to: 4,
        cost: 5,
      },
      MinimumSpanningTreeEdge {
        from: 0,
        to: 3,
        cost: 6,
      },
    ]);

    let actual = graph.minimum_spanning_tree_starting_from_vertex(&0);

    assert_eq!(actual, expected);
  }

  #[test]
  fn minimum_spanning_tree_starting_from_vertex_returns_error_when_starting_vertex_is_not_in_the_graph(
  ) {
    let graph = Graph::new();

    let expected = Err(VertexNotFoundError(0));

    let actual = graph.minimum_spanning_tree_starting_from_vertex(&0);

    assert_eq!(actual, expected)
  }

  #[test]
  fn returns_dfs_path() {
    let mut graph = Graph::new();

    graph.add_vertex(0).unwrap();
    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();
    graph.add_vertex(3).unwrap();
    graph.add_vertex(4).unwrap();

    graph.add_undirected_edge(1, 0, "a".to_owned(), 2).unwrap();
    graph.add_undirected_edge(0, 2, "b".to_owned(), 6).unwrap();

    graph.add_undirected_edge(2, 1, "c".to_owned(), 3).unwrap();
    graph.add_undirected_edge(0, 3, "d".to_owned(), 8).unwrap();
    graph.add_undirected_edge(1, 4, "e".to_owned(), 5).unwrap();

    let expected = Ok(vec![0, 3, 2, 1, 4]);

    let actual = graph.dfs_path(&0);

    assert_eq!(actual, expected);
  }

  #[test]
  fn dfs_path_returns_error_when_starting_vertex_is_not_in_the_graph() {
    let graph = Graph::new();

    let expected = Err(VertexNotFoundError(-3));

    let actual = graph.dfs_path(&-3);

    assert_eq!(actual, expected)
  }

  #[test]
  fn returns_bfs_path() {
    let mut graph = Graph::new();

    graph.add_vertex(0).unwrap();
    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();
    graph.add_vertex(3).unwrap();

    graph.add_directed_edge(0, 1, "a".to_owned(), 2).unwrap();
    graph.add_directed_edge(0, 2, "b".to_owned(), 6).unwrap();

    graph.add_directed_edge(1, 2, "c".to_owned(), 3).unwrap();
    graph.add_directed_edge(2, 0, "d".to_owned(), 8).unwrap();
    graph.add_directed_edge(2, 3, "e".to_owned(), 8).unwrap();
    graph.add_directed_edge(3, 3, "f".to_owned(), 5).unwrap();

    let expected = Ok(vec![2, 0, 3, 1]);

    let actual = graph.bfs_path(&2);

    assert_eq!(actual, expected);
  }

  #[test]
  fn bfs_path_returns_error_when_starting_vertex_is_not_in_the_graph() {
    let graph = Graph::new();

    let expected = Err(VertexNotFoundError(2));

    let actual = graph.bfs_path(&2);

    assert_eq!(actual, expected);
  }

  #[test]
  fn returns_graph_strongly_connected_components() {
    let mut graph = Graph::new();

    graph.add_vertex(1).unwrap();
    graph.add_vertex(2).unwrap();
    graph.add_vertex(3).unwrap();
    graph.add_vertex(4).unwrap();
    graph.add_vertex(5).unwrap();
    graph.add_vertex(6).unwrap();
    graph.add_vertex(7).unwrap();
    graph.add_vertex(8).unwrap();

    graph.add_directed_edge(1, 2, "1".to_owned(), 1).unwrap();
    graph.add_directed_edge(2, 5, "2".to_owned(), 1).unwrap();
    graph.add_directed_edge(2, 7, "3".to_owned(), 1).unwrap();
    graph.add_directed_edge(3, 1, "4".to_owned(), 1).unwrap();
    graph.add_directed_edge(3, 4, "5".to_owned(), 1).unwrap();
    graph.add_directed_edge(4, 6, "6".to_owned(), 1).unwrap();
    graph.add_directed_edge(5, 8, "7".to_owned(), 1).unwrap();
    graph.add_directed_edge(6, 5, "8".to_owned(), 1).unwrap();
    graph.add_directed_edge(7, 1, "9".to_owned(), 1).unwrap();
    graph.add_directed_edge(7, 3, "10".to_owned(), 1).unwrap();
    graph.add_directed_edge(7, 8, "11".to_owned(), 1).unwrap();
    graph.add_directed_edge(8, 4, "12".to_owned(), 1).unwrap();

    let expected = vec![vec![1, 2, 3, 7], vec![4, 5, 6, 8]];

    let mut actual = graph.strongly_connected_components(&1).unwrap();

    for component in &mut actual {
      component.sort();
    }

    assert_eq!(expected, actual);
  }

  #[test]
  fn strongly_connected_components_returns_error_when_vertex_is_not_in_the_graph() {
    let graph = Graph::new();

    let expected = Err(VertexNotFoundError(5));

    let actual = graph.strongly_connected_components(&5);

    assert_eq!(expected, actual);
  }
}
