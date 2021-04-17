# Setup

[Install rust](https://www.rust-lang.org/tools/install)

# How to run

`$ cargo run`

**Example commands**

```console
dfs
graph.new
vertex.add 0
vertex.add 1
vertex.add 2
vertex.add 3
vertex.add 4
edge.undirected.add 1 0 "a" 2
edge.undirected.add 0 2 "b" 6
edge.undirected.add 2 1 "c" 3
edge.undirected.add 0 3 "d" 8
edge.undirected.add 1 4 "e" 5
dfs 0

bfs
graph.new
vertex.add 0
vertex.add 1
vertex.add 2
vertex.add 3
edge.directed.add 0 1 "a" 2
edge.directed.add 0 2 "b" 6
edge.directed.add 1 2 "c" 3
edge.directed.add 2 0 "d" 8
edge.directed.add 2 3 "e" 8
edge.directed.add 3 3 "f" 5
bfs 0

prim
graph.new
vertex.add 0
vertex.add 1
vertex.add 2
vertex.add 3
vertex.add 4
edge.undirected.add 0 1 "a" 2
edge.undirected.add 0 3 "b" 6
edge.undirected.add 1 2 "c" 3
edge.undirected.add 1 3 "d" 8
edge.undirected.add 1 4 "e" 5
edge.undirected.add 2 4 "f" 7
edge.undirected.add 3 4 "g" 9
prim 0

roy
graph.new
vertex.add 1
vertex.add 2
vertex.add 3
vertex.add 4
vertex.add 5
vertex.add 6
vertex.add 7
vertex.add 8
edge.directed.add 1 2 "1" 1
edge.directed.add 2 5 "2" 1
edge.directed.add 2 7 "3" 1
edge.directed.add 3 1 "4" 1
edge.directed.add 3 4 "5" 1
edge.directed.add 4 6 "6" 1
edge.directed.add 5 8 "7" 1
edge.directed.add 6 5 "8" 1
edge.directed.add 7 1 "9" 1
edge.directed.add 7 3 "10" 1
edge.directed.add 7 8 "11" 1
edge.directed.add 8 4 "12" 1
roy 2
```
