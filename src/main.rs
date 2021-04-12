use std::io::{self, Write};

pub mod graph;

fn main() {
  loop {
    print!("> ");

    io::stdout().flush().expect("flush failed");

    let mut buffer = String::new();

    io::stdin()
      .read_line(&mut buffer)
      .expect("unable to read input");
  }
}
