#![allow(dead_code)]
mod pipeline;
use pipeline::Pipeline;

fn main() {
    let p = Pipeline::new(vec![1, 2, 3, 4, 5]);
    let odd = p.filter(|n| n % 2 == 1);
    let out: Vec<String> = odd.map(|n| format!("#{}", n)).execute();
    assert_eq!(out, vec!["#1", "#3", "#5"], "filter/map/execute result wrong");
    assert_eq!(p.execute(), vec![1, 2, 3, 4, 5], "filter must not change the original pipeline");
    println!("POLYGLOT_OK {}", std::env::var("EVAL_RUN_ID").unwrap_or_default());
}
