mod elastic_cut2;
mod elasticity2;
mod sand2;

#[kiss3d::main]
pub async fn main() {
    slosh_testbed2d::run(vec![
        ("sand".to_string(), sand2::sand_demo),
        ("elasticity".to_string(), elasticity2::elasticity_demo),
        ("elastic_cut".to_string(), elastic_cut2::elastic_cut_demo),
    ]).await;
}
