mod elastic_cut3;
mod heightfield3;
mod sand3;
mod centilever_beam3;

#[kiss3d::main]
pub async fn main() {
    slosh_testbed3d::run(vec![
        ("centilever beam".to_string(), centilever_beam3::beam_demo),
        ("sand".to_string(), sand3::sand_demo),
        ("heightfield".to_string(), heightfield3::heightfield_demo),
        ("elastic_cut".to_string(), elastic_cut3::elastic_cut_demo),
    ])
    .await;
}
