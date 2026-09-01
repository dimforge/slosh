mod centilever_beam2;
mod elastic_cut2;
mod elasticity2;
#[cfg(feature = "pml")]
mod non_reflecting2;
mod sand2;

#[kiss3d::main]
pub async fn main() {
    #[allow(unused_mut)]
    let mut scenes: slosh_testbed2d::SceneBuilders<_> = vec![
        ("centilever beam".to_string(), centilever_beam2::beam_demo),
        ("sand".to_string(), sand2::sand_demo),
        ("elasticity".to_string(), elasticity2::elasticity_demo),
        ("elastic_cut".to_string(), elastic_cut2::elastic_cut_demo),
    ];

    // The absorbing skirt needs `ParticleModel::absorbing_pml`.
    #[cfg(feature = "pml")]
    scenes.push((
        "non-reflecting boundary".to_string(),
        non_reflecting2::non_reflecting_demo,
    ));

    slosh_testbed2d::run(scenes).await;
}
