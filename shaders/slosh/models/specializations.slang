module specializations;

import slosh.models.interfaces;
import slosh.models.implicit_interfaces;
import slosh.models.default;


public export struct ParticleModel: IParticleModel = DefaultParticleModel;
public export struct ImplicitParticleModel: IImplicitParticleModel = DefaultParticleModel;
