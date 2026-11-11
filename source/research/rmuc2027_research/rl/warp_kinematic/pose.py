"""Loaded ride height for the planar backend, without a rigid-body solver.

This is a quasi-static support model: actuator command rates and static spring
compression match the design; suspension bounce and attitude remain unmodelled.
"""
import warp as wp


def collision_heights(design):
    """Upright body-local top offsets, not heights above the terrain.

    Match the native compound *solid* hull for the folding and wheel-legged
    sentries. The caller adds the current body origin (including loaded leg
    height) exactly once. This does not model a tilted hull or moving arms;
    other designs retain their existing coarse envelope.
    """
    muzzle=design.get('muzzle_z_m',.4)
    folded=design.get('folded_muzzle_z_m',muzzle)
    legacy=(muzzle+.06,folded+.06)
    if design.get('role')!='sentry' or design.get('arm_count',0):return legacy
    size=design.get('chassis_size_m')
    if not size:return legacy
    if design.get('chassis_type')=='wheel_legged':
        hip=design['hip_height_m']
        # PhysicsWorld::leg_hull: rounded trunk, gimbal, then wheel spheres.
        # Across the legal upright leg range the gimbal remains the highest
        # solid part for 2772; the loaded origin carries the leg extension.
        top=max(hip+size[2]/2.+.012,muzzle-.01,
                hip-(design['leg_min_m']-.01)+design['wheel_radius_m']*.6)
        return top,top
    if design.get('transform_allowed'):
        # PhysicsWorld::hull_with_arms: clearance includes the rounded margin;
        # the solid gimbal center is muzzle+.005 with half-height .060.
        clearance=design.get('ground_clearance_m',0.)+design.get('suspension_static_compression_m',0.)+.012
        chassis=max(clearance+size[2]+.012,clearance+.03+.018)
        return max(chassis,muzzle+.065),max(chassis,folded+.065)
    return legacy


def parameters(design):
    mass=design.get('mass_kg',25.)
    if design.get('chassis_type')=='wheel_legged':
        limits=[design['leg_min_m'],design['leg_max_m'],design['leg_nominal_m'],design['leg_speed_mps']]
        support=[design['leg_stiffness_npm'],mass,design['hip_height_m'],design['wheel_radius_m']]
        return limits,support,0.
    free=max(design.get('suspension_free_length_m',0.),design.get('suspension_rest_m',0.))
    stiffness=design.get('suspension_stiffness_npm',0.)
    ride=free-design.get('suspension_rest_m',0.)-mass*9.81/(4.*stiffness) if stiffness>0. else 0.
    return [0.,0.,0.,0.],[0.,mass,0.,0.],ride


def resting_origin(design):
    """Nominal support height when an initial pose is already airborne."""
    leg,support,ride=parameters(design)
    if leg[1]<=0:return ride
    load=support[1]*9.81*.5
    length=leg[2]-load/support[0]
    if length<leg[0]-.01:
        length=(support[0]*leg[2]+40000.*(leg[0]-.01)-load)/(support[0]+40000.)
    return support[3]+length-support[2]


@wp.func
def loaded_origin(command: float, minimum: float, support: wp.vec4):
    load=support[1]*9.81*.5
    length=command-load/support[0]
    stop=minimum-.01
    if length<stop:
        # Same 40 kN/m mechanical-stop spring as the native leg controller.
        length=(support[0]*command+40000.0*stop-load)/(support[0]+40000.0)
    return support[3]+length-support[2]
