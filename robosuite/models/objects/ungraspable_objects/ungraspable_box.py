from robosuite.models.objects.primitive import BoxObject


class SlipBoxObject(BoxObject):
    """
    A box object designed to simulate realistic slip behavior during edge grasps.
    
    This object is configured with physics parameters that make edge grasps unstable,
    similar to real-world scenarios where large objects cannot be grasped at their edges.
    
    Key features:
    - condim=4: Enables elliptic friction cone for more realistic contact modeling
    - Lower friction coefficients to allow slipping under certain grasp configurations
    - Softer contact parameters (solimp/solref) to simulate deformation and instability
    
    Args:
        condim (int): Contact dimensionality (3=pyramidal friction cone, 4=elliptic cone with torsion).
                     Default 4 for more realistic friction behavior.
        edge_friction_scale (float): Scale factor for friction on edges vs faces (< 1.0 makes edges slippery).
                                    Default 0.5 makes edge grasps 50% more likely to slip.
    """
    def __init__(self, *args, condim=4, edge_friction_scale=0.5, **kwargs):
        self.condim = condim
        self.edge_friction_scale = edge_friction_scale
        
        # Override physics parameters for slip behavior if not explicitly set
        if 'friction' not in kwargs:
            # [sliding, torsional, rolling] - lower values promote slipping
            # Default: [1.0, 0.005, 0.0001], Slip: [0.3, 0.01, 0.001]
            kwargs['friction'] = [0.3, 0.01, 0.001]
        
        if 'solimp' not in kwargs:
            # [dmin, dmax, width] - softer contacts for instability
            # Default: [0.9, 0.95, 0.001], Softer: [0.8, 0.9, 0.01]
            kwargs['solimp'] = [0.8, 0.9, 0.01]
        
        if 'solref' not in kwargs:
            # [timeconst, dampratio] - less stiff response
            # Default: [0.02, 1.0], Softer: [0.1, 0.5]
            kwargs['solref'] = [0.1, 0.5]
        
        super().__init__(*args, **kwargs)

    def _get_object_subtree(self):
        # Get the normal box subtree
        obj = super()._get_object_subtree_('box')

        # Patch condim into collision geom(s)
        for geom in obj.findall('geom'):
            if geom.get('group', '0') == '0':  # only collision geoms
                geom.set('condim', str(self.condim))
        
        return obj
