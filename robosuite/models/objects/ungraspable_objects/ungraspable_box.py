from robosuite.models.objects.primitive import BoxObject


class SlipBoxObject(BoxObject):
    def __init__(self, *args, condim=3, **kwargs):
        self.condim = condim
        super().__init__(*args, **kwargs)

    def _get_object_subtree(self):
        # Get the normal box subtree
        obj = super()._get_object_subtree_('box')

        # Patch condim into collision geom(s)
        for geom in obj.findall('geom'):
            if geom.get('group', '0') == '0':  # only collision geoms
                geom.set('condim', str(self.condim))
        return obj
