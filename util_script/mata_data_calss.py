class update_pro_class():
    def __init__(self) -> None:
        pass
    def update_attr(self,name,value):
        setattr(self,name,value)
    def add_attr(self,name,value):
        setattr(self,name,value)

    def __repr__(self) -> str:
        ans=''+type(self).__name__
        ans+='{'
        ids=1
        for k,v in self.__dict__.items():
            ans+=k
            ans+=':'
            ans+=str(v)
            if ids%10==0:
                ans+='\n'
            else:
                ans+=' '*5
            ids+=1
        ans+='}'
        return ans

# class seq_data(update_pro_class):
    
#     def __init__(self) -> None:
#         super().__init__()
ele_lis_name=['place_lis','path_lis','spatial_entity_lis','motion_lis','spatial_signal_lis','motion_signal_lis',
                    'nonmotion_event_lis','measure_lis']
ele_attributes_name={
    'place_lis':['type','dimensionality','form','domain','continent','state','country','ctv','gazref','latlong','elevation','mod',
                'dcl','countable','gquat','scopes'],
    'path_lis':[ 'beginid','endid','midid','type','dimensionality','form','domain','gazref','latlong','elevation','mod','dcl',
                'countable','gquat','scopes'],
    'spatial_entity_lis':['type','dimensionality','form','domain','latlong','elevation','mod','dcl','countable','gquat','scopes'],
    'motion_lis':['domain','latlong','elevation','motion_type','motion_class','motion_sense','mod','countable','gquant','scopes'],
    'spatial_signal_lis':['cluster','semantic_type'],
    'motion_signal_lis':['motion_signal_type'],
    'nonmotion_event_lis':['domain','latlong','elevation','mod','countable','gquant','scopes'],
    'measure_lis':['value','unit']
}

link_lis_name=['qslink_lis','olink_lis','movelink_lis']
link_attributes_name={
    'qslink_lis':['reltype','trajector','landmark','trigger'],
    'olink_lis':['reltype','trajector','landmark','trigger','frame_type','referencept','projective'],
    'movelink_lis':['trigger','source','goal','midpoint','mover','landmark','goal_reached','pathid','motion_signalid']
}

class ori_data(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id2obj=dict()
        self.seqlis=[] 
        self.seqoriids=[] 
        self.seqtoklis=[] 
        self.seqtokidslis=[] 

        self.all_link_seqs=[]
        self.all_e1_mask=[]
        self.all_e2_mask=[]
        self.all_tr_mask=[]
        self.all_link_candidate=[] 
        self.all_link_label=[] 
        self.text=''

        self.spatial_entity_lis=[]
        self.nonmotion_event_lis=[]
        self.motion_lis=[]
        self.spatial_signal_lis=[]
        self.motion_signal_lis=[]
        self.measure_lis=[]
        self.place_lis=[]
        self.path_lis=[]
        

        self.movelink_lis=[]
        self.qslink_lis=[]
        self.olink_lis=[]
        self.measurelink_lis=[]
        self.metalink_lis=[]

class place(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.text=''
        self.id=''
        self.start=-1
        self.end=-1
        self.type=''
        self.dimensionality=''
        self.form=''
        self.domain=''
        self.continent=''
        self.state=''
        self.country=''
        self.ctv=''
        self.gazref=''
        self.latlong=''
        self.elevation=''
        self.mod=''
        self.dcl=''
        self.countable=''
        self.gquat=''
        self.scopes=''
        self.comment=''

class path(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.start=-1
        self.end=-1
        self.text=''
        self.beginid=''
        self.endid=''
        self.midid=''
        self.type=''
        self.dimensionality=''
        self.form=''
        self.domain=''
        self.gazref=''
        self.latlong=''
        self.elevation=''
        self.mod=''
        self.dcl=''
        self.countable=''
        self.gquat=''
        self.scopes=''
        self.comment=''

class spatial_entity(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.start=-1
        self.end=-1
        self.text=''
        self.type=''
        self.dimensionality=''
        self.form=''
        self.domain=''
        self.latlong=''
        self.elevation=''
        self.mod=''
        self.dcl=''
        self.countable=''
        self.gquat=''
        self.scopes=''
        self.comment=''

class nonmotion_event(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.start=-1
        self.end=-1
        self.text=''
        self.domain=''
        self.latlong=''
        self.elevation=''
        self.mod=''
        self.countable=''
        self.gquant=''
        self.scopes=''
        self.comment=''

class motion(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.start=-1
        self.end=-1
        self.text=''
        self.domain=''
        self.latlong=''
        self.elevation=''
        self.motion_type=''
        self.motion_class=''
        self.motion_sense=''
        self.mod=''
        self.countable=''
        self.gquant=''
        self.scopes=''
        self.comment=''

class spatial_signal(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.start=-1
        self.end=-1
        self.text=''
        self.cluster=''
        self.semantic_type=''
        self.comment=''

class motion_signal(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.start=-1
        self.end=-1
        self.text=''
        self.motion_signal_type=''
        self.comment=''


class measure(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.start=-1
        self.end=-1
        self.text=''
        self.value=''
        self.unit=''
        self.comment=''

class qslink(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.fromid=''
        self.fromtext=''
        self.toid=''
        self.totext=''
        self.reltype=''
        self.trajector=''
        self.landmark=''
        self.trigger=''
        self.comment=''

class olink(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.fromid=''
        self.fromtext=''
        self.toid=''
        self.totext=''
        self.reltype=''
        self.trajector=''
        self.landmark=''
        self.trigger=''
        self.frame_type=''
        self.referencept=''
        self.projective=''
        self.comment=''

class movelink(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.fromid=''
        self.fromtext=''
        self.toid=''
        self.totext=''
        self.trigger=''
        self.source=''
        self.goal=''
        self.midpoint=''
        self.mover=''
        self.landmark=''
        self.goal_reached=''
        self.pathid=''
        self.motion_signalid=''
        self.comment=''

class measurelink(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.fromid=''
        self.fromtext=''
        self.toid=''
        self.totext=''
        self.trajector=''
        self.landmark=''
        self.reltype=''
        self.val=''
        self.endpoint1=''
        self.endpoint2=''
        self.comment=''


class metalink(update_pro_class):
    def __init__(self) -> None:
        super().__init__()
        self.id=''
        self.fromid=''
        self.fromtext=''
        self.toid=''
        self.totext=''
        self.objectid1=''
        self.objectid2=''
        self.reltype=''
        self.comment=''


if __name__ == '__main__':
    a=place()
    print(a)