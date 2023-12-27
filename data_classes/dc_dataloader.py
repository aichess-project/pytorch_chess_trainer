from dataclasses import dataclass

@dataclass
class DC_DataLoad:

    DICT_TYPE_SUBDIR = {"training": "train", "validation": "val", "testing": "test"}
    
    def get_all_types():
        return list(DC_DataLoad.DICT_TYPE_SUBDIR.keys())
    
    def get_sub_dir(type):
        return DC_DataLoad.DICT_TYPE_SUBDIR[type]
    
    def training():
        return DC_DataLoad.get_all_types()[0]
    
    def validating():
        return DC_DataLoad.get_all_types()[1]
    
    def testing():
        return DC_DataLoad.get_all_types()[2]