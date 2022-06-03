
import importlib.util

class Engine():
    def __init__(self):
        self.generator = None 

    def set_generator(self, generator_name, engines):
        self.generator = engines[generator_name]    
    # def set_generator(self, target_object, target_transformation, target_parameter, generator_list): 
    #     found = False
    #     for generator in generator_list:
    #         for support_transformation in generator.support_transformation:
    #             if target_transformation == support_transformation[0]:
    #                 for support_object in support_transformation[1]:
    #                     if support_object in target_object:
    #                 # if target_object in generator
    #                         self.generator = generator
    #                         found = True
    #                         break
    #                     if found:
    #                         break

    #                 if found:
    #                     break
    #         if found:
    #             break
    #     return found
    
    def apply_transform(self, object, transformation, parameter, dataset_path, semantic_path, output_path, phase=1):
        entry = self.generator['entry']
        spec = importlib.util.spec_from_file_location(entry.split('/')[-1].split('.')[0], entry)
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        gen_module.image_control(object, transformation, parameter, dataset_path, semantic_path, output_path, phase=phase)
