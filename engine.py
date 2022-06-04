
import importlib.util

class Engine():
    def __init__(self):
        self.generator = None 

    def set_generator(self, generator_name, engines):
        self.generator = engines[generator_name]    
    
    def apply_transform(self, object, transformation, parameter, dataset_path, semantic_path, output_path, phase=1):
        entry = self.generator['entry']
        spec = importlib.util.spec_from_file_location(entry.split('/')[-1].split('.')[0], entry)
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        gen_module.image_control(object, transformation, parameter, dataset_path, semantic_path, output_path, phase=phase)
