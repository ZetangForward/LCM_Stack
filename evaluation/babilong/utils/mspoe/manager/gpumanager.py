import os,sys
sys.path.append(".")
from argparse import Namespace
from typing import List
import torch
from .waitwork import ProducerConsumerManager
import multiprocessing as mp

class LLMEvalGPUManager(ProducerConsumerManager):
    def __init__(self, 
                 tp_size: int=1,
                 gpu_list: List[int]=list(range(torch.cuda.device_count())),
                 task_info_list: list=[],
                 basic_datas_number: int=None,
                 data_config: Namespace = Namespace(),
                 model_config: Namespace = Namespace(),
                 generate_config: Namespace= Namespace()):
        self.tp_size=tp_size
        self.gpu_list=gpu_list
        self.allocated_gpus=self.allocate_gpu(tp_size,gpu_list)

        super().__init__(task_info_list=task_info_list,
                         basic_datas_number=basic_datas_number,
                         max_producers=1,
                         max_consumers=len(self.allocated_gpus),
                         produce_config=Namespace(preprocess=self.preprocess,
                                                  data_config=data_config),
                         consume_config=Namespace(setup_model=self.setup_model,
                                                  process=self.process,
                                                  model_config=model_config,
                                                  generate_config=generate_config),
                         consume_env_config=Namespace(allocated_gpus=self.allocated_gpus))
        
    @classmethod
    def allocate_gpu(cls,tp_size,gpu_list):
        '''
        return :List[str] , each element is a group of gpus visable to a single process 
        '''
        gpu_list=list(map(str,gpu_list))
        if tp_size==1:return gpu_list
        return list(map(lambda x:", ".join(x),
                        [gpu_list[i:i+tp_size] for i in range(0,len(gpu_list),tp_size)]))

    @classmethod
    def set_consumer_environment(cls, consumer_id: int, consume_env_config: Namespace):
        os.environ['CUDA_VISIBLE_DEVICES'] = consume_env_config.allocated_gpus[consumer_id]
    @classmethod
    def set_consumer_global_variables(cls, consume_config: Namespace) -> Namespace:
        return Namespace(model=consume_config.setup_model(consume_config.model_config))
    
    @classmethod
    def produce(cls, task_info, produce_config: Namespace, glb: Namespace):
        return produce_config.preprocess(task_info, produce_config.data_config, glb)

    @classmethod
    def consume(cls, task_sample, consume_config: Namespace, glb: Namespace):
        return consume_config.process(glb.model, task_sample, consume_config.generate_config)

    @classmethod
    def setup_model(cls, model_config: Namespace) -> object:
        '''
        model_config is passed when initiating

        keep device_map="auto"
        return the loaded model 
        '''
        raise NotImplementedError

    @classmethod
    def preprocess(cls, task_info, data_config: Namespace, glb: Namespace) -> object:
        '''
        task_info is the element of the task_info_list when initating 
        data_config is passed when initiating
        glb is the return value of the `set_producer_global_variables`

        preprocess datas for model inputs,
        for each sample, do: return sample
        
        the sample will soon be passed to the argment :`sample` of the function: `process` 

        '''
        raise NotImplementedError

    @classmethod
    def process(cls, model, sample, generate_config: Namespace) -> object:
        '''
        generate_config is passed when init

        glb is the return value of the `set_consumer_global_variables`
    
        model generate process, sample is the prepared data,

        while the output comes, do: return output 
        '''
        raise NotImplementedError

if __name__=="__main__":
    with LLMEvalGPUManager(
        tp_size=1,
        gpu_list=[0],
        task_info_list=[]
    ) as manager:
        result_list=manager.result_list