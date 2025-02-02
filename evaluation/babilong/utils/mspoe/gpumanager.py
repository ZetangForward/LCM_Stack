import os,sys
import torch
from argparse import Namespace
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class MultiGPUManager:
    def __init__(self,
                 tp_size=1,
                 gpu_list=list(range(torch.cuda.device_count())),
                 max_workers=1,
                 data_config=None,
                 model_config=None,
                 generate_config: dict=None):
        self.tp_size=tp_size
        self.gpu_list=gpu_list
        self.max_workers=max_workers
        self.data_config=data_config
        self.model_config=model_config
        self.generate_config=generate_config
        self.allocated_gpus=self.allocate_gpu(tp_size,gpu_list)
        self.processes=[]

        mp.set_start_method("spawn")
        self.task_queue=mp.Queue()
        self.result_list=mp.Manager().list()

    def allocate_gpu(self,tp_size,gpu_list):
        '''
        return :List[str] , each element is a group of gpus visable to a single process 
        '''
        gpu_list=list(map(str,gpu_list))
        if tp_size==1:return gpu_list
        return list(map(lambda x:", ".join(x),
                        [gpu_list[i:i+tp_size] for i in range(0,len(gpu_list),tp_size)]))
    
    @classmethod
    def setup_model(cls, model_config: Namespace):
        '''
        model_config is passed when init

        keep device_map="auto"
        return the loaded model 
        '''
        raise NotImplementedError


    def preprocess(self, task_queue: mp.Queue, data_config: Namespace):
        '''
        data_config is passed when init
        preprocess datas for model inputs,
        for each sample, do: task_queue.put(sample)
        
        the sample will soon be passed to the argment :`sample` of the function: `process` 

        '''
        raise NotImplementedError

    @classmethod
    def process(cls, model, generate_config:Namespace, sample, result_list : list):
        '''
        generate_config is passed when init

        model generate process, sample is the prepared data,

        while the output comes, do: result_list.append(output) 
        '''
        raise NotImplementedError


    @classmethod
    def worker(cls, setup_model, model_config, generate_config, max_workers, 
               task_queue: mp.Queue, result_list: list):
        model=setup_model(model_config)
        progress_bar = tqdm(desc=f"Process {os.getpid()}:", total=0)  # 初始化时 total=0
        i=0
        while True:
            sample=task_queue.get()
            progress_bar.total = i + 10  
            i+=1
            progress_bar.update(1)
            
            if sample is None:break
            cls.process(model, generate_config, sample, result_list)
        progress_bar.total=i
        progress_bar.close()
        # with ThreadPoolExecutor(max_workers=max_workers) as excutor:
        #     while True:
        #         sample=task_queue.get()
        #         if sample is None:break
        #         excutor.submit(cls.process, model, generate_config,
        #                         sample, result_list)


    def __exit__(self, exc_type, exc_value, traceback):
        for process in self.processes:
            process.terminate()
        # for visable_gpu in enumerate(self.allocated_gpus):
        #     os.environ['CUDA_VISIBLE_DEVICES'] = visable_gpu
        #     torch.cuda.empty_cache()
        return False    
    @classmethod
    def set_gpu(cls,i,visable_gpus):
        os.environ['CUDA_VISIBLE_DEVICES'] = visable_gpus[i]
    def __enter__(self):
        for i,visable_gpu in enumerate(self.allocated_gpus):
            
            # os.environ['CUDA_VISIBLE_DEVICES'] = visable_gpu
            self.set_gpu(i,self.allocated_gpus)
            process=mp.Process(target=self.worker,
                               args=(self.setup_model, 
                                     self.model_config, 
                                     self.generate_config,
                                     self.max_workers,
                                     self.task_queue,
                                     self.result_list))
            self.processes.append(process)
            process.start()
        
        self.preprocess(self.task_queue,self.data_config)

        for _ in range(len(self.allocated_gpus)):
            self.task_queue.put(None)

        for process in self.processes:
            process.join()

        return self

