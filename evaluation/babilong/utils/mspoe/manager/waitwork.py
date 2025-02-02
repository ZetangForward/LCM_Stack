import os,sys,random
from typing import List,Optional
import torch
from argparse import Namespace
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time


class ProducerConsumerManager:
    '''
    If the class inheriting from this class is not the end-use class,
    override the `__init__` method and put all newly defined functions into a namespace
    at initialization space, and call them in that namespace instead of directly
    '''

    def __init__(self,
                 task_info_list : list=[],
                 basic_datas_number: int=None,
                 max_producers: int = 1,
                 max_consumers: int = 1,
                 produce_config: Namespace=Namespace(),
                 consume_config: Namespace=Namespace(),
                 produce_env_config: Optional[Namespace]=None,
                 consume_env_config: Optional[Namespace]=None
                 ):
        self.task_info_list=[_ for _ in task_info_list]
        random.shuffle(self.task_info_list)
        self._estimate_avg_tasks=max(1,len(self.task_info_list)//max_consumers) \
            if basic_datas_number is None else basic_datas_number//max_consumers
        self.max_producers=max_producers
        self.max_consumers=max_consumers
        self.produce_config=produce_config
        self.consume_config=consume_config
        self.produce_env_config=produce_env_config
        self.consume_env_config=consume_env_config

        self.producers=[]
        self.consumers=[]

        mp.get_start_method("spawn")

        self.manager=mp.Manager()
        self._task_queue=self.manager.Queue()
        self._result_list=self.manager.list()

    
    def __exit__(self,exc_type, exc_value, traceback):
        for process in self.producers+self.consumers:
            process.terminate()
        self.manager.shutdown()
        return False
    def __enter__(self):
        for i in range(self.max_consumers):
            self.set_consumer_environment(i,self.consume_env_config)
            process=mp.Process(
                target=self._consumer,
                args=(self.set_consumer_global_variables, self.consume,
                      self._estimate_avg_tasks,
                      self._task_queue,self.consume_config, self._result_list)
            )
            self.consumers.append(process)
            process.start()
            

        for i,task_info_list in enumerate(self.chunks(self.task_info_list,self.max_producers)):
            self.set_producer_environment(i,self.produce_env_config)
            process=mp.Process(
                target=self._producer,
                args=(self.set_producer_global_variables, self.produce,
                      task_info_list, self._task_queue, self.produce_config)
            )
            
            self.producers.append(process)
            process.start()

        
        assert len(self.producers)==self.max_producers, "Chunk Error!!"


        for process in self.producers:
            process.join()

        for _ in range(self.max_consumers):
            self._task_queue.put(None)
    
        for process in self.consumers:
            process.join()        

        self.result_list=list(self._result_list)

        return self
    
    @classmethod
    def chunks(cls, lst, chunk_num):
        chunk_size=(len(lst)-1)//chunk_num + 1
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    @classmethod
    def set_producer_environment(cls, producer_id: int, produce_env_config: Namespace):
        '''
        set the environment variable for producers before created

        produce_env_config is the object you passed when initating

        '''
        if produce_env_config is not None:raise NotImplementedError
    @classmethod
    def set_consumer_environment(cls, consumer_id: int, consume_env_config: Namespace):
        '''
        
        set the environment variable for consumers before created

        consume_env_config is the object you passed when initating
        '''
        
        if consume_env_config is not None:raise NotImplementedError

    @classmethod
    def _producer(cls, set_producer_global_variables, produce,
                  task_info_list: list, task_queue: mp.Queue, produce_config: Namespace):
        global_variables=set_producer_global_variables(produce_config)
        for task_info in tqdm(task_info_list,desc=f"Producer: {os.getpid()}"):
            for task_sample in tqdm(produce(task_info, produce_config, global_variables),
                                    desc=f"P single: {os.getpid()}"):
                if task_queue.qsize()>2048:
                    while task_queue.qsize()>1024:
                        time.sleep(1)
                task_queue.put(task_sample)
    @classmethod
    def _consumer(cls, set_consumer_global_variables, consume, _estimate_avg_tasks,
                  task_queue: mp.Queue, consume_config: Namespace, result_list: List):
        progress_bar = tqdm(desc=f"Consumer: {os.getpid()}", total=_estimate_avg_tasks)  
        delta=10
        number_tasks=0
        
        global_variables=set_consumer_global_variables(consume_config)
        while True:
            task_sample=task_queue.get()
            if task_sample is None:break
            for task_result in consume(task_sample, consume_config, global_variables):
                result_list.append(task_result)

                number_tasks+=1
                if number_tasks==progress_bar.total:
                    progress_bar.total+=delta
                progress_bar.update(1)
        progress_bar.total=number_tasks
        progress_bar.close()

    @classmethod
    def set_producer_global_variables(cls, produce_config: Namespace) -> Namespace:
        '''
        set the public global in every producer process, which will be used during loop

        Sentence like `cls.func` is not permitted, since it may cause error!
        Please Use any functions or variables in produce_config
        '''
        return Namespace()
    @classmethod
    def set_consumer_global_variables(cls, consume_config: Namespace) -> Namespace:
        '''
        set the public global in every consumer process, which will be used during loop
        Sentence like `cls.func` is not permitted, since it may cause error!
        Please Use any functions or variables in consume_config
        '''
        return Namespace()

    @classmethod
    def produce(cls, task_info , produce_config: Namespace, glb: Namespace) -> object:
        '''
        task_info is the sample in task_info_list you passed when initiating the manager
        produce_config is also the object you passed when initiating.

        After gaining a sample , do : yield sample
        
        '''
        raise NotImplementedError

    @classmethod
    def consume(cls, task_sample, consume_config: Namespace, glb: Namespace) -> object:
        '''
        task_sample is the return value of the   `produce` you implement
        consume_config is the object you passed when initating

        After gaining a result, do : yield result
        '''
        raise NotImplementedError

if __name__=="__main__":
    with ProducerConsumerManager(
        task_info_list=[{'task_type':'x','info': 'int'},{'task_type':'y','info': 'str'}],
        max_producers=2,
        max_consumers=2,
        produce_config=Namespace(),
        consume_config=Namespace()
    ) as manager:
        result_list=manager.result_list
