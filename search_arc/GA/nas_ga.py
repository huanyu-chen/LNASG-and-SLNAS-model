#nas_ga.py
import copy
import torch
import numpy as np

#seq_creater.py
from numba import jit
from numba.typed import List
from numba import njit, prange
import numpy as np
import utils
from tqdm import tqdm
@njit
def seq_creater():
    """
    random generator architecure sequence 
    varible
        prob_list : every operation's possible choice
    """
    res = List()
    prob_list = [2,5,2,5,3,5,3,5,4,5,4,5,5,5,5,5,6,5,6,5]
    [res.append(i) for i in prob_list ]
    for i in range(len(prob_list)):
        prob_list[i] = np.random.randint(0,prob_list[i])
    return prob_list
# arr = np.array( [[seq_creater() for i in range(11)] for i in range(100)])
# print(arr)

class nas_ga():
    """
    GA generator training step 
    """
    def __init__(self,dataloader,population_number = 20 ,chromosome_s = 2,mutation_rate=0.001,cross_rate=0.7):
        # self.model = model
        # self.model.cuda()
        self.dataloader = dataloader
        self.mutation_rate = mutation_rate
        self.cross_rate  = cross_rate
        self.population = np.array( [[seq_creater() for i in range(chromosome_s)] for i in range(population_number)])
        self.population_number = population_number
        self.chromosome_s = chromosome_s
        self.score = None
        self.master = []
        self.master_score = []
    def select(self):
        idx = np.random.choice(np.arange(self.population_number), size=self.population_number//2, replace=True, p=np.power(self.score,2)/np.power(self.score,2).sum())
        strange_guy = np.array( [[seq_creater() for i in range(self.chromosome_s)] for i in range(self.population_number//2)])
        return np.vstack((self.population[idx],strange_guy))
    def crossover(self,father):
        #father = self.population.copy()
        np.random.shuffle(self.population)
        self.population = np.concatenate((father , self.population ),axis=2).reshape(self.population_number,self.chromosome_s,2,20)[:,:,np.random.choice(2, size=20,p = [ self.cross_rate,1-self.cross_rate]),[i for i in range(20)]]
    def mutation(self):
        child = self.population.copy()
        x_ray = np.array( [[seq_creater() for i in range(self.chromosome_s)] for i in range(self.population_number)])
        self.population =  np.concatenate((child , x_ray ),axis=2).reshape(self.population_number,self.chromosome_s,2,20)[:,:,np.random.choice(2, size=20,p = [ 1-self.mutation_rate,self.mutation_rate]),[i for i in range(20)]]
    def fitness(self,model):
        # fit = self.population.sum(axis=1).sum(axis=1)#np.random.sample(self.population_number)
        # print(fit.sum(),np.array( [[seq_creater() for i in range(self.chromosome_s)] for i in range(self.population_number)]).sum(axis=1).sum(axis=1).sum())
        total_reward = utils.AvgrageMeter()
        fit = []
        for step,dag in enumerate(self.population):
            #print(dag)
            data, target = self.dataloader.next_batch()
            n = data.size(0)

            data = data.cuda()
            target = target.cuda()
            with torch.no_grad():
                logits,aux = model(dag.tolist(), data)
                #print(dag.tolist())
                reward = utils.accuracy(logits, target)[0]

            fit.append(reward.item())
            total_reward.update(reward.item(), n)
        self.score = np.array(fit)
        print(self.score.mean())
        return np.array(fit)
    def history_hero(self):
        """
        every epoch's best architecture sequence 
        in my experiment the the history is not useful
        """
        score , idx = torch.topk(torch.from_numpy(self.score),5,0 , True)
        self.master.append(self.population[idx].tolist())
        self.master_score.append(score.tolist())
    def GA_training(self,generation,model):
        for i in tqdm(range(generation)):
            self.score = self.fitness(model)
            self.history_hero()
            father = self.select()
            self.crossover(father)
            self.mutation()
            if i % 10 ==  0:
                self.mutation()
    def dag_creater(self):
        idx = np.random.choice(np.arange(self.population_number), size=1, replace=False)
        return self.population[idx][0].tolist()

