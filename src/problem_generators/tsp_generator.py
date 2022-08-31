import numpy as np
import os

class progen:
    def __init__(self):
        self.MAXCOORD=1000000
        self.PRANDMAX=1000000000

        self.a,self.b=0,0
        self.arr_size=55
        self.arr=np.zeros((self.arr_size,),dtype=int)

    def generate(self,num_of_cities:int,init_seed:int,output_folder:str):
        filename="{}{}_{}.tsp".format("E",num_of_cities,init_seed)

        facotr=int(self.PRANDMAX/self.MAXCOORD)
        i=0
        x, y=0,0
            
        N = int(num_of_cities)
        seed = int(init_seed)

        #initialize random number generator

        self.sprand(seed)

        with open(os.path.join(output_folder,filename),'w') as file:
            file.write("NAME : portgen-{}-{}\n".format(N, seed))
            file.write("COMMENT : portgen N={}, seed={}\n".format(N, seed))
            file.write("TYPE : TSP\n")
            file.write("DIMENSION : {}\n".format(N))
            file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            file.write("NODE_COORD_SECTION\n")

            for i in range(1,N+1):
                x = int(int(self.lprand())/facotr)
                y = int(int(self.lprand())/facotr)
                file.write("{} {} {}\n".format(i,x,y))
        
        return 0

    def sprand(self,seed:int):
        i,ii=0,0
        last_,next_=0,0
        self.arr[0] = last_ = seed
        next_ = 1
        for i in range(1,55):
            ii = (21*i)%55
            self.arr[ii] = next_
            next_=last_-next_+self.PRANDMAX if last_-next_<0 else last_-next_
            last_=self.arr[ii]
        self.a = 0
        self.b = 24
        for i in range(0,165):
            last_ = int(self.lprand())

    def lprand(self):
        t=0
        self.a=54 if self.a==0 else self.a-1
        self.b=54 if self.b==0 else self.b-1
        t = self.arr[self.a] - self.arr[self.b]

        if t<0:
            t+= self.PRANDMAX
        self.arr[self.a] = t
        return t

class procgen(progen):
    def __init__(self):
        super().__init__()
        self.MAXN=1000000
        self.CLUSTERFACTOR=100
        self.SCALEFACTOR=np.float32(1)

        self.center=np.zeros((self.MAXN+1,2),dtype=int)

        self.goodstill = 0
        self.nextvar=np.float32(0)

    def generate(self,num_of_cities:int,init_seed:int,output_folder:str):
        filename="{}{}_{}.tsp".format("C",num_of_cities,init_seed)

        c=0
        i,j=0,0
        x, y=0,0
        nbase=0
        scale=np.float32(0)
            
        N = int(num_of_cities)
        seed = int(init_seed)

        #initialize random number generator

        self.sprand(seed)

        nbase = N//self.CLUSTERFACTOR
        scale = self.SCALEFACTOR/np.sqrt(N)

        with open(os.path.join(output_folder,filename),'w') as file:
            file.write("NAME : portcgen-{}-{}\n".format(N, seed))
            file.write("COMMENT : portcgen N={}, seed={}\n".format(N, seed))
            file.write("TYPE : TSP\n")
            file.write("DIMENSION : {}\n".format(N))
            file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            file.write("NODE_COORD_SECTION\n")

            for i in range(1,nbase+1):
                for j in range(0,2):
                    self.center[i][j] = int(self.lprand()/self.PRANDMAX*self.MAXCOORD)

            for i in range(1,N+1):
                c = int(self.lprand()/self.PRANDMAX*nbase) + 1
                x = self.center[c][0] + int(self.normal()*scale*self.MAXCOORD)
                y = self.center[c][1] + int(self.normal()*scale*self.MAXCOORD)
                file.write("{} {} {}\n".format(i,x,y))
        
        return 0

    def normal(self)->np.float32:
        """Algorithm 3.4.1.P, p. 117, Knuth v. 2"""

        s,t,v1,v2=np.float32(0),np.float32(0),np.float32(0),np.float32(0)
        if self.goodstill:
            self.goodstill = 0
            return self.nextvar
        
        else:
            self.goodstill = 1
            while True:
                v1 = 2*self.lprand()/self.PRANDMAX - 1.0
                v2 = 2*self.lprand()/self.PRANDMAX - 1.0
                s = v1*v1 + v2*v2
                if s<1.0:
                    break

            t = np.sqrt((-2.0*np.log(s)) / s)
            self.nextvar = v1 * t	#Knuth's x1
            return v2 * t		#Knuth's x2

GENERATORS={
    "progen":progen,
    "procgen":procgen,
}

def generate_tsp_problems(num_of_city:int,output_folder:str="../data/tsp_instances/",start_seed:int=1,end_seed:int=100,type_of_generator:str="procgen"):
    output_path=os.path.join(output_folder,str(num_of_city))

    generator_class=type_of_generator
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    train_folder=os.path.join(output_path,"train")
    eval_folder=os.path.join(output_path,"eval")
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    if not os.path.isdir(eval_folder):
        os.mkdir(eval_folder)

    generator=GENERATORS[generator_class]()
    for i in range(start_seed,end_seed+1):
        generator.generate(num_of_city,i,train_folder)

    if not os.listdir(eval_folder):
        eval_id=np.random.randint(start_seed,end_seed+1)
        generator.generate(num_of_city,eval_id,eval_folder)
