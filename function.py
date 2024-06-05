import time
import seaborn as sns
from mpire import WorkerPool
from mpire.utils import make_single_arguments
import pickle

## Multiprocess Functions ##

def multiprocess_func(func):
    '''
    Uses MPIRE multiprocessing to create a parameter
    sweep for a function
    '''
    def wrapper(*args):
        with WorkerPool(n_jobs=4) as pool:
            results = pool.map(func, *args)
        return results
    
    return wrapper


def multiprocess_model(model):
    '''
    Uses MPIRE multiprocessing to create a parameter
    sweep for a complex model. Assumes model has a 
    .run() and .reset_states() methods.
    '''
    # Pickle model for each worker
    with open('model.p3', 'wb') as file:
        pickle.dump(model, file)
    
    def wrapper(*args):
        # Convert runs data frame into single argument dicts
        runs = df_to_arguments(*args)
        
        with WorkerPool(n_jobs=4, use_worker_state=True) as pool:
            # Best practice to limit worker lifespan (due to poor python garbage collection)
            # Opting for shorter lifespans because model class could be large
            results = pool.map(run_model, runs, worker_init=worker_init, worker_lifespan=4, chunk_size=4)

        return results
    
    return wrapper


def worker_init(worker_state):
    # Send a model to worker (done only once per worker)
    with open('model.p3', 'rb') as file:
        worker_state['model'] = pickle.load(file)
        
        
def run_model(worker_state, parameters):
    # Run model with given parameters
    model_output = worker_state['model'].run(parameters)
    # Clean up model states for another run
    worker_state['model'].reset_state()

    return model_output


## Dummy Model ##

class DummyModel():
    
    def __init__(self, reference_data):
        
        self.data = reference_data
        self.ticks = 0
        
    def run(self, params):     
        # Run time consuming function and compare results to reference data
        x = slow_function(params['x']) - self.data['x']
        y = slow_function(params['y']) - self.data['y']
        
        return x, y
    
    def reset_state(self):
        # Reset model state parameters that changed over the run
        self.ticks = 0

## Helper Functions ##

def df_to_arguments(df):
    # Converts a data frame of model runs into MPIRE single argument dicts
    return make_single_arguments([row.to_dict() for _, row in df.iterrows()])


def slow_function(x):
    time.sleep(1)  # Simulate that this function takes long to complete
    return x*2 