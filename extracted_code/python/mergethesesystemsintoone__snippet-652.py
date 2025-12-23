"""
def __init__(
    self,
    node_id: Optional [str] = None, # if no id assigne unique ID
    initial_energy: float = 100.0, # initial enery of a given node upon creation, must be of float datatype.
    parent_node: Optional['BaseNode'] = None ,
    initial_dna = None
      ):
    self.node_id = node_id or str(uuid.uuid4()) # sets unque id . id allows tracking of system processess across the AI as operations are preformed in various threads/tasks at various stages. this keeps a history that can be used to further evaluate all behaviour downstream.
    self.trait_manager = initial_dna if initial_dna is not None else TraitManager()# Allows injection of a trait if one exists . traits guide and modify how the behaviour happens from base traits if created otherwise will fallback on  default configurations.
    self.dna =  initial_dna if initial_dna is not None else KnowledgeDNA()  # Creates DNA encoding which encodes all processes at its node level allowing inheritance of that core component information. for adaptive and dynamic evolutionary capabilities.
    self.metrics = NodeMetrics(self.node_id) # Metrics for internal logging and monitoring of core resources or if you add additiional information for testing etc for performance, energy used per step .

    self._lock = Lock() # safety control, lock for preventing threads writing to a state or parameter in object by limiting processes access. concurrency .
    self.state = NodeState ( # base definition of state using default set to manage runtime and core functionality .
       energy=initial_energy,
        health=1.0,
         tasks_completed =0 ,  
         last_activity = datetime.now() ,
        processing_capacity=self.trait_manager.get_trait_value('processing_speed'),
         memory_usage=0.0, 
         connections=set() # set list of nodes connected via graph based representation for node connections . these are node ids or name string values.
     ) # This is the state which represents how we use or make calculations for nodes which is core state
    self.input_queue: List[DataWrapper] = [] # creates buffer queues and data store to allow various components of system to feed it new tasks for performance and prevent overflow behaviours that would slow systems if too many messages are queued with poor scheduling and load management techniques ( as in synchronous approach.)
    self.output_queue: List[Dict] = [] # buffer output ( allows more options down stream in a modular pattern )
    self.processing_buffer: Dict = {} #  additional buffer , is implemented for local operations that may be present if code grows with a lot of operations (to not repeat processing ). this can be done better if optimized memory methods are available

  # track of who the creator of this obj is if required for tracing information on complex graphs. it can aid better understanding
    self.parent_node = parent_node 

# core operational layers that can be extended and implmeented for specilisations later. for easy node customization using composition (instead of full new objects.) also to keep things concise by providing basic set functions here and not every time as code is run

    # Load specified Processing Units modules as part of construction. (Data Handlers specific to operation based on type/metadata)
  self.processing_units: Dict [str,ProcessingUnit] ={ # sets all processing modules based on declared format or object attribute values during run time. 
           "text": TextProcessingUnit() , # Use text based functions for processing.
        "image":ImageProcessingUnit (),   # specific process chains. (if a node or action needs a certain method of handling data ( text -  images - audio- numerical . or complex matrix) that should be implented here for clear separation). for performance reasons during scale up. or unique module based applications that do not effect overall core performance as new methods added/ or requirements expand upon this project over time for research etc.. 
     "numerical": NumericalProcessingUnit(),
         # additional proccess chains for any data
    }
  self.memory_field = MemoryField(dimensions=config.cube_size)   # intialising this within node scope and unique based upon node structure. this acts a short / long-term dynamic local store. 
  self.internal_state = { # this contains all attributes tracked internally (state ) also keeps consistent if new behaviours arise or different states have added rules downstream in AI models that uses those specific attributes. so code maintains consistencies .  . it creates base framework to then add additional details (if you make this method to collect more ) or for other custom metrics. if needed by a module using that core object/ class attributes and functions in it

  }
  logger.info (f"Node {self.node_id} initialized with DNA generation {self.dna.generation}.") # debugging logs

def get_processing_unit(self, data_type: str) -> Optional [ProcessingUnit]:
     """
       Selects data proccessing units , these contain processing and logic needed to ingest datat and is specific to data structures

         parameters data_type : specific  to input (string format from Datawrapper ).

          returns None  or data unit processing method instance. 

       """
     if data_type not in self.processing_units: # check of defined list exists or handles it. this avoid exceptions and failures during execution flow as operations progress or as new information becomes available by other modules or operations.
           return None # skip process if datatype is unavalilable

          # Check the processing capacity for prioritisation of what is important to operation, higher energy consumption ( or lower if resource heavy.) using the node traits
     affinity_trait = f"affinity_{data_type}" # dynamically get what to process based on dna/traits and set a baseline of prioritizaton or logic for behaviour in an action at this step (allows self specialization ) . using string form to reduce over complication
     if affinity_trait in self.trait_manager.traits: # safe checking to confirm type exists , else fallback.
         affinity = self.trait_manager.get_trait_value (affinity_trait)
         if affinity < 0.5:
            return None

     return self.processing_units [data_type]

async def process_data_chunk(self, data_wrapper: DataWrapper) -> Optional [Dict [str,Any] ]:
    """ Core execution pathway, main functionality is implmeneted within method by connecting internal process functions for data transaformation -> state management, or knowledge injection . for a more streamlined pipeline

          Parameters: a structured dictionary as required. Datawapper, must be correct type, has built in fail checks for consistency. 

           Returns dictionary information used by calling system, allows type detection, data integrity and operations results

     Implements dynamic processing logic using traits, to handle the workflow from the core data processing. before adding it into engine or the knowledge graph.  Also uses this core process pathway as it takes place sequentially  to generate  memory traces.

        Steps are: - Verfy basic operating criteria for that node, based on resource limitations/ or valid state to take actions for data. , use dynamic processing module chains that converts to new structured insights , and logs information ( state ) into the memory
    """
    if not self._can_process():  # ensure min requirements for core node behaviour or resources exists . skips entire cycle if true . allows for early fail / save. more accurate outputs of what operations work under stress.
      logger.warning (f"Node {self.node_id} cannot process data: insufficient energy or health.")  # track to avoid over loaded behaviours with data to influence the flow.

      return None # Returns non to short cycle, skip actions for unworkable node, with log.

    try:  
        if not validate_data_chunk (data_wrapper):  # if there data is not proper, skips it to stop system from failing . can log failure
              raise ValueError("Invalid data chunk format from wrapper."  # validate with basic data structture verification method for inputs 

      # dynamic selection of handler to convert , scale / perform complex feature processing and data conversion  of generic type, this will return a structured dict. as all nodes must pass data based on structered format. using metadata and type specified
        processing_unit = self.get_processing_unit(data_wrapper.data_type) # Get process chains (handler of a single or mult type) based upon requirements .
        if not processing_unit:
             logger.warning (f" Node {self.node_id} does not have a suitable processing unit for data type: {data_wrapper.data_type}") # logging system fail when certain types are passed into nodes when a given type processing can't exist within them or its path is unavlaible. this helps debug type inconsistencies if something breaks in process chains or if data set is non compatable for the core design intention. also used as simple way to filter for what info can influence this system/
             return None
   # resource usage calulated before every action using helper functions as stated to track operational capabilities to help guide selection, adaptation or reproduction processes for optimization for resources . so the resource is considered for before all logic functions of a node operation. this step if done before will stop processing flow or use less costly operations, based on dynamic requirements of resource management.
        energy_cost = self._calculate_energy_cost (data_wrapper)
        if not self._consume_energy (energy_cost): # Uses trait from dynamic properties to see if can handle task. with internal helper to update current states and flag false if fails

           return None  # stop if required resource not available for actions. for core node process , skipping subsequent steps. which has core data transormations

         # Create new task operations as a copy to not change shared variables but specific to the node and data under use, ensures thread safetey if it happens in threads
        async with self._lock:  # ThreadLock implemented  . so code in block does not produce thread error. all parts need safe read access with minimal lock durations if need threads in future versions 
            processed_data = await self.loop.run_in_executor ( # creates threads to allow concurrency at node operations using async
              self.executor, processing_unit.process, data_wrapper  # get data for operation specific module as selected by type


          # after process step perform integration to higher layers. This section implment and enforces all logic for this process step in one chain from data to end results in memory
          if processed_data: # skip steps below, using conditional execution of block for logic purposes ( as per python doc recommended to verify true status of operation rather than using global or state ).

              self.state.tasks_completed += 1 # count each run through operations to track how much activity each unit/ node goes though in current runtime of simulation. can be used downstream in different components . especially learning mechanisms.  can also include performance measures and timing data points to better identify resources usage or cost efficiency within
