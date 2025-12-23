  insights = self.pattern_recognizer._extract_text_patterns(text) # basic function that takes simple data as inputs
  # return all found details and analysis related to these systems for storage downstream or more advanced engine based data extractions etc..

  return insights ,tfidf_data


def _simulate_node_interaction(self, other_node: 'BaseNode'):
    """Simulates interaction between nodes, with enhanced exchange using both text and visual data from graphs and with dynamic behaviour. 
       Returns : does not return info as its self updated node specific data
   """

   # select interactions by trait. higher values equal more frequent interaction (more traffic) between specific type
  knowledge_transfer_opportunity = random.random() *  self.traits.get_trait_value('communication')  # scale from 0 ->1 to influence more or less transfer

  if (knowledge_transfer_opportunity > 0.5 and other_node) and not "type" in other_node.internal_state :
   logger.info (f" Node : {self.node_id} started interaction with Node : {other_node.node_id} ")

        # exchange knowledge based on type. If either exists
   if 'knowledge_graph' in self and self.knowledge_graph and  'knowledge_graph' in other_node and  other_node.knowledge_graph: # added safety net and prevents broken behaviour where methods not implamented correctly during setup. 
      new_insights= random.choice(list (self.knowledge_graph.nodes (data = True)))[1] if random.random() < 0.5  and  self.knowledge_graph.number_of_nodes()> 0  else {}  # add choice of either node with higher weights to provide variability for self contained functions, but also to enforce rule following if required by external config systems. for complex multi variable and self regulated processes

      other_node.add_knowledge ("concept",  new_insights ) # use internal method to inject directly with helper,  to control and track process easily for debug, validation or monitoring.

       # Update the interaction based on success
      logger.debug (f" Node {self.node_id} exchanged {knowledge_transfer_opportunity} knowledge to Node {other_node.node_id}.")  # verbose output so users have understanding if certain methods were exectuted as expected and debug for later iterations as requirements evolve.

