  """Extracts Concepts and runs reasoning modules to return results"""
  # Extracts concepts and adds nodes to the graph for memory based inference
  concepts = self.knowledge_graph.extract_concepts(input_data)

  #Adds the nodes identified into the memory of the knowledgde graph
  for concept in concepts:
    self.knowledge_graph.add_node(concept["id"], concept)

    # Applies the current reasoning modules based on what's available from this concept
  insights = self.reasoning_engine.apply (concepts[0] if len (concepts) > 0 else {})

  # Makes decisions on those inputs. for testing this does nothing as all steps will feed information forward sequentially
  decisions = self.decision_maker.evaluate (insights)
  self.belief_system.observe(insights)

  return {
      'insights': insights,
        'decisions': decisions,
        'updated_beliefs': self.belief_system.get_probabilities() # gets current network state probabilities to influence downstream operations
       }

