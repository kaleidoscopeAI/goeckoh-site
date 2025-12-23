def _process_text_data (self, text_wrapper): # method designed to pass through core TextProccess logic, but leaves it open to be overridden at runtime, making nodes specialized without changes at module code level, making system very modular. 
  text = text_wrapper.get_data ()

 # update with any processed information via selected module.
  tfidf_data =  self.processing_units ["text"].process(text_wrapper)
