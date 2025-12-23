• I'm planning to import matplotlib lazily inside the plotting check with try/
  except to suppress import errors caused by numpy mismatches, accepting some
  noisy stderr messages but avoiding script failure.

