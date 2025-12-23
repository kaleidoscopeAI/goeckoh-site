os.makedirs(path, exist_ok=True)
files = [f for f in os.listdir(path) if f.endswith('.txt')]
if not files:
    # create small sample docs
    samples = {
        'ai.txt': 'Artificial intelligence studies algorithms that learn from data.',
        'cognition.txt': 'Cognition explores perception, attention, memory, and reasoning.',
        'agent.txt': 'An agent observes, decides and acts in an environment to achieve goals.'
    }
    for name, text in samples.items():
        with open(os.path.join(path, name), 'w') as fh:
            fh.write(text)
    logging.info('Sample corpus created in ./corpus')

