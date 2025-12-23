        if (function_exists('pcntl_signal')) {

            pcntl_signal(SIGINT, [$this, 'signal_handler']);

            pcntl_signal(SIGTERM, [$this, 'signal_handler']);

        }

    }

    public function signal_handler(int $signum): void {

        echo "\nInterrupt signal ($signum) received. Shutting down gracefully...\n";

        $this->running = false;

    }

    public function step(): void {

        $text = 'Artificial Intelligence evolves';

        $vec = embed_text($text);

        $this->phi = $this->math->integrated_information($vec);

        if (!$this->conscious && $this->phi > 0.7) {

            $this->conscious = true;

            echo "Consciousness threshold reached: Φ={$this->phi}\n";

        }

        $this->dna->replicate();

        $this->memory->save_state($this->dna->generation, $this->phi);

        array_unshift($this->history, "Φ={$this->phi}");

        if (count($this->history) > 1000) array_pop($this->history);

    }

    public function run(): void {

        while ($this->running) {

            try {

                $this->step();

                echo "Tick: gen={$this->dna->generation}, Φ={$this->phi}, conscious=" . ($this->conscious ? 'true' : 'false') . "\n";

                sleep(1);

                if (function_exists('pcntl_signal_dispatch')) pcntl_signal_dispatch();

            } catch (Throwable $e) {

                error_log('Runtime error: ' . $e->getMessage());

                $this->running = false;

            }

        }

        echo "AGI Orchestrator stopped gracefully.\n";

    }

}

try {

