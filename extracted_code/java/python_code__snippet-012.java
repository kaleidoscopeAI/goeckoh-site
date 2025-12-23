"""Demonstrate the unified system"""
print("Initializing Echo V4.0 System...")
system = EchoV4System()

print("Running simulation...\n")

for step in range(num_steps):
    # Simulate varying sensory input
    t = step / 10.0

    # Stress wave (simulates periodic challenges)
    stress_wave = 0.3 + 0.2 * np.sin(t)

    sensory_input = {
        'audio_rms': stress_wave,
        'text_sentiment': 0.5 + 0.3 * np.cos(t * 0.5),
        'external_arousal': 0.4 + 0.1 * np.sin(t * 2.0)
    }

    # Update system
    metrics = system.step(sensory_input)

    # Print status every 20 steps
    if step % 20 == 0:
        print(f"\n{system.get_summary()}")
        print(f"\nCurrent Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key:20s}: {value:+.4f}")
            else:
                print(f"  {key:20s}: {value}")

print("\n" + "=" * 60)
print("Simulation complete.")
print("\nLife Intensity Trajectory:")
life_vals = list(system.life_history)
for i, L in enumerate(life_vals[-20:]):
    bar = "█" * int(50 * (L + 1) / 2)  # Map [-1,1] to bar
    print(f"  t={len(life_vals)-20+i:3d}: {L:+.3f} {bar}")


