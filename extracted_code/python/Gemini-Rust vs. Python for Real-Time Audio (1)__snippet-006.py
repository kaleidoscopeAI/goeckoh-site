You now have the audio engine defined, but `lib.rs` (from the previous step) doesn't know about it yet. You need to update the `InnerState` struct in `lib.rs` to hold this `AudioStreamManager`.

