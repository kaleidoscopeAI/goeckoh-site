@StateObject private var client = EchoClient()
@State private var isRecording = false
@State private var recorder: AVAudioRecorder?
@State private var currentFileURL: URL?
