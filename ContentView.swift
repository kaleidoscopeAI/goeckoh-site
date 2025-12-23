//
//  ContentView.swift
//  EchoMobile
//
//  Created by AI on 11/18/25.
//

import SwiftUI
import Starscream

struct ContentView: View {
    @StateObject private var echoClient = EchoClient()

    var body: some View {
        VStack(spacing: 20) {
            Text("Echo Companion")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.blue)

            ScrollView {
                Text(echoClient.logMessages)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
            }
            .frame(height: 300)

            HStack {
                Button(action: {
                    if echoClient.isConnected {
                        echoClient.disconnect()
                    } else {
                        echoClient.connect()
                    }
                }) {
                    Text(echoClient.isConnected ? "Disconnect" : "Connect")
                        .padding()
                        .background(echoClient.isConnected ? Color.red : Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                
                Button(action: {
                    // This would be the record button
                    // For now, it sends a test message
                    echoClient.send(message: "{\"type\": \"test\", \"data\": \"Hello from mobile\"}")
                }) {
                    Image(systemName: "mic.fill")
                        .font(.largeTitle)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .clipShape(Circle())
                }
                .disabled(!echoClient.isConnected)
            }
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}