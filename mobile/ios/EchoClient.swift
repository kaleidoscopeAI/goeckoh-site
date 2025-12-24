//
//  EchoClient.swift
//  EchoMobile
//
//  Created by AI on 11/18/25.
//

import Foundation
import Starscream
import Combine

class EchoClient: NSObject, WebSocketDelegate, ObservableObject {
    var socket: WebSocket?
    
    @Published var isConnected = false
    @Published var logMessages = "Welcome to Echo Companion!\n"

    override init() {
        super.init()
    }

    func connect() {
        // Replace with your server's IP address
        let serverURL = URL(string: "ws://localhost:8765")!
        var request = URLRequest(url: serverURL)
        request.timeoutInterval = 5
        
        socket = WebSocket(request: request)
        socket?.delegate = self
        socket?.connect()
    }

    func disconnect() {
        socket?.disconnect()
    }

    func send(message: String) {
        socket?.write(string: message)
        log("Sent: \(message)")
    }

    func didReceive(event: WebSocketEvent, client: WebSocketClient) {
        switch event {
        case .connected(let headers):
            DispatchQueue.main.async {
                self.isConnected = true
                self.log("Connected to Echo Server!")
                self.log("Headers: \(headers)")
            }
        case .disconnected(let reason, let code):
            DispatchQueue.main.async {
                self.isConnected = false
                self.log("Disconnected: \(reason) with code: \(code)")
            }
        case .text(let string):
            log("Received: \(string)")
        case .binary(let data):
            log("Received \(data.count) bytes")
            // Here you would handle incoming audio data
        case .ping(_):
            break
        case .pong(_):
            break
        case .viabilityChanged(_):
            break
        case .reconnectSuggested(_):
            break
        case .cancelled:
            DispatchQueue.main.async {
                self.isConnected = false
                self.log("Connection Cancelled")
            }
        case .error(let error):
            DispatchQueue.main.async {
                self.isConnected = false
                self.log("Error: \(error?.localizedDescription ?? "Unknown error")")
            }
        case .peerClosed:
            log("Peer closed connection")
            break
        }
    }
    
    private func log(_ message: String) {
        DispatchQueue.main.async {
            self.logMessages.append("\(message)\n")
        }
    }
}