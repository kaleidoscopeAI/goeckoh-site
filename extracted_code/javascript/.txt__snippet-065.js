guard let baseURL = config.baseURL else { return }
let latestURL = baseURL.appendingPathComponent("api/latest")
let statsURL = baseURL.appendingPathComponent("api/stats")
