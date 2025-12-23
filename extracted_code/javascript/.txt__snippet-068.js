guard let v = value as? [String: Any] else { continue }
let attempts = v["attempts"] as? Int ?? 0
let corrections = v["corrections"] as? Int ?? 0
let rate = v["correction_rate"] as? Double ?? 0.0
