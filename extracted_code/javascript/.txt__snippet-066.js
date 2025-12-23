guard let self = self, let data = data else { return }
if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
