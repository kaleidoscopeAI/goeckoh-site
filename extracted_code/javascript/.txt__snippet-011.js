let task = URLSession.shared.dataTask(with: req) { [weak self] data, resp, error in
guard let self = self else { return }
if let error = error {
