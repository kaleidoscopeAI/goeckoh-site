 9
10 -  ApiService(this._baseUrl);
10 +  // Use 10.0.2.2 for Android emulator by default; override as needed.
11 +  static const String defaultBase = 'http://10.0.2.2:8080';
12 +
13 +  ApiService([String? baseUrl]) : _baseUrl = baseUrl ?? defaultBase;
14

