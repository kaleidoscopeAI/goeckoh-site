# protocol, hostname, port
# Taken from Chrome's list of secure origins (See: http://bit.ly/1qrySKC)
("https", "*", "*"),
("*", "localhost", "*"),
("*", "127.0.0.0/8", "*"),
("*", "::1/128", "*"),
("file", "*", None),
# ssh is always secure.
("ssh", "*", "*"),
