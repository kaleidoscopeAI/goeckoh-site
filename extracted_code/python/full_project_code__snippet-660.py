from pip._vendor.pyparsing import Word, alphas

# define grammar of a greeting
greet = Word(alphas) + "," + Word(alphas) + "!"

hello = "Hello, World!"
print(hello, "->", greet.parse_string(hello))

