from transliterator import Transliterator


transliterator = Transliterator()

text = "namaste"

result = transliterator.roman_to_indic(
    text=text,
    language_code="hi",
)

print("Input :", text)
print("Output:", result)