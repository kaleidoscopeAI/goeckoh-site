• I discovered that although _to_jsonable handles complex types, the serialization error likely arises from nested complex
  numbers not fully converted before Flask's jsonify runs. I'll review the _to_jsonable implementation in detail to confirm and
  identify gaps causing this failure.

