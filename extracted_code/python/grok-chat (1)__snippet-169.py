corrected = correct_to_first_person(input_text)
print(f"[Inner Voice Echo]: {corrected}")
# Sim arousal from length, stress from "error" (length diff)
arousal = len(input_text) / 20.0
agency_stress = abs(len(input_text) - len(corrected)) / 10.0
return corrected, arousal, agency_stress

