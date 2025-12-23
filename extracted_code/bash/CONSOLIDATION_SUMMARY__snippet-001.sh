mkdir -p legacy/deprecated_systems

# Move from GOECKOH location
mv GOECKOH/goeckoh/systems/unified_neuro_acoustic_system.py legacy/deprecated_systems/goeckoh_unified_neuro_acoustic_system.py
mv GOECKOH/goeckoh/systems/robust_unified_system.py legacy/deprecated_systems/goeckoh_robust_unified_system.py
mv GOECKOH/goeckoh/systems/enhanced_unified_system.py legacy/deprecated_systems/goeckoh_enhanced_unified_system.py

# Move from systems location
mv systems/unified_neuro_acoustic_system.py legacy/deprecated_systems/unified_neuro_acoustic_system.py
mv systems/robust_unified_system.py legacy/deprecated_systems/robust_unified_system.py
mv systems/enhanced_unified_system.py legacy/deprecated_systems/enhanced_unified_system.py

# Fix the misnamed file
rm systems/complete_unified_system.py  # or replace with correct content
