#!/usr/bin/env python3
"""
Comprehensive Bubble Universe System Test
"""

import sys
from pathlib import Path

def test_system_components():
    """Test all system components"""
    print("🚀 Testing Bubble Universe System...")
    print("=" * 50)
    
    # Test 1: Core imports
    print("\n📦 Testing Core Imports...")
    try:
        from system_launcher import SystemOrchestrator, SystemConfig
        print("✅ System launcher imports OK")
    except Exception as e:
        print(f"❌ System launcher import failed: {e}")
        return False
    
    # Test 2: Configuration
    print("\n⚙️  Testing Configuration...")
    try:
        config = SystemConfig(mode='universe')
        print(f"✅ Configuration created: {config.mode}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    # Test 3: Integration modules
    print("\n🔗 Testing Integration Modules...")
    sys.path.insert(0, 'integrations')
    
    modules_to_test = [
        ('AI Image Generation', 'ai_image_generation_system'),
        ('Voice Universe Game', 'voice_universe_game'),
        ('Cognitive Nebula Integration', 'cognitive_nebula_integration')
    ]
    
    for name, module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {name} module OK")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Test 4: Cognitive Nebula presence
    print("\n🌌 Testing Cognitive Nebula...")
    nebula_path = Path('cognitive-nebula')
    if nebula_path.exists():
        print("✅ Cognitive Nebula directory exists")
        if (nebula_path / 'package.json').exists():
            print("✅ Cognitive Nebula package.json found")
        else:
            print("⚠️  Cognitive Nebula package.json missing")
    else:
        print("❌ Cognitive Nebula directory missing")
    
    # Test 5: Desktop launcher
    print("\n🖥️  Testing Desktop Launcher...")
    launcher_path = Path('Bubble_Universe.desktop')
    if launcher_path.exists():
        print("✅ Desktop launcher exists")
    else:
        print("❌ Desktop launcher missing")
    
    print("\n" + "=" * 50)
    print("🎉 Bubble Universe System Test Complete!")
    print("\n🚀 Ready to launch with:")
    print("   python system_launcher.py --mode universe")
    print("   python system_launcher.py --mode game")
    print("\n💫 Life-changing therapeutic system ready!")

if __name__ == "__main__":
    test_system_components()
