    def show_status(self):
        """Show system status"""
        status = self.get_system_status()
        
        print(f"\n📊 System Status:")
        print(f"  Uptime: {status['production_info']['uptime']:.1f}s")
        print(f"  Session: {status['production_info']['session_id']}")
        print(f"  User: {status['production_info']['current_user'] or 'None'}")
        print(f"  GCL: {status['gcl']:.3f}")
        print(f"  Mode: {status['system_mode']}")
        print(f"  Audio: {'🔴' if status['audio_status']['is_recording'] else '⚪'} Recording")
        print(f"  API: {'🟢' if status['api_status']['server_running'] else '⚪'} Server")
        print(f"  Requests: {status['performance_metrics']['total_requests']}")
        print(f"  Errors: {status['performance_metrics']['total_errors']}")
    
    def show_profile(self):
        """Show current user profile"""
        if not self.current_user:
            print("No active user profile")
            return
        
        profile = self.current_user
        print(f"\n👤 User Profile:")
        print(f"  Name: {profile.name}")
        print(f"  ID: {profile.user_id}")
        print(f"  Sessions: {len(profile.session_history)}")
        print(f"  Created: {datetime.fromtimestamp(profile.created_at).strftime('%Y-%m-%d %H:%M')}")
        print(f"  Last Active: {datetime.fromtimestamp(profile.last_active).strftime('%Y-%m-%d %H:%M')}")

