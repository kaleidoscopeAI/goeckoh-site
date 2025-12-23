let path = format!(\"/sys/devices/system/cpu/cpu{}/cpufreq/scaling_setspeed\", core_id);
if let Err(e) = try_write_sysfs(&path, (freq_mhz as u64 * 1000).to_string()) {
