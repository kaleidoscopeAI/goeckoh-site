type OpenData = ();
type Data = Arc<Mutex<OmniMindState>>;

fn open(_context: &(), _file: &File) -> Result<Self::Data> {
    pr_info!("Cognitive Crystal AI OS device opened\n");
    Ok(Arc::new(Mutex::new(OmniMindState::new())))
}

fn read(
    _data: &Self::Data,
    _file: &File,
    writer: &mut impl IoBufferWriter,
    offset: u64,
) -> Result<usize> {
    if offset > 0 {
        return Ok(0);
    }

    let mut state = OMNIMIND_STATE.lock();
    state.last_update = ktime_get_ns();

    // Serialize for user-space: header + bit-packed metrics
    let mut buffer = [0u8; 1024];
    let mut pos = 0;

    buffer[pos..pos+8].copy_from_slice(&state.total_decisions.to_ne_bytes());
    pos += 8;
    let count = core::cmp::min(state.metrics_history.len(), 100);
    buffer[pos..pos+4].copy_from_slice(&(count as u32).to_ne_bytes());
    pos += 4;

    for metrics in state.metrics_history.iter().rev().take(count) {
        let packet = MetricsPacket {
            pid: metrics.pid,
            bit_state: metrics.bit_state,
            timestamp: metrics.timestamp,
        };
        let packet_bytes: [u8; core::mem::size_of::<MetricsPacket>()] = unsafe { core::mem::transmute(packet) };
        if pos + packet_bytes.len() > buffer.len() {
            break;
        }
        buffer[pos..pos + packet_bytes.len()].copy_from_slice(&packet_bytes);
        pos += packet_bytes.len();
    }

    let len = core::cmp::min(pos, writer.len());
    writer.write_slice(&buffer[..len])?;
    Ok(len)
}

fn write(
    _data: &Self::Data,
    _file: &File,
    reader: &mut impl IoBufferReader,
) -> Result<usize> {
    let mut buffer = [0u8; 1024];
    let len = reader.read_slice(&mut buffer)?;

    // Parse decision packets from user-space (L3 interventions, possibly from optimal control)
    let mut pos = 0;
    while pos + core::mem::size_of::<DecisionPacket>() <= len {
        let packet_bytes = &buffer[pos..pos + core::mem::size_of::<DecisionPacket>()];
        let packet: DecisionPacket = unsafe { core::mem::transmute_copy(packet_bytes) };
        pos += core::mem::size_of::<DecisionPacket>();

        if let Some(task) = Task::from_pid(packet.pid) {
            let task_ptr = task.as_ptr() as *mut task_struct;

            // Apply priority (L1 awareness A_i influence)
            let current_nice = unsafe { (*task_ptr).static_prio - MAX_RT_PRIO };
            let new_nice = current_nice + packet.priority_delta;
            unsafe { set_user_nice(task_ptr, new_nice.clamp(-20, 19)) };

            // IO class (L2 propagation)
            let ioprio = (packet.io_class as i32) << 13;
            unsafe { set_task_ioprio(task_ptr, ioprio) };

            // Memory reclaim (L0 field source term)
            if packet.memory_hint != 0 {
                let nr_pages = 256;  // ~1MB
                unsafe { try_to_free_mem_cgroup_pages((*task_ptr).mm, nr_pages, GFP_KERNEL, true) };
            }

            pr_info!(
                "Cognitive Crystal applied decision for PID {}: prio_delta={}, io_class={}, mem_hint={}, ts={}\n",
                packet.pid,
                packet.priority_delta,
                packet.io_class,
                packet.memory_hint,
                packet.timestamp
            );

            let mut state = OMNIMIND_STATE.lock();
            state.total_decisions += 1;
        }
    }

    Ok(len)
}
