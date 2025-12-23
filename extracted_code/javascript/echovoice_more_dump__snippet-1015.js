if task_ptr.is_null() {
    return 0;
}

let task = unsafe { &*task_ptr };

let cpu_time_jiffies = unsafe { (*task_ptr).utime + (*task_ptr).stime };
let cpu_time_ms = jiffies_to_msecs(cpu_time_jiffies as u64);

let memory_pages = unsafe { get_mm_rss((*task_ptr).mm) };
let memory_kb = (memory_pages * (1 << PAGE_SHIFT) / 1024) as u64;

let io_reads = unsafe { (*task_ptr).ioac.read_bytes };
let io_writes = unsafe { (*task_ptr).ioac.write_bytes };
let io_total = io_reads + io_writes;

let prio = unsafe { (*task_ptr).static_prio - MAX_RT_PRIO };

// Pack into bit state for Crystal lattice node
let bit_state = pack_state(cpu_time_ms, memory_kb, io_total, prio, 0);  // Entropy computed next
let entropy = bit_entropy(bit_state);
let bit_state = pack_state(cpu_time_ms, memory_kb, io_total, prio, entropy);

let metrics = ProcessMetrics {
    pid: task.pid.as_raw(),
    bit_state,
    timestamp: ktime_get_ns(),
};

let mut state = OMNIMIND_STATE.lock();
let decision = state.decision_engine.analyze_process(&metrics);
state.last_update = metrics.timestamp;

if state.metrics_history.len() >= 1000 {
    state.metrics_history.remove(0);
}
state.metrics_history.push(metrics);
state.total_decisions += 1;

// Apply immediate bit-level decisions
let task_ptr_mut = task_ptr;
let current_nice = prio;
let new_nice = current_nice + decision.priority_adjustment;
unsafe { set_user_nice(task_ptr_mut, new_nice.clamp(-20, 19)) };

let ioprio_class = match decision.io_class {
    IoClass::RealTime => 1,
    IoClass::BestEffort => 2,
    IoClass::Idle => 3,
};
let ioprio = ioprio_class << 13;
unsafe { set_task_ioprio(task_ptr_mut, ioprio) };

if let MemoryAction::Reclaim = decision.memory_action {
    let nr_pages = 256;
    unsafe { try_to_free_mem_cgroup_pages((*task_ptr_mut).mm, nr_pages, GFP_KERNEL, true) };
}

let (cpu, mem, io, prio_un, entropy_un) = unpack_state(bit_state);
pr_info!(
    "Cognitive Crystal bit-analyzed PID {}: bit_state={:016x}, cpu={} mem={} io={} prio={} entropy={} | Dec: prio_adj={}, io_class={:?}, mem_act={:?}\n",
    metrics.pid,
    bit_state,
    cpu,
    mem,
    io,
    prio_un,
    entropy_un,
    decision.priority_adjustment,
    decision.io_class,
    decision.memory_action
);

decision.priority_adjustment
