if let Some(task) = Task::from_pid(packet.pid) {
    let task_ptr = task.as_ptr() as *mut task_struct;
    if task_ptr.is_null() {
        continue;
    }
    // ... rest of code
