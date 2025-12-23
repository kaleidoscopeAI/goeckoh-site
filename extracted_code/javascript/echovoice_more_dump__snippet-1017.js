let task_ptr = task.as_ptr() as *mut task_struct;
if task_ptr.is_null() {
    continue;
}
// ... rest of code
